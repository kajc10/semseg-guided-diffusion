import copy
import os
import random
import numpy as np
import torch
import torchvision
from pytorch_lightning import LightningModule, LightningDataModule
from network import SemsegUNet, EMA , SemsegUNetAttentionReduced
from dataset import OneHotEncode, SemSegDataset
from diffusion import Diffusion
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import json
import hydra
from omegaconf import DictConfig
import wandb
from PIL import Image
import gc
from torch.optim.lr_scheduler import StepLR  # or any other 
from utils import save_tensors_as_images, set_seed

import torch.distributed as dist



class SemsegGuidedDataModule(LightningDataModule):
    def __init__(self, cfg, onehotencoder):
        super().__init__()
        self.cfg = cfg
        self.onehotencoder = onehotencoder

    def setup(self, stage=None):
        image_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        mask_transform = torchvision.transforms.Compose([])   

        self.dataset = SemSegDataset(
            image_folder=self.cfg.dataset.image_folder_path, 
            semseg_folder=self.cfg.dataset.semseg_folder_path, 
            colormap_path=self.cfg.dataset.colormap_path, 
            image_transform=image_transform,
            onehotencoder=self.onehotencoder,
            mask_transform=mask_transform
        )

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.cfg.training.batch_size, shuffle=True, drop_last=True, num_workers=8)

class SemsegGuidedModule(LightningModule):
    def __init__(self, cfg, onehotencoder):
        super().__init__()
        self.cfg = cfg
        if self.cfg.model.attention:
            self.model = SemsegUNetAttentionReduced(c_in=3, c_out=3, hidden_dim=cfg.model.hidden_dim, emb_dim=cfg.model.emb_dim, num_classes=cfg.model.num_classes)
        else:
            self.model = SemsegUNet(c_in=3, c_out=3, hidden_dim=cfg.model.hidden_dim, emb_dim=cfg.model.emb_dim, num_classes=cfg.model.num_classes)
        self.ema, self.ema_model = (EMA(0.995), copy.deepcopy(self.model).eval().requires_grad_(False)) if cfg.training.ema else (None, None)
        self.diffusion = Diffusion(noise_steps=cfg.algo.noise_steps)    
        self.criterion = torch.nn.MSELoss()
        self.onehotencoder = onehotencoder
        self.save_hyperparameters()

    def forward(self, x, t, y):
        return self.model(x, t, y)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        t = self.diffusion.sample_timesteps(images.shape[0]).cpu()  
        x_t, noise = self.diffusion.noise_images(images, t)
        if np.random.random() < 0.1: #cfg, 10% of the time, use uncond  #TODO it s only for a batch
            labels = None
        predicted_noise  = self(x_t, t, labels)
        loss = self.criterion(predicted_noise, noise)
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', current_lr, on_step=False, on_epoch=True)        
        return loss


    def on_train_batch_end(self, *args, **kwargs):
        if self.ema:
            self.ema.step_ema(self.ema_model, self.model)

    def on_train_epoch_end(self, outputs=None):
        if self.current_epoch % self.cfg.training.log_freq == 0:
            if self.trainer.global_rank == 0:                       # or # if self.trainer.is_global_zero:
                print('generating images on gpu rank (start from 0 on each training) ', self.trainer.global_rank)
                self.log_images(self.current_epoch)

        if self.current_epoch % self.cfg.training.save_freq == 0 and self.current_epoch >= self.cfg.training.min_log_epoch: # TODO discuss
            model_checkpoint_path = os.path.join(self.cfg.paths.ckpt_dir, f"model_epoch_{self.current_epoch}.ckpt")
            self.trainer.save_checkpoint(model_checkpoint_path)

            if self.ema_model is not None:
                ema_checkpoint_path = os.path.join(self.cfg.paths.ckpt_dir, f"ema_model_epoch_{self.current_epoch}.ckpt")
                torch.save(self.ema_model.state_dict(), ema_checkpoint_path)


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.training.lr)
        scheduler = StepLR(optimizer, step_size=1000, gamma=1.0) 
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'train_loss',  # Optional: specify a metric to monitor for scheduling (if your scheduler requires this)
            }
        }
    
    # -------------------------------------
    def log_images(self, epoch='infer', to_wandb=True, save_iim=False):
        print(f"Inferring images at {epoch} epoch, for main model{' and EMA model' if self.ema_model else ''}")
        raw_tensors_main, _ = self.infer_noise(self.model) # noramlised tensors
        raw_tensors_ema, _ = self.infer_noise(self.ema_model) if self.ema_model else (None, None)

        for img_set, model_name in zip([raw_tensors_main, raw_tensors_ema], ["main", "ema"]):
            if img_set is not None:
                infered_images = save_tensors_as_images(img_set, self.cfg.paths.img_dir, f'{epoch}_{model_name}') #normalised tensors to pil image
                if to_wandb:
                    self.logger.experiment.log(
                        {f"generated_images_{model_name}": [wandb.Image(infered_images.to(torch.float32) / 255.0, caption=f"Generated images for epoch {epoch}")]}
                    )

    def infer_noise(self,model,test_mask_path=None):
        if test_mask_path is None:test_mask_path = self.cfg.dataset.test_mask_path
        cfg = self.cfg
        random_noise = torch.randn((cfg.training.test_sample_num, 3, cfg.dataset.image_size, cfg.dataset.image_size), device=self.device)
        rgb_mask = Image.open(test_mask_path).convert('RGB')
        one_hot_mask = self.onehotencoder(rgb_mask).to(self.device)     # to!
        one_hot_mask = one_hot_mask.permute(2, 0, 1)                    
        labels = one_hot_mask.repeat(cfg.training.test_sample_num, 1, 1, 1) 

        images, iim = self.diffusion.denoise(model, 
                                            batch_size=self.cfg.training.test_sample_num,   
                                            x=random_noise, 
                                            labels=labels, 
                                            cfg_scale=self.cfg.algo.cfg_scale, 
                                            log_freq=self.cfg.training.log_freq
                                            )
        return images, iim



@hydra.main(config_path="config", config_name="base", version_base="1.3") #config/base.yaml
def main(cfg: DictConfig):
    os.makedirs(cfg.paths.log_dir, exist_ok=True)
    os.makedirs(cfg.paths.img_dir, exist_ok=True)
    os.makedirs(cfg.paths.ckpt_dir, exist_ok=True)

    set_seed(10) 

    onehotencoder = OneHotEncode(json.load(open(cfg.dataset.colormap_path, 'r')))

    data_module = SemsegGuidedDataModule(cfg, onehotencoder)
    module = SemsegGuidedModule(cfg, onehotencoder)
    logger = WandbLogger(name=cfg.run_name, project=cfg.training.wandb_project, mode=cfg.training.wandb_mode, save_dir=cfg.paths.log_dir)
    checkpoint_callback = ModelCheckpoint(dirpath=cfg.paths.ckpt_dir, save_top_k=3, mode='min',monitor='train_loss')

    ckpt_path = cfg.model.checkpoint_path if cfg.model.checkpoint_path is not None and os.path.isfile(cfg.model.checkpoint_path) else None
    trainer = Trainer(max_epochs=cfg.training.epochs, logger=logger, callbacks=[checkpoint_callback], 
                      devices=cfg.training.gpus, strategy='auto', accelerator='gpu', log_every_n_steps=100,
                      precision=32) #TODO discuss precision "16-mixed" / 32
                      #'ddp_find_unused_parameters_true' if len(cfg.training.gpus)>1 else 'auto'

    trainer.fit(module, data_module, ckpt_path=ckpt_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        torch.cuda.empty_cache()
        gc.collect()
        wandb.finish()
        print("wandb cleanup complete")

