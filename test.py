
import os
import hydra
from omegaconf import DictConfig
from train import SemsegGuidedModule
from utils import save_tensors_as_images
from pathlib import Path
import datetime  # Import the datetime module

@hydra.main(config_path="config", config_name="base", version_base="1.3")
def main(cfg: DictConfig):
    gpu_index = cfg.training.gpus[0]  # Use the first GPU in the list
    device = f'cuda:{gpu_index}'
    
    ckpt_path = cfg.model.checkpoint_path
    original_run_dir = Path(ckpt_path).parents[1]

    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    images_dir = original_run_dir / f'generations/inference_{timestamp}'  # Append timestamp to folder name
    os.makedirs(images_dir, exist_ok=True)

    # Load model from checkpoint
    model = SemsegGuidedModule.load_from_checkpoint(ckpt_path, map_location=device).to(device)
    model.eval()
    model.cfg = cfg

    # Inference and image saving
    input_semsegs = cfg.dataset.test_folder
    image_extensions = ['.jpg', '.jpeg', '.png']
    all_files = os.listdir(input_semsegs)
    semseg_paths = sorted([os.path.join(input_semsegs, file) for file in all_files if os.path.splitext(file)[1].lower() in image_extensions])

    print(f'Look for outputs at {images_dir}')
    for idx,test_mask_path in enumerate(semseg_paths):
        images_ema, _ = model.infer_noise(model.ema_model,test_mask_path=test_mask_path)
        _ = save_tensors_as_images(images_ema, str(images_dir), f'{test_mask_path.split("/")[-1].split(".")[0]}')


if __name__ == "__main__":
    # Override hydra's output directory behavior before running main
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(config_path="config", job_name="base", version_base="1.3")
    cfg = hydra.compose(config_name="base", overrides=["hydra.output_subdir=null"])
    
    main(cfg)