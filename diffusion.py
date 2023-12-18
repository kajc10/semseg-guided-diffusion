import numpy as np
import torch
from tqdm import tqdm

# B: 0.0001 - 0.02
# alphat = 1-Bt
# alpha_hatt = cumprodalpha #alpha0*alpha1*alpha2...alphan
# t = 1000
class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.cosine_noise_schedule()
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        print(f'\n ### Diffusion module inited with {noise_steps} noise_steps')

    def cosine_noise_schedule(self):
        t = torch.linspace(0, 1, self.noise_steps) 
        return reversed(self.beta_start + (self.beta_end - self.beta_start) * (1 + torch.cos(torch.tensor(np.pi) * t)) / 2)

    def noise_images(self, x0, t):  
        device = x0.device  # Use the device of the input tensor
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t].to(device))[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t].to(device))[:, None, None, None]
        e = torch.randn_like(x0)
        return sqrt_alpha_hat * x0 + sqrt_one_minus_alpha_hat * e, e   #return both noised img and noise!

    def sample_timesteps(self, batch_size):
        return torch.randint(low=1, high=self.noise_steps, size=(batch_size,))
    
    @torch.inference_mode()
    def denoise(self, model, batch_size, x, labels, cfg_scale=3,log_freq=100): 
        device = x.device
        beta, alpha, alpha_hat = self.beta.to(device), self.alpha.to(device), self.alpha_hat.to(device)
        i_list = torch.arange(self.noise_steps-1, -1, -1) #999...0
        model.eval() # Model should already be on the correct device
        intermediate_images = []
        with torch.no_grad():
            for i in tqdm(i_list): 
                t = (torch.ones(batch_size) * i).long().to(device) #timestep, bsize times 999...0
                predicted_noise = model(x, t, labels) #can passs None as well
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alphai = alpha[t][:, None, None, None]
                alpha_hati = alpha_hat[t][:, None, None, None]
                betai = beta[t][:, None, None, None]
                if i > 0: #if not last, add noise
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alphai) * (x - ((1 - alphai) / (torch.sqrt(1 - alpha_hati))) * predicted_noise) + torch.sqrt(betai) * noise #renoise
                if i%log_freq == 0:
                    intermediate_images.append(x)
        model.train()
        return x, intermediate_images #return denoised images(batch) and intermediate image logs