
import random
import numpy as np
import torch
import yaml
import os
from PIL import Image
import torchvision
from torchvision.transforms import ToPILImage

def load_yaml(yaml_file):
    with open(yaml_file, 'r') as f:
        return yaml.safe_load(f)


def save_tensors_as_images(images , path, groupname, **kwargs):
    images = images.cpu()
    images = (images.clamp(-1, 1) + 1) / 2
    images = (images * 255).type(torch.uint8)

    # save montage
    os.makedirs(path, exist_ok=True)
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').detach().numpy()
    im = Image.fromarray(ndarr)
    im.save(path+'/'+str(groupname)+'.png')

    # separate elements of montage
    epoch_folder = os.path.join(path, f"epoch_{groupname}")
    os.makedirs(epoch_folder, exist_ok=True)
    for idx, image in enumerate(images):
        ndarr = image.permute(1, 2, 0).to('cpu').detach().numpy()
        im = Image.fromarray(ndarr)        
        im.save(os.path.join(epoch_folder, f"image_{idx}.png"))

    return images # returns rgb image

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
