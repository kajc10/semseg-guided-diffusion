import json
import os
import time
from PIL import Image
from torch.utils.data import Dataset
import torchvision
import torch
from torch.utils.data import DataLoader
import numpy as np
from torchvision.transforms import ToPILImage
from utils import set_seed
from torchvision.utils import make_grid

set_seed(10)

class OneHotEncode:
    def __init__(self, colormap):
        self.colormap = colormap
        self.num_classes = len(self.colormap)
        self.ID_TO_RGB = list(self.colormap.values())
        self.color_dict = {i: tuple(color) for i, color in enumerate(self.ID_TO_RGB)}

    def __call__(self, mask):
        one_hot_mask = self.rgb_to_onehot(np.array(mask), self.color_dict)
        return torch.tensor(one_hot_mask).float()

    @staticmethod
    def rgb_to_onehot(rgb_arr, color_dict):
        num_classes = len(color_dict)
        shape = rgb_arr.shape[:2] + (num_classes,)
        arr = np.zeros(shape, dtype=np.int8)
        for i, color in color_dict.items():
            mask = np.all(rgb_arr == color, axis=-1)
            arr[:, :, i] = mask
        return arr

    def onehot_to_rgb(self, onehot):
        single_layer = torch.argmax(onehot, dim=-1).numpy()
        output = np.zeros(single_layer.shape + (3,), dtype=np.uint8)
        for k, color in self.color_dict.items():
            output[single_layer == k] = color
        return output


class SemSegDataset(Dataset):
    '''
     Images and corresponding semsegs in seperate folders
    '''
    def __init__(self, image_folder, semseg_folder, colormap_path, image_transform, onehotencoder, mask_transform=None):
        self.image_folder = image_folder
        self.semseg_folder = semseg_folder
        self.image_transform = image_transform
        self.onehotencoder = onehotencoder 
        self.image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg')])

        with open(colormap_path, 'r') as f:
            self.colormap = json.load(f)

        self.onehotencoder = OneHotEncode(self.colormap) 
        print('### Using SemsSegDataset, gonna return images and one-hot masks ###')

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.image_files[idx])
        image = Image.open(image_path).convert('RGB')
        
        semseg_filename = self.image_files[idx].replace('.jpg', '.png')
        semseg_path = os.path.join(self.semseg_folder, semseg_filename)
        semseg_mask = Image.open(semseg_path)

        if self.image_transform:
            image = self.image_transform(image)

        one_hot_mask = self.onehotencoder(semseg_mask)
        one_hot_mask = one_hot_mask.permute(2, 0, 1) # permute is done!!

        return image, one_hot_mask #, semseg_mask_tensor  return b,c,h,w

    


