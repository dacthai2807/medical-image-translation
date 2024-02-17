from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import numpy as np
import cv2 as cv

class PETCTDataset(Dataset):
    def __init__(self, ct_paths, pet_paths, image_size=(256, 256), ct_max_pixel=255.0, pet_max_pixel=255.0, flip=False):
        self.image_size = image_size
        self.ct_paths = ct_paths
        self.pet_paths = pet_paths
        self._length = len(ct_paths)
        self.ct_max_pixel = float(ct_max_pixel)
        self.pet_max_pixel = float(pet_max_pixel)
        self.flip = flip
        
    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length

    def __getitem__(self, index):
        p = 0.0
        
        if index >= self._length:
            index = index - self._length
            p = 1.0

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=p),
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])
        
        ct_path = self.ct_paths[index]
        pet_path = self.pet_paths[index]
        
        try:
            np_ct_image = np.load(ct_path, allow_pickle=True)
            np_ct_image = np_ct_image / float(self.ct_max_pixel)

            ct_image = Image.fromarray(np_ct_image) 
            ct_image = transform(ct_image) 

            np_pet_image = np.load(pet_path, allow_pickle=True)
            np_pet_image = np_pet_image / float(self.pet_max_pixel)
            
            pet_image = Image.fromarray(np_pet_image) 
            pet_image = transform(pet_image) 
            
            pet_image = (pet_image - 0.5) * 2.
            
        except BaseException as e:
            print(ct_path)
            print(pet_path)
        
        image_name = Path(ct_path).stem
        
        return ct_image, pet_image, image_name
      