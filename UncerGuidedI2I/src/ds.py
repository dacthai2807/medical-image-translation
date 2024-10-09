from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import numpy as np
import cv2 as cv
import torch
import os
from scipy.interpolate import interp1d

class PETCTDataset(Dataset):
    def __init__(self, ct_paths, pet_paths, image_size=(256, 256), ct_max_pixel=255.0, pet_max_pixel=255.0, flip=False, stage='train'):
        self.image_size = image_size
        self.ct_paths = ct_paths
        self.pet_paths = pet_paths
        self._length = len(ct_paths)
        self.ct_max_pixel = float(ct_max_pixel)
        self.pet_max_pixel = float(pet_max_pixel)
        self.flip = flip
        self.stage = stage
        
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
            np_ct_image = np.load(ct_path, allow_pickle=True)[0]
            # np_ct_image = np_ct_image / float(self.ct_max_pixel)

            ct_image = Image.fromarray(np_ct_image) 
            ct_image = transform(ct_image) 

            np_pet_image = np.load(pet_path, allow_pickle=True)[0]
            # np_pet_image = np_pet_image / float(self.pet_max_pixel)
            
            pet_image = Image.fromarray(np_pet_image) 
            pet_image = transform(pet_image) 
            
            # pet_image = (pet_image - 0.5) * 2.
            
        except BaseException as e:
            print(ct_path)
            print(pet_path)
        
        image_name = Path(ct_path).stem
        
        # att_map = self.get_attention_map(image_name, ct_image, self.stage)
        # atte_map = self.get_attenuation_map(ct_image, ct_image)
        
        # ct_image = torch.cat([ct_image, att_map, atte_map], dim=0)

        return ct_image, pet_image, image_name
    
    def attenuationCT_to_511(self, KVP, reresized):
        # values from: Accuracy of CT-based attenuation correction in PET/CT bone imaging
        if KVP == 100:
            a = [9.3e-5, 4e-5, 0.5e-5]
            b = [0.093, 0.093, 0.128]
        elif KVP == 80:
            a = [9.3e-5, 3.28e-5, 0.41e-5]
            b = [0.093, 0.093, 0.122]
        elif KVP == 120:
            a = [9.3e-5, 4.71e-5, 0.589e-5]
            b = [0.093, 0.093, 0.134]
        elif KVP == 140:
            a = [9.3e-5, 5.59e-5, 0.698e-5]
            b = [0.093, 0.093, 0.142]
        else:
            print('Unsupported kVp, interpolating initial values')
            a1 = [9.3e-5, 3.28e-5, 0.41e-5]
            b1 = [0.093, 0.093, 0.122]
            a2 = [9.3e-5, 4e-5, 0.5e-5]
            b2 = [0.093, 0.093, 0.128]
            a3 = [9.3e-5, 4.71e-5, 0.589e-5]
            b3 = [0.093, 0.093, 0.134]
            a4 = [9.3e-5, 5.59e-5, 0.698e-5]
            b4 = [0.093, 0.093, 0.142]
            aa = np.array([a1, a2, a3, a4])
            bb = np.array([b1, b2, b3, b4])
            c = np.array([80, 100, 120, 140])
            a = np.zeros(3)
            b = np.zeros(3)
            for kk in range(3):
                a[kk] = np.interp(KVP, c, aa[:, kk])
                b[kk] = np.interp(KVP, c, bb[:, kk])

        z = np.array([[-1000, b[0] - 1000 * a[0]],
                    [0, b[1]],
                    [1000, b[1] + a[1] * 1000],
                    [3000, b[1] + a[1] * 1000 + a[2] * 2000]])

        tarkkuus = 0.1
        vali = np.arange(-1000, 3000 + tarkkuus, tarkkuus)
        inter = interp1d(z[:, 0], z[:, 1], kind='linear', fill_value='extrapolate')(vali)

        # trilinear interpolation
        attenuation_factors = np.interp(reresized.flatten(), vali, inter).reshape(reresized.shape)
        
        return attenuation_factors
    
    def get_attention_map(self, x_name, x_cond_latent, stage):    
        attention_map_path = '/home/PET-CT/thaind/ckpts/Unet_resnet50/lightning_logs/version_0/samples/test'
        
        if stage == 'train':
            attention_map_path = '/home/PET-CT/thaind/ckpts/Unet_resnet50/lightning_logs/version_0/samples/train'
        
        np_cond = np.load(os.path.join(attention_map_path, f'{x_name}.npy'), allow_pickle=True)
        np_cond = cv.resize(np_cond, (x_cond_latent.shape[1], x_cond_latent.shape[2]))

        tensor = torch.from_numpy(np_cond).unsqueeze(0)
        
        return tensor.to(x_cond_latent.device) 
    
    def get_attenuation_map(self, x_cond, x_cond_latent):  
        rescale_slope = 1.
        rescale_intercept = -1024.

        np_x_cond = x_cond.squeeze(0).cpu().numpy()
        np_x_cond = np_x_cond * 2047.
        HU_map = np_x_cond * rescale_slope + rescale_intercept

        KVP = 140  # Giá trị kVp
        attenuation_factors = self.attenuationCT_to_511(KVP, HU_map)
        attenuation_factors = np.exp(-attenuation_factors)
        # attenuation_factors = 1 - attenuation_factors
        
        transform = transforms.Compose([
            # transforms.Resize((x_cond_latent.shape[2], x_cond_latent.shape[3])),
            transforms.ToTensor()
        ])

        attenuation_factors = Image.fromarray(attenuation_factors)
        attenuation_factors = transform(attenuation_factors)
            
        return attenuation_factors.to(x_cond_latent.device) 
      