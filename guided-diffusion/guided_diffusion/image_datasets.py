import math
import random
import os
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset


def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader

def load_PETnCT_data(
    pet_root, ct_root, pet_flist, ct_flist,
    batch_size,
    class_cond=False,
    deterministic=False,
):
    dataset = PETnCTDataset(pet_root, ct_root, pet_flist, ct_flist, class_cond) 
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader

def load_PET_data(
    pet_root, pet_flist,
    batch_size,
    deterministic=False,
):
    dataset = PETDataset(pet_root, pet_flist) 
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader

def load_CT_data(
    ct_root, ct_flist,
    batch_size,
    deterministic=False,
):
    dataset = CTDataset(ct_root, ct_flist) 
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader

class PairedPETnCTDataset(Dataset):
    def __init__(self, ct_paths, pet_paths, image_size=(256, 256), ct_max_pixel=255.0, pet_max_pixel=255.0, flip=False, class_cond=True):
        self.image_size = image_size
        self.ct_paths = ct_paths
        self.pet_paths = pet_paths
        self._length = len(ct_paths) + len(pet_paths)
        self.ct_max_pixel = float(ct_max_pixel)
        self.pet_max_pixel = float(pet_max_pixel)
        self.flip = flip
        self.class_cond = class_cond
        
    def __len__(self):
        return self._length

    def __getitem__(self, index):
        if self.flip:
            p = 0.5 
        else:
            p = 0.

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=p),
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])
        
        if index < len(self.ct_paths):
            ct_path = self.ct_paths[index]

            try:
                np_ct_image = np.load(ct_path, allow_pickle=True)
                np_ct_image = np_ct_image / float(self.ct_max_pixel)

                ct_image = Image.fromarray(np_ct_image) 
                ct_image = transform(ct_image) 
                
            except BaseException as e:
                print(ct_path)
            
            image_name = Path(ct_path).stem
            
            out_dict = {}
            
            if self.class_cond == True:
                ct_lab = 0 
                out_dict["y"] = np.array(ct_lab, dtype=np.int64)
            
            return ct_image, out_dict, image_name
        
        pet_path = self.pet_paths[index - len(self.ct_paths)]

        try:
            np_pet_image = np.load(pet_path, allow_pickle=True)
            np_pet_image = np_pet_image / float(self.pet_max_pixel)
            
            pet_image = Image.fromarray(np_pet_image) 
            pet_image = transform(pet_image) 
            
            pet_image = (pet_image - 0.5) * 2.
            
        except BaseException as e:
            print(pet_path)
        
        image_name = Path(pet_path).stem
        
        out_dict = {}
        
        if self.class_cond == True:
            pet_lab = 1
            out_dict["y"] = np.array(pet_lab, dtype=np.int64)
        
        return pet_image, out_dict, image_name

class PETnCTDataset(Dataset):
    '''
    can act as supervised or un-supervised based on flists
    '''
    def __init__(self, pet_root, ct_root, pet_flist, ct_flist, class_cond):
        self.pet_root = pet_root
        self.ct_root = ct_root
        self.pet_flist = pet_flist
        self.ct_flist = ct_flist
        self.class_cond = class_cond
    def __getitem__(self, index):
        if index < len(self.pet_flist):
            impath = self.pet_flist[index]
            img = np.load(os.path.join(self.pet_root, impath))
            lab = 1
        else:
            impath = self.ct_flist[index - len(self.pet_flist)]
            img = np.load(os.path.join(self.ct_root, impath))
            lab = 0
        # crop to 128x128
        from skimage.transform import resize
        img = resize(img, (128, 128))
        # augment
        if random.random() < 0.5:
            img = np.fliplr(img)
        # normalize
        img = (img - img.min()) / (img.max() - img.min())
        img = 2 * img - 1
        img = img[np.newaxis, :]
        img = np.tile(img, (3, 1, 1))
        img = img.astype(np.float32)
        out_dict = {}
        if self.class_cond == True:
            out_dict["y"] = np.array(lab, dtype=np.int64)
        return img, out_dict
    def __len__(self):
        return len(self.pet_flist + self.ct_flist) 

class PETDataset(Dataset):
    '''
    can act as supervised or un-supervised based on flists
    '''
    def __init__(self, pet_root, pet_flist, do_aug=True):
        self.pet_root = pet_root
        self.pet_flist = pet_flist
        self.do_aug = do_aug
    def __getitem__(self, index):
        impath = self.pet_flist[index]
        img = np.load(os.path.join(self.pet_root, impath))
        # crop to 128x128
        from skimage.transform import resize
        img = resize(img, (128, 128))
        # augment
        if self.do_aug == True:
            if random.random() < 0.5:
                img = np.fliplr(img)
        # normalize
        img = (img - img.min()) / (img.max() - img.min())
        img = 2 * img - 1
        img = img[np.newaxis, :]
        img = np.tile(img, (3, 1, 1))
        img = img.astype(np.float32)
        
        out_dict = {}
        return img, out_dict
    def __len__(self):
        return len(self.pet_flist) 

class CTDataset(Dataset):
    '''
    can act as supervised or un-supervised based on flists
    '''
    def __init__(self, ct_root, ct_flist, do_aug=True):
        self.ct_root = ct_root
        self.ct_flist = ct_flist
        self.do_aug = do_aug
    def __getitem__(self, index):
        impath = self.ct_flist[index]
        img = np.load(os.path.join(self.ct_root, impath))
        # crop to 128x128
        from skimage.transform import resize
        img = resize(img, (128, 128))
        # augment
        if self.do_aug == True:
            if random.random() < 0.5:
                img = np.fliplr(img)
        # normalize
        img = (img - img.min()) / (img.max() - img.min())
        img = 2 * img - 1
        img = img[np.newaxis, :]
        img = np.tile(img, (3, 1, 1))
        img = img.astype(np.float32)
        
        out_dict = {}
        return img, out_dict
    def __len__(self):
        return len(self.ct_flist) 

def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
