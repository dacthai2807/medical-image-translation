import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from ds import *
from trainer import *
from losses import *
from networks import *
from torch.utils.data import DataLoader

DATA_PATH = '/home/PET-CT/splited_data_15k'
IMAGE_SIZE = 256
CT_MAX = 2047
PET_MAX = 32767
BATCH_SIZE = 16

def get_image_paths_from_dir(fdir):
    flist = os.listdir(fdir)
    flist.sort()
    image_paths = []
    for i in range(0, len(flist)):
        fpath = os.path.join(fdir, flist[i])
        if os.path.isdir(fpath):
            image_paths.extend(get_image_paths_from_dir(fpath))
        else:
            image_paths.append(fpath)
    return image_paths

def get_dataset_by_stage(data_path, stage, image_size, ct_max_pixel, pet_max_pixel, flip):
    ct_paths = get_image_paths_from_dir(os.path.join(data_path, f'{stage}/A'))
    pet_paths = get_image_paths_from_dir(os.path.join(data_path, f'{stage}/B'))

    return PETCTDataset(ct_paths, pet_paths, image_size, ct_max_pixel, pet_max_pixel, flip, stage)

def main():
    train_dataset = get_dataset_by_stage(DATA_PATH, 'train', (IMAGE_SIZE, IMAGE_SIZE), CT_MAX, PET_MAX, True)
    val_dataset = get_dataset_by_stage(DATA_PATH, 'test', (IMAGE_SIZE, IMAGE_SIZE), CT_MAX, PET_MAX, False)
    test_dataset = get_dataset_by_stage(DATA_PATH, 'test', (IMAGE_SIZE, IMAGE_SIZE), CT_MAX, PET_MAX, False)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16)
    valid_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=16)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=16)

    # init net and train
    # netG_A = CasUNet(1,1)
    netG_A = UNet(3,1)
    netD_A = NLayerDiscriminator(1, n_layers=4)
    netG_A, netD_A = train_I2I_CasUNetGAN(
        netG_A, netD_A,
        train_dataloader, valid_dataloader,
        dtype=torch.cuda.FloatTensor,
        device='cuda',
        num_epochs=30,
        init_lr=1e-4,
        ckpt_path='../ckpt_wacv_2/conditional_CT2PET_UNet',
    ) 

    # init net and train
    # netG_A = CasUNet_3head(1,1)
    # netG_A = UNet_3head(3,1)
    # netD_A = NLayerDiscriminator(1, n_layers=4)
    # netG_A, netD_A = train_I2I_CasUNet3headGAN(
    #     netG_A, netD_A,
    #     train_dataloader, valid_dataloader,
    #     dtype=torch.cuda.FloatTensor,
    #     device='cuda',
    #     num_epochs=50,
    #     init_lr=1e-5,
    #     ckpt_path='../ckpt_wacv_1/conditional_CT2PET_UNet_3head_block1',
    # ) 

    # init net and train
    # netG_A1 = CasUNet_3head(1,1)
    # netG_A1 = UNet_3head(3,1)
    # netG_A1.load_state_dict(torch.load('/home/PET-CT/thaind/medical-image-translation/UncerGuidedI2I/ckpt_wacv/conditional_CT2PET_UNet_3head_block1_G_best_mae_0.015252234414219856.pth'))
    # netG_A2 = UNet_3head(6,1)

    # netD_A = NLayerDiscriminator(1, n_layers=4)
    # list_netG_A, list_netD_A = train_I2I_Sequence_CasUNet3headGAN(
    #     [netG_A1, netG_A2], [netD_A],
    #     train_dataloader, valid_dataloader,
    #     dtype=torch.cuda.FloatTensor,
    #     device='cuda',
    #     num_epochs=50,
    #     init_lr=1e-5,
    #     ckpt_path='../ckpt_wacv_1/conditional_CT2PET_UNet_3head_block2',
    # )

    # init net and train
    # netG_A1 = CasUNet_3head(1,1)
    # netG_A1.load_state_dict(torch.load('../ckpt/CT2PET_CasUNet_3head_block1_G_best_mape_10093.70298949962.pth'))
    # netG_A2 = UNet_3head(4,1)
    # netG_A2.load_state_dict(torch.load('../ckpt/CT2PET_Sequence_CasUNet_3head_block2_G_best_mape_6861.368122823695.pth'))
    # netG_A3 = UNet_3head(4,1)

    # netD_A = NLayerDiscriminator(1, n_layers=4)
    # list_netG_A, list_netD_A = train_I2I_Sequence_CasUNet3headGAN(
    #     [netG_A1, netG_A2, netG_A3], [netD_A],
    #     train_dataloader, valid_dataloader,
    #     dtype=torch.cuda.FloatTensor,
    #     device='cuda',
    #     num_epochs=50,
    #     init_lr=1e-5,
    #     ckpt_path='../ckpt/CT2PET_Sequence_CasUNet_3head_block3',
    # )
    
if __name__ == '__main__':
    main()