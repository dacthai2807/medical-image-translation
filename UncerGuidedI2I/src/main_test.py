import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from ds import *
from trainer import *
from losses import *
from networks import *
from tqdm import tqdm

from torch.utils.data import DataLoader

# DATA_PATH = '/home/PET-CT/splited_data_15k'
DATA_PATH = '/home/PET-CT/tiennh/autopet256'
IMAGE_SIZE = 256
CT_MAX = 2047
PET_MAX = 65535
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

@torch.no_grad()
def save_single_image(image, save_path, file_name, max_pixel, to_normal=True):
    image = image.detach().clone()
    if to_normal:
        image = image.mul_(0.5).add_(0.5).clamp_(0, 1.)
    if max_pixel == 1:
        image = image.permute(1, 2, 0).to('cpu').numpy()
    else:
        image = image.mul_(max_pixel).add_(0.2).clamp_(0, max_pixel).permute(1, 2, 0).to('cpu').numpy()

    print(os.path.join(save_path, file_name))
    np.save(os.path.join(save_path, file_name), image)

def make_dir(dir):
    os.makedirs(dir, exist_ok=True)
    return dir

def main():
    # train_dataset = get_dataset_by_stage(DATA_PATH, 'train', (IMAGE_SIZE, IMAGE_SIZE), CT_MAX, PET_MAX, True)
    # val_dataset = get_dataset_by_stage(DATA_PATH, 'val', (IMAGE_SIZE, IMAGE_SIZE), CT_MAX, PET_MAX, False)
    test_dataset = get_dataset_by_stage(DATA_PATH, 'test', (IMAGE_SIZE, IMAGE_SIZE), CT_MAX, PET_MAX, False)

    # train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16)
    # valid_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=16)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=16)

    CKPT_PATH = '/home/PET-CT/thaind/medical-image-translation/UncerGuidedI2I/ckpt_autopet/CT2PET_UNet_3head_block2_G_best_mae_0.025437891483306885.pth'
    SAMPLE_PATH = '/home/PET-CT/thaind/medical-image-translation/UncerGuidedI2I/samples/UPGAN_autopet'
    
    # netG_A = CasUNet(1,1)
    # # netG_A = CasUNet_3head(1,1)
    # # netG_A = UNet(3,1)
    # netG_A.load_state_dict(torch.load(CKPT_PATH))
    # netG_A.type(torch.FloatTensor)
    # netG_A.eval()
    
    netG_A1 = CasUNet_3head(1,1)
    # netG_A1 = UNet_3head(3,1)
    netG_A1.load_state_dict(torch.load('/home/PET-CT/thaind/medical-image-translation/UncerGuidedI2I/ckpt_autopet/CT2PET_UNet_3head_block1_G_best_mae_0.013043863698840141.pth'))
    netG_A1.type(torch.FloatTensor)
    netG_A1.eval()
    netG_A2 = UNet_3head(4,1)
    netG_A2.load_state_dict(torch.load(CKPT_PATH))
    netG_A2.type(torch.FloatTensor)
    netG_A2.eval()
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            xA, xB, x_name = batch[0].type(torch.FloatTensor), batch[1].type(torch.FloatTensor), batch[2]
            
            # rec_B = netG_A(xA)
            # rec_B, rec_alpha_B, rec_beta_B = netG_A(xA)
            for nid, netG in enumerate([netG_A1, netG_A2]):
                if nid == 0:
                    rec_B, rec_alpha_B, rec_beta_B = netG(xA)
                else:
                    xch = torch.cat([rec_B, rec_alpha_B, rec_beta_B, xA], dim=1)
                    rec_B, rec_alpha_B, rec_beta_B = netG(xch)
            
            n = xB.shape[0]
            
            for t in range(n):
                # gt_path = make_dir(os.path.join(SAMPLE_PATH, 'ground_truth'))
                # save_single_image(xB[t], gt_path, f'{x_name[t]}.npy', max_pixel=32767, to_normal=True)

                pred_path = make_dir(os.path.join(SAMPLE_PATH, 'predicted'))
                save_single_image(rec_B[t], pred_path, f'{x_name[t]}.npy', max_pixel=PET_MAX, to_normal=True)
    
if __name__ == '__main__':
    main()