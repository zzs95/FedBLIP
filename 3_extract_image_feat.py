import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['CUDA_VISIBLE_DEVICES'] = '7' # debug

import torch
from tqdm import tqdm
import monai.transforms as mtf
import numpy as np
from datasets_utils.abnormality_list_56 import abnormality_dict
organ_abn_nums = [len(abnormality_dict[k]) for k in abnormality_dict.keys()]
import monai
from train_stage1_base import ImgABNDataset
from models.image_classifier import ImageClassifier_BASE as ImageClassifier
from Utils.file_and_folder_operations import *
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 基础设置 ===

def extract_feat(client_name, model, loader, client_test_path):
    print(f"\n🏥 Infering on client: {client_name} ")
    scaler = torch.amp.GradScaler('cuda')
    device = DEVICE
    for batch_data in tqdm(loader):
        img_path = batch_data['image_path'][0]
        accNum_modal = img_path.split('/')[-1].replace('.nii.gz', '')
        feat_dir = os.path.join(client_test_path, accNum_modal)
        # if os.path.exists(feat_dir):
        #     continue
        os.makedirs(feat_dir, exist_ok=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=scaler is not None):
            images = batch_data['image']
            images = images.float().to(device, non_blocking=True)
            region_mask = None
            labels = batch_data['label'].float().to(device, non_blocking=True)
            splits = torch.cumsum(torch.tensor(organ_abn_nums), dim=0)[:-1]
            label_groups = torch.tensor_split(labels, splits.tolist(), dim=1)
            label_organs = torch.stack([x.sum(dim=1) for x in label_groups], dim=1) > 0
            label_organs = label_organs.float()
            
            region_onehot = []
        
            with torch.no_grad():
                skips, pooled_feat = model(images, region_mask, region_onehot)

            BATCH_SIZE = images.shape[0]
            if 'train' == client_test_path.split('/')[-1] and BATCH_SIZE>1:
                save_size = int(max(2, min(BATCH_SIZE, labels[0].sum()) ))
            else:
                save_size = BATCH_SIZE
            for b_i in range(save_size):                
                skips_dict = {}
                for l_i in range(1, 5):
                    patch_size = int(32/2**l_i)
                    patched_feat = model.i3_resnet.avgpool(patchify_3d(torch.squeeze(skips[l_i][b_i]))).squeeze().permute(1, 0).reshape(-1, patch_size,patch_size,patch_size)
                    skips_dict['pooled_scale_'+str(l_i)] = patched_feat.as_tensor().detach().cpu().numpy().astype(np.float16)
                    
                l_i = 5
                patched_feat = torch.squeeze(skips[l_i][b_i])
                skips_dict['scale_'+str(l_i)] = patched_feat.as_tensor().detach().cpu().numpy().astype(np.float16)
                
                img_path = batch_data['image_path'][b_i]
                accNum_modal = img_path.split('/')[-1].replace('.nii.gz', '')
                npz_name = str(b_i) + '_' + accNum_modal
                npz_path = os.path.join(client_test_path, accNum_modal, npz_name)
                if not os.path.exists(npz_path + '.npz'):
                    np.savez(npz_path, **skips_dict)
                else:
                    b_ii = b_i
                    if_save = False
                    while not if_save:
                        b_ii = b_ii + 1
                        npz_name = str(b_ii) + '_' + accNum_modal
                        npz_path = os.path.join(client_test_path, accNum_modal, npz_name)
                        if not os.path.exists(npz_path + '.npz'):
                            np.savez(npz_path, **skips_dict)
                            if_save = True
                        


def patchify_3d(image, patch_size=(7,7,10)):
    """
    Patchify a 3D image into smaller patches.

    Args:
        image (torch.Tensor): The input 3D image tensor (C, D, H, W).
        patch_size (tuple): The size of each patch (patch_d, patch_h, patch_w).

    Returns:
        torch.Tensor: The tensor containing the patches.
    """

    patches = image.unfold(1, patch_size[0], patch_size[0])
    patches = patches.unfold(2, patch_size[1], patch_size[1])
    patches = patches.unfold(3, patch_size[2], patch_size[2])
    patches = patches.permute(0, 1, 2, 3, 4, 5, 6).contiguous()
    patches = patches.view(-1, image.shape[0], patch_size[0], patch_size[1], patch_size[2])
    return patches

train_transform = mtf.Compose(
        [
            mtf.LoadImaged(keys=["image", ], ensure_channel_first=True),
            mtf.ScaleIntensityRanged(
                keys=["image", ], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True
            ),

            mtf.RandRotate90d(keys=["image", ],prob=0.7, spatial_axes=(0, 1)), # ?
            mtf.RandFlipd(keys=["image", ],prob=0.10, spatial_axis=0),
            mtf.RandFlipd(keys=["image", ],prob=0.10, spatial_axis=1),
            mtf.RandFlipd(keys=["image", ],prob=0.10, spatial_axis=2),
            mtf.RandScaleIntensityd(keys=["image"],factors=0.1, prob=0.9),
            mtf.RandShiftIntensityd(keys=["image"],offsets=0.1, prob=0.9),
            mtf.ToTensord(keys=["image",  ], dtype=torch.float),
        ]
    )
val_transform = mtf.Compose(
        [
            mtf.LoadImaged(keys=["image", ], ensure_channel_first=True),
            mtf.ScaleIntensityRanged(
                keys=["image", ], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True
            ),
            mtf.ToTensord(keys=["image",  ], dtype=torch.float),
        ]
    )

def seed_everything(seed):
    '''
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    '''
    import os
    import random
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def main():
    seed_everything(24)
    print("🚀 Starting 3-Center Federated Pretraind Model Testing")
    model = ImageClassifier(out_channels=[len(v) for v in abnormality_dict.values()], extract_feat=True).to(DEVICE)
    
    # feat_path = '/media/brownradx/ssd_data2/VLM_PE_feat/fed_feat'
    # ckpt_dir = '/media/brownradx/ssd_code/Projects_zhusi/PE_data_process/pe_25_code/abn_classification_fed/fed_server_store/'
    # ckpt_dir = os.path.join(exp_path, "checkpoint")
    # ckeckpoint_name = f"global_round_{100}.pth"
    
    # save_path = os.path.join(ckpt_dir, ckeckpoint_name)
    # checkpoint_dict = torch.load(save_path, weights_only=True)
    # print(model.load_state_dict(checkpoint_dict))
    # model = model.to(DEVICE, non_blocking=True)
    
    # feat_path = '/media/brownradx/ssd_data2/VLM_PE_feat/mix_feat'
    # ckpt_dir = '/media/brownradx/ssd_code/Projects_zhusi/PE_data_process/pe_25_code/abn_classification_fed/saved/exp_mix/checkpoint/'
    # ckeckpoint_name = f"best_auc_epoch_6.pth"
    # save_path = os.path.join(ckpt_dir, ckeckpoint_name)
    # checkpoint_dict = torch.load(save_path)['model']
    # print(model.load_state_dict(checkpoint_dict))
    # model = model.to(DEVICE, non_blocking=True)
    
    
    feat_path = './stage1_img_feat'
    ckpt_dir = './stage1_fed_server_store/'
    ckpt_round_num = 150

    # CLIENT_NAMES = ['Brown','INSPECT', 'JHU'] 
    CLIENT_NAMES = ['INSPECT'] 
    for client_name in CLIENT_NAMES:
        ckeckpoint_name = f"{client_name}_round_{ckpt_round_num}.pth"
        
        save_path = os.path.join(ckpt_dir, ckeckpoint_name)
        checkpoint_dict = torch.load(save_path, weights_only=True)
        print(model.load_state_dict(checkpoint_dict))
        model = model.to(DEVICE, non_blocking=True)

        if client_name == 'Brown':
            total_train_num = 250000
        elif client_name == 'INSPECT':
            total_train_num = 100000
        elif client_name == 'JHU':
            total_train_num = 25000 
        client_test_dir = feat_path + '/' + client_name
        args = type('args', (), {})()
        args.setname = client_name
        args.workers = 6
        for trts in ['train', 'valid', 'test']:
            client_test_path = client_test_dir + '/' + trts
            os.makedirs(client_test_path, exist_ok=True) 
            if trts == 'train':
                args.batch_size = 10
                test_list = ImgABNDataset(client_name, mode=trts)
                test_list = test_list[:30]
                test_list_repeat = [val for val in test_list for i in range(args.batch_size)]
                test_ds = monai.data.Dataset(test_list_repeat, transform=train_transform)
            else:
                args.batch_size = 1
                test_list = ImgABNDataset(client_name, mode=trts)
                test_list = test_list[:30]
                test_ds = monai.data.Dataset(test_list, transform=val_transform)
            test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

            extract_feat(client_name, model, test_loader, client_test_path)

            # if trts == 'train':
                # args.batch_size = 1
                # test_list = ImgABNDataset(client_name, mode=trts)
                
                # all_files = []
                # for accNum_modal in subfolders(client_test_path):
                #     all_files += subfiles(accNum_modal)
                # run_idx = np.random.randint(0, len(test_list), total_train_num - len(all_files))
                # test_list_supp = []
                # for i in run_idx:
                #     test_list_supp.append(test_list[i])
                    
                # test_ds = monai.data.Dataset(test_list_supp, transform=train_transform)
                # test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
                # extract_feat(client_name, model, test_loader, client_test_path)
            
            
        

if __name__ == "__main__":
    main()