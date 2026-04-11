import os
import torch
from torch.utils.data import DataLoader, Dataset 
import monai.transforms as mtf
import pandas as pd
import numpy as np
from datasets_utils.abnormality_list_56 import abnormality_dict, abnormality_list

def anno_files(setname):
    if setname == 'Brown':
        data_split = pd.read_excel('/media/brownradx/ssd_data2/Brown_PE/tables/labels_20251104_splits_filterNoLung.xlsx', index_col=False) 
        abntext_path = '/media/brownradx/ssd_code/Projects_zhusi/PE_data_process/pe_25_code/LLM_abn_text_gptoss_brown_56/brown_CTPA_PE_disease_finding_56.xlsx'
        image_root = '/media/brownradx/ssd_data2/Brown_PE/CTPA_process/CTPA_cleaned_224160'
    elif setname == 'INSPECT':
        data_split = pd.read_excel('/media/brownradx/ssd_data2/INSPECT_PE/tables/labels_20250611_named_splits_casewEHR_filterNoLung.xlsx', index_col=False) 
        abntext_path = '/media/brownradx/ssd_code/Projects_zhusi/PE_data_process/pe_25_code/LLM_abn_text_gptoss_inspect_56/inspecta_CTPA_PE_disease_finding_56.xlsx'
        image_root = '/media/brownradx/ssd_data2/INSPECT_PE/CTPA_process/CTPA_cleaned_224160'
    elif setname == 'JHU':
        data_split = pd.read_excel('/media/brownradx/ssd_data2/JHU_PE/tables/labels_20251109_splits_filterNoLung.xlsx', index_col=False)   
        abntext_path = '/media/brownradx/ssd_code/Projects_zhusi/PE_data_process/pe_25_code/LLM_abn_text_gptoss_jhu_56/jhu_CTPA_PE_disease_finding_56.xlsx'
        image_root = '/media/brownradx/ssd_data2/JHU_PE/CTPA_process/CTPA_cleaned_224160'
    else:
        raise ValueError(f"Unsupported setname: {setname}. Expected one of ['Brown', 'INSPECT', 'JHU'].")
    return data_split, abntext_path, image_root

def ImgABNDataset(setname, mode="train"):
    # Backward-compatible: allow passing an args-like object with `.setname`.
    if not isinstance(setname, str):
        setname = getattr(setname, "setname", None)

    data_split, abntext_path, image_root = anno_files(setname)
        
    if mode in ['train', 'valid', 'test']:
        image_df_ids = data_split[data_split['split'] == mode]
    else:
        image_df_ids = data_split
        
    abnLabel_path = abntext_path.replace('.xlsx', '_abnLabel.xlsx')
    abnLabel_df = pd.read_excel(abnLabel_path)
        
    set_abn_labels_df = abnLabel_df[abnLabel_df['AccessionNumber_md5'].isin(image_df_ids['AccessionNumber_md5'])]
    set_abn_labels_df = set_abn_labels_df.set_index('AccessionNumber_md5')
    set_abn_labels_df['abn_labels'] = set_abn_labels_df[abnormality_list].values.tolist()
    set_abn_labels_df = set_abn_labels_df.reset_index(drop=False)
    
    data_list = []
    for d in image_df_ids.iterrows():
        d = d[1]
        accNum = d['AccessionNumber_md5']
        case_abn_labels_d = set_abn_labels_df[set_abn_labels_df['AccessionNumber_md5'] == accNum].iloc[0]
        image_abs_path = os.path.join(image_root, d['image_id']+'.nii.gz')
        d_dict = {"image": image_abs_path, "image_path": image_abs_path, "label": np.array(case_abn_labels_d['abn_labels']).astype(int)}
        data_list.append(d_dict)
    return data_list

train_transform = mtf.Compose(
        [
            mtf.LoadImaged(keys=["image", ], ensure_channel_first=True),
            mtf.ScaleIntensityRanged(
                keys=["image", ], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True
            ),
            mtf.RandRotate90d(keys=["image", ],prob=0.7, spatial_axes=(0, 1)),
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