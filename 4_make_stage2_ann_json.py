# anno_file_test.json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from Utils.file_and_folder_operations import *
from datasets_utils.abnormality_list_56 import abnormality_list
from datasets_utils.img_cls import anno_files
import numpy as np
from tqdm import tqdm
feat_root = './stage1_img_feat/'

# CLIENT_NAMES = ['Brown', 'INSPECT', 'JHU'] 
CLIENT_NAMES = ['INSPECT'] 
for setname in CLIENT_NAMES:
    data_split, abntext_path, image_root = anno_files(setname)
        
    abn_text_df = pd.read_excel(abntext_path, index_col=False)
    abn_label_df = pd.read_excel(abntext_path.replace('.xlsx', '_abnLabel.xlsx'), index_col=False)
    img_feat_root = join(feat_root, setname)    
    for trts in ['train', 'valid', 'test']:
    # for trts in ['train']:
        ans_file= open(join(img_feat_root, 'anno_file_'+trts+'.json'), "w")
        trts_feat_root = join(img_feat_root, trts)
        for img_folder in tqdm(subfolders(trts_feat_root)):
            for img_aug_feat in subfiles(img_folder):
                accNum = os.path.split(img_aug_feat)[-1].replace('.npz', '').split('_', 1)[1].split('_CT')[0]
                try:
                    abn_label = abn_label_df[abn_label_df['AccessionNumber_md5'] == accNum].iloc[0]
                    abn_text = abn_text_df[abn_text_df['AccessionNumber_md5'] == accNum].iloc[0]
                    abn_label = abn_label[abnormality_list].values
                    text_list = abn_text[abnormality_list].values
                    text_list = text_list[np.argwhere(abn_label)].reshape(-1).tolist()
                    feat_dict = {}
                    feat_dict['image_feat'] = img_aug_feat.replace(img_feat_root, '')
                    feat_dict['AccessionNumber_md5'] = accNum
                    feat_dict['abn_label'] = abn_label.astype(int).tolist()
                    feat_dict['abn_text'] = text_list
                    
                    ans_file.write(json.dumps(feat_dict) + "\n")
                    ans_file.flush()
                except:
                    print(accNum)