import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['CUDA_VISIBLE_DEVICES'] = '2' # dist_debug
import argparse
import json
# import math
import torch

from tqdm import tqdm
import numpy as np
import pandas as pd
from datasets_utils.abnormality_list_56 import abnormality_dict, abnormality_list
organ_abn_nums = [len(abnormality_dict[k]) for k in abnormality_dict.keys()]
import monai
import torchvision
from train_stage1_base import ImgABNDataset, val_transform
from models.image_classifier import ImageClassifier_BASE as ImageClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def local_test(client_name, global_model, loader):
    print(f"\n🏥 Testing on client: {client_name} ")
    args = type('args', (), {})()
    args.setname = client_name
    args.batch_size = 10
    args.workers = 6
    scaler = torch.amp.GradScaler('cuda')
    device = DEVICE
    
    # 统计指标（分布式上先逐卡累计，后面 all_gather 再汇总）
    probs_buf = []
    preds_buf = []
    labels_buf = []
    probs_o_buf = []
    preds_o_buf = []
    labels_o_buf = []

    for batch_data in tqdm(loader):
        batch_data = batch_data
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
                logits, organ_logits = global_model(images, region_mask, region_onehot)
            logits = logits.squeeze() 
            
        # 收集本卡预测与标签（用于 epoch 末 all_gather 计算 F1）
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            probs_buf.append(probs.detach())
            preds_buf.append(preds.detach())
            labels_buf.append(labels.detach())
            
            # probs = torch.sigmoid(organ_logits)
            probs = torch.zeros(label_organs.shape)
            preds = (probs > 0.5).float()
            probs_o_buf.append(probs.detach())
            preds_o_buf.append(preds.detach())
            labels_o_buf.append(label_organs.detach())

    probs_local = torch.cat(probs_buf, dim=0)
    labels_local = torch.cat(labels_buf, dim=0)
    probs_o_local = torch.cat(probs_o_buf, dim=0)
    labels_o_local = torch.cat(labels_o_buf, dim=0)

    return probs_local, labels_local, probs_o_local, labels_o_local


def main():
    print("🚀 Starting 3-Center Federated Pretraind Model Testing")
    global_model = ImageClassifier(out_channels=[len(v) for v in abnormality_dict.values()]).to(DEVICE)
    
    exp_path = './'
    test_output = exp_path + 'stage1_test_result'
    ckpt_dir = exp_path + 'stage1_fed_server_store/'
    ckpt_round_num = 2
    
    # CLIENT_NAMES = ['Brown', 'INSPECT', 'JHU'] 
    CLIENT_NAMES = ['INSPECT', ] 
    for client_name in CLIENT_NAMES:
        ckeckpoint_name = f"{client_name}_round_{ckpt_round_num}.pth"
        save_path = os.path.join(ckpt_dir, ckeckpoint_name)
        checkpoint_dict = torch.load(save_path, weights_only=True)
        print(global_model.load_state_dict(checkpoint_dict))
        model = global_model.to(DEVICE, non_blocking=True)
        
        client_test_path = test_output + '/' + client_name
        os.makedirs(client_test_path, exist_ok=True) 
        args = type('args', (), {})()
        args.setname = client_name
        args.batch_size = 10
        args.workers = 6
        test_list = ImgABNDataset(args, mode='test')
        test_list = test_list[:30] # debug
        test_ds = monai.data.Dataset(test_list, transform=val_transform)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

        probs_abn, labels_abn, probs_organ, labels_organ = local_test(client_name, model, test_loader)

        probs_abn = probs_abn.detach().cpu().numpy()
        labels_abn = labels_abn.detach().cpu().numpy()
        probs_organ = probs_organ.detach().cpu().numpy()
        labels_organ = labels_organ.detach().cpu().numpy()
    
        vqa_test_df = {}
        vqa_test_df['image_path'] = pd.DataFrame.from_dict(test_list)['image']
        for i_organ, organ in enumerate(abnormality_dict):
            vqa_test_df['gt_' + organ] = labels_organ[:,i_organ]
            vqa_test_df['pred_' + organ] = probs_organ[:,i_organ]
            for i_abn, abn in enumerate(abnormality_dict[organ]):
                vqa_test_df['gt_' + abn] = labels_abn[:,i_abn]
                vqa_test_df['pred_' + abn] = probs_abn[:,i_abn]
        vqa_test_df = pd.DataFrame.from_dict(vqa_test_df)
        vqa_test_df.to_excel(client_test_path + '/test_output_'+ckeckpoint_name.split('.')[0]+'.xlsx')
            
        from Utils.metricx_class import calculate_metrics
        metrics_df = []
        for i_abn, abn in enumerate(abnormality_list):
            auc0, acc, sensitivity, specificity, precision, f1_score, mAP, threshold, _ = \
                calculate_metrics(labels_abn[:,i_abn], probs_abn[:,i_abn], threshold=0.5)
            print(threshold, auc0, acc, sensitivity, specificity, precision, f1_score, mAP,)
            metrics_df_ = {}
            metrics_df_['abn_name'] = abn
            metrics_df_['acc'] = acc
            metrics_df_['auc'] = auc0
            metrics_df_['sensitivity'] = sensitivity
            metrics_df_['specificity'] = specificity
            metrics_df_['precision'] = precision
            metrics_df_['f1_score'] = f1_score
            metrics_df_['mAP'] = mAP
            metrics_df_['threshold'] = threshold
            metrics_df.append(metrics_df_)
        metrics_df = pd.DataFrame.from_dict(metrics_df)
        metrics_df = metrics_df.set_index('abn_name')
        metrics_df.loc['average'] = metrics_df.values.mean(0).tolist()
        metrics_df.to_excel(client_test_path + '/test_metrics_abn_'+ckeckpoint_name.split('.')[0]+'_tr0.5.xlsx')
        
        metrics_df = []
        for i_abn, abn in enumerate(abnormality_dict.keys()):
            auc0, acc, sensitivity, specificity, precision, f1_score, mAP, threshold, _ = \
                calculate_metrics(labels_organ[:,i_abn], probs_organ[:,i_abn], threshold=0.5)
            print(threshold, auc0, acc, sensitivity, specificity, precision, f1_score, mAP,)
            metrics_df_ = {}
            metrics_df_['abn_name'] = abn
            metrics_df_['acc'] = acc
            metrics_df_['auc'] = auc0
            metrics_df_['sensitivity'] = sensitivity
            metrics_df_['specificity'] = specificity
            metrics_df_['precision'] = precision
            metrics_df_['f1_score'] = f1_score
            metrics_df_['mAP'] = mAP
            metrics_df_['threshold'] = threshold
            metrics_df.append(metrics_df_)
        metrics_df = pd.DataFrame.from_dict(metrics_df)
        metrics_df = metrics_df.set_index('abn_name')
        metrics_df.loc['average'] = metrics_df.values.mean(0).tolist()
        metrics_df.to_excel(client_test_path +  '/test_metrics_organ_'+ckeckpoint_name.split('.')[0]+'_tr0.5.xlsx')


if __name__ == "__main__":
    main()
