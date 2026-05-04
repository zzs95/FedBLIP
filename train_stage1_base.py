import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

import torch.distributed as dist
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from Utils.dist_utils import *
from Utils.lr_utils import *

def run_one_epoch(model, loader, criterion, device, optimizer=None, scaler=None, is_validate=False, clip_grad=1.0, nan_guard=True):
    model.train()
    total_loss = 0
    abn_total_loss = 0
    organ_total_loss = 0
    n_batches = 0
    
    # 统计指标（分布式上先逐卡累计，后面 all_gather 再汇总）
    preds_buf = []
    labels_buf = []

    if dist.get_rank() == 0:
        loader = tqdm(loader, mininterval=60.0)

    for batch_data in loader:
        batch_data = batch_data
        if not is_validate:
            optimizer.zero_grad(set_to_none=True)
        
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=scaler is not None):
            images = batch_data['image']
            images = images.float().to(device, non_blocking=True)
            labels = batch_data['label'].float().to(device, non_blocking=True)
            
            if not is_validate:
                logits, _ = model(images) # 2048 1024
            else:
                with torch.no_grad():
                    logits, _ = model(images) # 2048 1024
            loss_abn = criterion(logits, labels)
            loss = loss_abn
            
        if nan_guard and (not torch.isfinite(loss)):
            print('nan_guard')
            # 跳过异常 batch
            continue
        
        if not is_validate:
            if scaler is not None:
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
                optimizer.step()
        
        total_loss += loss.detach()
        abn_total_loss += loss_abn.detach()
        # organ_total_loss += loss_organ.detach()
        organ_total_loss = torch.tensor(0).cuda()
        n_batches += 1
        
        # 收集本卡预测与标签（用于 epoch 末 all_gather 计算 F1）
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            preds_buf.append(preds.detach())
            labels_buf.append(labels.detach())
            
    if n_batches == 0:
        avg_loss = torch.tensor(float("nan"), device=device)
    else:
        avg_loss = total_loss / n_batches
        abn_avg_loss = abn_total_loss / n_batches
        organ_avg_loss = organ_total_loss / n_batches
        
    # 先聚合 loss（求平均）
    if dist.is_initialized():
        avg_loss = reduce_tensor(avg_loss)
        abn_avg_loss = reduce_tensor(abn_avg_loss)
        organ_avg_loss = reduce_tensor(organ_avg_loss)

    # 拼接本卡的预测和标签
    if len(preds_buf) > 0:
        preds_local = torch.cat(preds_buf, dim=0)
        labels_local = torch.cat(labels_buf, dim=0)
    else:
        preds_local = torch.zeros(0, device=device)
        labels_local = torch.zeros(0, device=device)

    # 分布式收集（可变长）
    if dist.is_initialized():
        preds_all = all_gather_variable_len_tensor(preds_local)
        labels_all = all_gather_variable_len_tensor(labels_local)
    else:
        preds_all, labels_all = preds_local, labels_local

    # 计算 micro-F1（在 CPU 上）
    if preds_all.numel() > 0:
        f1 = f1_score(
            labels_all.cpu().numpy(),
            preds_all.cpu().numpy(),
            average="micro",
            zero_division=0,
        )
        acc = accuracy_score(
            labels_all.cpu().numpy(),
            preds_all.cpu().numpy(),
            )
        auc = roc_auc_score(
            labels_all.cpu().numpy(),
            preds_all.cpu().numpy(),
            average="micro",
        )
    else:
        f1 = float("nan")
        acc = float("nan")
        auc = float("nan")

    return avg_loss.item(), organ_avg_loss.item(), abn_avg_loss.item(), {'f1':f1, 'acc': acc, 'auc': auc}

        