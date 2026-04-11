import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import torch

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset 
import numpy as np
from datasets_utils.img_cls import ImgABNDataset, train_transform, val_transform
import monai
import torchvision
from models.image_classifier import ImageClassifier_BASE as ImageClassifier
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from Utils.dist_utils import *
from Utils.lr_utils import *

from datasets_utils.abnormality_list_56 import abnormality_dict
organ_abn_nums = [len(abnormality_dict[k]) for k in abnormality_dict.keys()]


def run_one_epoch(model, loader, criterion, optimizer, scaler, device, is_validate=False, clip_grad=1.0, nan_guard=True):
    model.train()
    total_loss = 0
    abn_total_loss = 0
    organ_total_loss = 0
    n_batches = 0
    
    # 统计指标（分布式上先逐卡累计，后面 all_gather 再汇总）
    preds_buf = []
    labels_buf = []

    for batch_data in tqdm(loader):
        batch_data = batch_data
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=scaler is not None):
            images = batch_data['image']
            images = images.float().to(device, non_blocking=True)
            # segms = batch_data['seg']
            # region_mask = segms.int().to(device, non_blocking=True)
            region_mask = None
            labels = batch_data['label'].float().to(device, non_blocking=True)
            splits = torch.cumsum(torch.tensor(organ_abn_nums), dim=0)[:-1]
            label_groups = torch.tensor_split(labels, splits.tolist(), dim=1)
            label_organs = torch.stack([x.sum(dim=1) for x in label_groups], dim=1) > 0
            label_organs = label_organs.float()
            
            region_onehot = []
            if not is_validate:
                logits, organ_logits = model(images, region_mask, region_onehot) # 2048 1024
            else:
                with torch.no_grad():
                    logits, organ_logits = model(images, region_mask, region_onehot) # 2048 1024
            loss_abn = criterion(logits, labels)
            # loss_organ = criterion(organ_logits, label_organs)
            # loss = 0.8*loss_organ + 0.2*loss_abn
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

        

from Utils.dist_utils import *
from Utils.lr_utils import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-path", type=str, default='exps_stage1_debug')
    parser.add_argument("--setname", type=str, default='Mix')
    
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eta_min", type=float, default=1e-6)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    # 兼容 launch 传入的 --local_rank；torchrun 通过 LOCAL_RANK 环境变量传入
    parser.add_argument("--local-rank", type=int, default=0)
    
    args = parser.parse_args()
    
    setup_seed(args.seed)
    train_root = args.exp_path + '/' + args.setname+ '/'
    checkpoint_dir = train_root + "checkpoint/" 
    os.makedirs(os.path.dirname(checkpoint_dir), exist_ok=True)
    log_dir = train_root + "runs"
    writer = SummaryWriter(log_dir=log_dir) 
    
    if not dist.is_initialized(): # dist_debug
        init_distributed(args) # dist_debug
    local_rank = args.local_rank 
    device = torch.device("cuda", local_rank)
    

    if args.setname == 'Mix':
        training_list = []
        for setname_curr in ['Brown', 'INSPECT', 'JHU']:
            training_list_curr = ImgABNDataset(setname_curr, mode='train')
            training_list += training_list_curr
    elif args.setname in ['Brown', 'INSPECT', 'JHU']:
        training_list = ImgABNDataset(args.setname, mode='train')
    # training_list = training_list[:300]
    training_data = monai.data.Dataset(data=training_list, transform=train_transform)
    
    if args.setname == 'Mix':
        val_list = []
        for setname_curr in ['Brown', 'INSPECT', 'JHU']:
            list_curr = ImgABNDataset(setname_curr, mode='valid')
            val_list += list_curr
    elif args.setname in ['Brown', 'INSPECT', 'JHU']:
        val_list = ImgABNDataset(args.setname, mode='valid')
    # val_list = val_list[:300]
    val_data = monai.data.Dataset(data=val_list, transform=val_transform)
    
    print(len(training_data), len(val_data))
    
    train_sampler = DistributedSampler(training_data, shuffle=True, drop_last=True) # dist_debug
    
    target = torch.tensor(np.concatenate([np.array(a['label'])[None] for a in training_list], axis=0).astype(int))# [:, :len(abnormality_dict['Pulmonary arteries'])]
    class_sample_count = target.sum(dim=0)   # [C]
    class_sample_count = class_sample_count.clamp_min(1e-6)
    mean_weight = 1.0 / class_sample_count.float()              # 频率反比
    reverse_weight = mean_weight / class_sample_count.float()   # 更强反比（平方）
    mean_samples_weight = (target * mean_weight).sum(dim=1)
    reverse_samples_weight = (target * reverse_weight).sum(dim=1)
    mean_samples_weight = mean_samples_weight.clamp_min(mean_samples_weight.mean() * 0.1)
    reverse_samples_weight = reverse_samples_weight.clamp_min(reverse_samples_weight.mean() * 0.1)
    
    from Utils.sampler import WeightedDistributedSampler # dist_debug
    train_sampler = WeightedDistributedSampler(weights=mean_samples_weight, # dist_debug
        rank=local_rank, # dist_debug
        replacement=True,    # dist_debug
        shuffle=True, # dist_debug
        seed=args.seed # dist_debug
    ) # dist_debug
    
    train_loader = DataLoader(training_data, 
                            batch_size=args.batch_size,
                            num_workers=args.workers,
                            sampler=train_sampler,# dist_debug
                            pin_memory=True,
                            )
    
    val_sampler = DistributedSampler(val_data, shuffle=False, drop_last=True) # dist_debug
    val_loader = DataLoader(val_data, 
                            batch_size=args.batch_size, 
                            sampler=val_sampler, # dist_debug
                            drop_last=True) 
    

    model = ImageClassifier(out_channels=organ_abn_nums)
    pretrained_dict = torch.load('./models/I3D_resnetInit.pth', weights_only=True)

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    
    model_state_dict = model.state_dict()
    for k, v in model_state_dict.items():
        name = k   
        if name in list(pretrained_dict.keys()):
            new_state_dict[name] = pretrained_dict[name]
            # print(name)
        else:
            new_state_dict[name] = v
            print(name, 'mismatched')
    print(model.load_state_dict(new_state_dict))

    model = model.to(device, non_blocking=True)
    model = DDP(model, device_ids=[local_rank],  output_device=local_rank, find_unused_parameters=False)# find_unused_parameters=True) # dist_debug
    model._set_static_graph()  # dist_debug
    
        
    criterion = torch.nn.BCEWithLogitsLoss() # multi-class
    optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda')

    scheduler = WarmupCosine(
        optimizer=optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        base_lr=args.lr,
        eta_min=args.eta_min,
    )

    # CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=2 --master_port=29600 train_dist_base.py --setname='JHU'
    # CUDA_VISIBLE_DEVICES=2,3,4 torchrun --nproc_per_node=3 --master_port=29602 train_dist_base.py --setname='Brown'
    # CUDA_VISIBLE_DEVICES=5,6,7 torchrun --nproc_per_node=3 --master_port=29604 train_dist_base.py --setname='INSPECT'
    best_metric = {}
    best_metric['acc'] = -1
    best_metric['auc'] = -1
    best_metric['f1'] = -1
    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)  # 保证每轮打乱不同（DDP 要点）# dist_debug
        lr_now = scheduler.step()

        start = time.time()
        avg_loss, organ_loss, abn_loss, metrics_dict = run_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, clip_grad=1.0, nan_guard=True
        )
        elapsed = time.time() - start

        if is_main_process():
            writer.add_scalar("train loss", avg_loss, epoch + 1)
            writer.add_scalar("train organ loss", organ_loss, epoch + 1)
            writer.add_scalar("train abn loss", abn_loss, epoch + 1)
            writer.add_scalar("train acc", metrics_dict['acc'], epoch + 1)
            writer.add_scalar("train auc", metrics_dict['auc'], epoch + 1)
            writer.add_scalar("train f1", metrics_dict['f1'], epoch + 1)
            writer.add_scalar("learning rate", lr_now, epoch + 1)
            print(f"[Epoch {epoch:03d}/{args.epochs}] lr={lr_now:.2e} | loss={avg_loss:.6f} | microF1={metrics_dict['f1']:.4f} | {elapsed:.1f}s")

        if (epoch + 1) % args.val_interval == 0:
            print("validating")
            start = time.time()
            avg_loss, organ_loss, abn_loss, metrics_dict = run_one_epoch(
                model, val_loader, criterion, optimizer, scaler, device, is_validate=True, clip_grad=1.0, nan_guard=True
            )
            elapsed = time.time() - start
            
            if is_main_process():
                writer.add_scalar("val loss", avg_loss, epoch + 1)
                writer.add_scalar("val organ loss", organ_loss, epoch + 1)
                writer.add_scalar("val abn loss", abn_loss, epoch + 1)
                writer.add_scalar("val acc", metrics_dict['acc'], epoch + 1)
                writer.add_scalar("val auc", metrics_dict['auc'], epoch + 1)
                writer.add_scalar("val f1", metrics_dict['f1'], epoch + 1)
                
                if metrics_dict['auc'] >= best_metric['auc'] and dist.get_rank() == 0:
                    best_metric['acc'] = metrics_dict['acc']
                    best_metric['auc'] = metrics_dict['auc']
                    best_metric['f1'] = metrics_dict['f1']
                    best_metric_epoch = epoch + 1
                    tempdir = os.path.join(checkpoint_dir, "best_auc_epoch_"+str(best_metric_epoch)+".pth")
                    torch.save(
                            {
                                "epoch": epoch,
                                "model": model.module.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "scaler": scaler.state_dict() if scaler is not None else None,
                                "best_micro_f1": metrics_dict['f1'],
                                "best_micro_acc": metrics_dict['acc'],
                                "best_micro_auc": metrics_dict['auc'],
                                "args": vars(args),
                            }, tempdir)
                    print(f"✓ Saved best checkpoint to {tempdir} (ACC={metrics_dict['acc']:.4f} AUC={metrics_dict['auc']:.4f} microF1={metrics_dict['f1']:.4f})")

        if is_main_process() and (epoch+1) % 100 == 0 and (epoch+1) > 99:                
            tempdir = os.path.join(checkpoint_dir, 'epoch' + str(epoch + 1)+".pth")
            torch.save(
                    {
                        "epoch": epoch,
                        "model": model.module.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict() if scaler is not None else None,
                        "micro_f1": metrics_dict['f1'],
                        "micro_acc": metrics_dict['acc'],
                        "micro_auc": metrics_dict['auc'],
                        "args": vars(args),
                    }, tempdir)
            print(f"✓ Saved epoch {str(epoch + 1)} checkpoint to {tempdir} (ACC={metrics_dict['acc']:.4f} AUC={metrics_dict['auc']:.4f} microF1={metrics_dict['f1']:.4f})")

    cleanup_distributed()

if __name__ == "__main__":
    main()
