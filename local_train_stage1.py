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
from Utils.dist_utils import *
from Utils.lr_utils import *

from train_stage1_base import run_one_epoch
from datasets_utils.abnormality_list_56 import abnormality_dict
organ_abn_nums = [len(abnormality_dict[k]) for k in abnormality_dict.keys()]

from Utils.dist_utils import *
from Utils.lr_utils import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-path", type=str, default='stage1_local')
    parser.add_argument("--setname", type=str, default='Mix')
    
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--eta_min", type=float, default=1e-6)
    parser.add_argument("--warmup_epochs", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-rank", type=int, default=0)

    args = parser.parse_args()

    setup_seed(args.seed)
    train_root = os.path.join(args.exp_path, args.setname)
    checkpoint_dir = os.path.join(train_root, "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_dir = os.path.join(train_root, "runs")

    if not dist.is_initialized():
        init_distributed(args)

    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    args.local_rank = local_rank
    device = torch.device("cuda", local_rank)
    writer = SummaryWriter(log_dir=log_dir) if is_main_process() else None

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

    if is_main_process():
        print(f"Dataset={args.setname} | train={len(training_data)} | val={len(val_data)}")

    train_sampler = DistributedSampler(training_data, shuffle=True, drop_last=False)
    val_sampler = DistributedSampler(val_data, shuffle=False, drop_last=False)

    num_workers = min(args.workers, 4)
    train_loader = DataLoader(
        training_data,
        batch_size=args.batch_size,
        num_workers=num_workers,
        sampler=train_sampler,
        pin_memory=False,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        num_workers=num_workers,
        sampler=val_sampler,
        pin_memory=False,
        persistent_workers=(num_workers > 0),
    )

    model = ImageClassifier(region_abn_counts=organ_abn_nums)

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
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )
    model._set_static_graph()

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda")

    from torch.optim.lr_scheduler import (
        SequentialLR, LinearLR, CosineAnnealingWarmRestarts,
        ExponentialLR, ConstantLR,
    )
    warmup_epochs = max(0, args.warmup_epochs)
    cosine_epochs = max(1, args.epochs - warmup_epochs)
    warmup = (
        LinearLR(optimizer, start_factor=0.2, total_iters=warmup_epochs)
        if warmup_epochs > 0
        else ConstantLR(optimizer, factor=1.0, total_iters=1)
    )
    cosine = CosineAnnealingWarmRestarts(
        optimizer, T_0=cosine_epochs, T_mult=2, eta_min=args.eta_min
    )
    exp_decay = ExponentialLR(optimizer, gamma=0.97)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, cosine, exp_decay],
        milestones=[warmup_epochs, warmup_epochs + cosine_epochs],
    )

    best_metric = {"acc": -1, "auc": -1, "f1": -1}

    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)

        start = time.time()
        avg_loss, organ_loss, abn_loss, metrics_dict = run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            clip_grad=1.0,
            nan_guard=True,
        )
        elapsed = time.time() - start
        lr_now = optimizer.param_groups[0]["lr"]

        if is_main_process():
            writer.add_scalar("train loss", avg_loss, epoch)
            writer.add_scalar("train organ loss", organ_loss, epoch)
            writer.add_scalar("train abn loss", abn_loss, epoch)
            writer.add_scalar("train acc", metrics_dict["acc"], epoch)
            writer.add_scalar("train auc", metrics_dict["auc"], epoch)
            writer.add_scalar("train f1", metrics_dict["f1"], epoch)
            writer.add_scalar("learning rate", lr_now, epoch)
            print(
                f"[Epoch {epoch:03d}/{args.epochs}] lr={lr_now:.2e} "
                f"| loss={avg_loss:.6f} | microF1={metrics_dict['f1']:.4f} | {elapsed:.1f}s"
            )

        scheduler.step()

        if epoch % args.val_interval == 0:
            if is_main_process():
                print("validating")
            start = time.time()
            avg_loss, organ_loss, abn_loss, metrics_dict = run_one_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                optimizer=None,
                scaler=None,
                is_validate=True,
                clip_grad=1.0,
                nan_guard=True,
            )
            elapsed = time.time() - start

            if is_main_process():
                writer.add_scalar("val loss", avg_loss, epoch)
                writer.add_scalar("val organ loss", organ_loss, epoch)
                writer.add_scalar("val abn loss", abn_loss, epoch)
                writer.add_scalar("val acc", metrics_dict["acc"], epoch)
                writer.add_scalar("val auc", metrics_dict["auc"], epoch)
                writer.add_scalar("val f1", metrics_dict["f1"], epoch)

                if metrics_dict["auc"] >= best_metric["auc"]:
                    best_metric["acc"] = metrics_dict["acc"]
                    best_metric["auc"] = metrics_dict["auc"]
                    best_metric["f1"] = metrics_dict["f1"]
                    tempdir = os.path.join(checkpoint_dir, f"best_auc_epoch_{epoch}.pth")
                    torch.save(
                        {
                            "epoch": epoch,
                            "model": model.module.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scaler": scaler.state_dict() if scaler is not None else None,
                            "best_micro_f1": metrics_dict["f1"],
                            "best_micro_acc": metrics_dict["acc"],
                            "best_micro_auc": metrics_dict["auc"],
                            "args": vars(args),
                        },
                        tempdir,
                    )
                    print(
                        f"✓ Saved best checkpoint to {tempdir} "
                        f"(ACC={metrics_dict['acc']:.4f} AUC={metrics_dict['auc']:.4f} "
                        f"microF1={metrics_dict['f1']:.4f}, val_time={elapsed:.1f}s)"
                    )

        if is_main_process() and epoch % args.epochs == 0:
            tempdir = os.path.join(checkpoint_dir, f"epoch{epoch}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict() if scaler is not None else None,
                    "micro_f1": metrics_dict["f1"],
                    "micro_acc": metrics_dict["acc"],
                    "micro_auc": metrics_dict["auc"],
                    "args": vars(args),
                },
                tempdir,
            )
            print(
                f"✓ Saved epoch {epoch} checkpoint to {tempdir} "
                f"(ACC={metrics_dict['acc']:.4f} AUC={metrics_dict['auc']:.4f} "
                f"microF1={metrics_dict['f1']:.4f})"
            )

    if writer is not None:
        writer.close()
    cleanup_distributed()

if __name__ == "__main__":
    main()
