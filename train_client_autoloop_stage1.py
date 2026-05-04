#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import io
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import requests
from datetime import timedelta

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import (
    SequentialLR, LinearLR, CosineAnnealingWarmRestarts,
    ExponentialLR, ConstantLR
)
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

import monai
import numpy as np
import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")

# ===============================
# 📦 本地模块
# ===============================
from train_stage1_base import ImgABNDataset, train_transform, val_transform, run_one_epoch
from models.image_classifier import ImageClassifier_BASE as ImageClassifier
from datasets_utils.abnormality_list_56 import abnormality_dict


# ===============================
# ⚙️ 基本配置
# ===============================
SERVER_URL    = os.environ.get("SERVER_URL", "http://127.0.0.1:8008")
CLIENT_NAME   = os.environ.get("CLIENT_NAME", "Brown")

MAX_ROUND     = int(os.environ.get("MAX_ROUND", 150))
EPOCHS_LOCAL  = int(os.environ.get("EPOCHS_LOCAL", 2))
BATCH_SIZE    = int(os.environ.get("BATCH_SIZE", 10))

LR            = float(os.environ.get("LR", 3e-5))
WEIGHT_DECAY  = float(os.environ.get("WD", 1e-5))
ETA_MIN       = float(os.environ.get("ETA_MIN", 1e-6))

EXP_DIR       = os.environ.get("EXP_DIR", f"stage1_fed_logs/{CLIENT_NAME}")
os.makedirs(EXP_DIR, exist_ok=True)

POLL_INTERVAL = 20  # 秒


# ===============================
# 📝 logging
# ===============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


# ===============================
# 🧩 初始化 DDP
# ===============================
def init_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank, world_size, local_rank = 0, 1, 0

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(seconds=1800),
    )
    return rank, world_size, local_rank


# ===============================
# 🌐 Server API
# ===============================
def get_server_status():
    try:
        r = requests.get(f"{SERVER_URL}/status", timeout=15)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        logging.warning(f"[status] failed: {e}")
    return None


def download_personalized_weights():
    """
    ✅ 新版：从 server 拉“自己的模型”
    """
    try:
        r = requests.get(
            f"{SERVER_URL}/global",
            params={"client_name": CLIENT_NAME},
            timeout=60
        )
        if r.status_code == 200:
            sd = torch.load(io.BytesIO(r.content), map_location="cpu")
            logging.info("✓ Downloaded personalized model from server")
            return sd
        else:
            logging.info("ℹ️ No personalized weights available yet")
    except Exception as e:
        logging.warning(f"[global] failed: {e}")
    return None


def upload_local_update(
    round_idx: int,
    model: nn.Module,
    num_samples: int,
    val_loss_prev: float,
    val_loss_curr: float,
):
    """
    ✅ 新版 server 需要：
      - num_samples
      - val_loss_prev
      - val_loss_curr
    """
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    buf.seek(0)

    files = {
        "weights_file": ("weights.pth", buf, "application/octet-stream")
    }
    data = {
        "client_name": CLIENT_NAME,
        "round_idx": str(round_idx),
        "num_samples": str(num_samples),
        "val_loss_prev": str(val_loss_prev),
        "val_loss_curr": str(val_loss_curr),
    }

    try:
        r = requests.post(
            f"{SERVER_URL}/submit_update",
            files=files,
            data=data,
            timeout=180,
        )
        r.raise_for_status()
        logging.info(
            f"📤 Uploaded update | round={round_idx} "
            f"| n={num_samples} | Δval={val_loss_prev - val_loss_curr:+.4f}"
        )
        return True
    except Exception as e:
        logging.error(f"[submit_update] failed: {e}")
        return False


def broadcast_state_dict_from_rank0(model: DDP):
    """
    DDP 同步（保持你原来的逻辑）
    """
    obj_list = [None]
    if dist.get_rank() == 0:
        buf = io.BytesIO()
        torch.save(model.module.state_dict(), buf)
        obj_list[0] = buf.getvalue()

    dist.broadcast_object_list(obj_list, src=0)

    if dist.get_rank() != 0 and obj_list[0] is not None:
        sd = torch.load(io.BytesIO(obj_list[0]), map_location="cpu")
        model.module.load_state_dict(sd, strict=False)


# ===============================
# 📦 数据加载
# ===============================
def build_dataloader(args):
    train_list = ImgABNDataset(args.setname, mode="train")
    # train_list = train_list[:300]
    train_ds = monai.data.Dataset(train_list, transform=train_transform)

    sampler = DistributedSampler(
        train_ds,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=True
    )

    loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=min(args.workers, 4),
        pin_memory=False,
        persistent_workers=True,
        prefetch_factor=2,
    )
    return train_ds, loader

def build_val_dataloader(args):
    val_list = ImgABNDataset(args.setname, mode="valid")
    val_ds = monai.data.Dataset(val_list, transform=val_transform)  

    sampler = DistributedSampler(
        val_ds,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=False
    )

    loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=min(args.workers, 4),
        pin_memory=False,
        persistent_workers=True,
    )
    return val_ds, loader

# ===============================
# 🎛️ Scheduler
# ===============================
def build_scheduler(optimizer):
    warmup_epochs = 1 if EPOCHS_LOCAL > 1 else 0
    cosine_epochs = max(1, EPOCHS_LOCAL - warmup_epochs)

    warmup = (
        LinearLR(optimizer, start_factor=0.2, total_iters=warmup_epochs)
        if warmup_epochs > 0
        else ConstantLR(optimizer, factor=1.0, total_iters=1)
    )
    cosine = CosineAnnealingWarmRestarts(
        optimizer, T_0=cosine_epochs, T_mult=2, eta_min=ETA_MIN
    )
    exp_decay = ExponentialLR(optimizer, gamma=0.97)

    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine, exp_decay],
        milestones=[warmup_epochs, warmup_epochs + cosine_epochs],
    )


# ===============================
# 🧠 本地训练
# ===============================
def local_train(
    model, loader, epochs, criterion,
    optimizer, scheduler, scaler,
    device, writer, round_idx
):
    model.train()
    epoch_losses = []

    for ep in range(epochs):
        if isinstance(loader.sampler, DistributedSampler):
            loader.sampler.set_epoch(round_idx * epochs + ep)

        avg_loss, _, _, metrics = run_one_epoch(
            model=model,
            loader=loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            is_validate=False,
        )
        epoch_losses.append(avg_loss)

        if dist.get_rank() == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            logging.info(
                f"[{CLIENT_NAME}] "
                f"Round {round_idx} | Epoch {ep+1}/{epochs} "
                f"| LR={lr_now:.2e} "
                f"| Loss={avg_loss:.4f} "
                f"| F1={metrics['f1']:.3f} "
                f"| AUC={metrics['auc']:.3f}"
            )
            writer.add_scalar("loss", avg_loss, round_idx * epochs + ep)
            writer.add_scalar("f1", metrics["f1"], round_idx * epochs + ep)
            writer.add_scalar("auc", metrics["auc"], round_idx * epochs + ep)
            writer.add_scalar("lr", lr_now, round_idx * epochs + ep)
        scheduler.step()

    return float(np.mean(epoch_losses))

@torch.no_grad()
def evaluate(model, loader, criterion, device, writer=None, round_idx=0):
    val_loss, _, _, val_metrics = run_one_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        optimizer=None,
        scaler=None,
        device=device,
        is_validate=True,
    )

    is_rank0 = (not dist.is_initialized()) or dist.get_rank() == 0

    if is_rank0:
        logging.info(
            f"[{CLIENT_NAME}] Round {round_idx} | Validation "
            f"| Loss={val_loss:.4f} "
            f"| F1={val_metrics.get('f1', 0):.3f} "
            f"| AUC={val_metrics.get('auc', 0):.3f}"
        )

        if writer is not None:
            writer.add_scalar("val/loss", val_loss, round_idx)
            writer.add_scalar("val/f1", val_metrics.get("f1", 0), round_idx)
            writer.add_scalar("val/auc", val_metrics.get("auc", 0), round_idx)

    return val_loss, val_metrics

# ===============================
# 🚀 主流程
# ===============================
def main():
    rank, world_size, local_rank = init_ddp()
    device = torch.device("cuda", local_rank)

    # ---- dataset args ----
    args = type("args", (), {})()
    args.setname = CLIENT_NAME
    args.batch_size = BATCH_SIZE
    args.workers = 4
    args.rank = rank
    args.world_size = world_size

    train_ds, train_loader = build_dataloader(args)
    val_ds, val_loader = build_val_dataloader(args)
    organ_abn_nums = [len(abnormality_dict[k]) for k in abnormality_dict.keys()]

    model = ImageClassifier(region_abn_counts=organ_abn_nums).to(device)

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

    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False
    )

    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda")

    writer = SummaryWriter(
        log_dir=os.path.join(EXP_DIR, "runs")
    ) if rank == 0 else None

    local_round = 0
    prev_val_loss = None

    while local_round < MAX_ROUND:
        status = get_server_status()
        if not status:
            time.sleep(POLL_INTERVAL)
            continue

        server_round = status.get("current_round", 0)

        if server_round < local_round:
            logging.info(
                f"⏸ Waiting server: server={server_round}, local={local_round}"
            )
            time.sleep(POLL_INTERVAL)
            continue

        local_round = server_round

        # ---- 下载个性化模型 ----
        if rank == 0:
            sd = download_personalized_weights()
            if sd is not None:
                model.module.load_state_dict(sd, strict=False)

        dist.barrier()
        broadcast_state_dict_from_rank0(model)
        dist.barrier()

        optimizer = optim.AdamW(
            model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY
        )
        scheduler = build_scheduler(optimizer)

        # ---- 本地训练 ----
        avg_train_loss = local_train(
            model, train_loader, EPOCHS_LOCAL,
            criterion, optimizer, scheduler,
            scaler, device, writer, local_round
        )

        val_loss_curr, val_metrics_curr = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            writer=writer,
            round_idx=local_round,
        )

        val_loss_prev = (
            prev_val_loss if prev_val_loss is not None else val_loss_curr
        )

        # ---- 上传 ----
        if rank == 0:
            upload_local_update(
                round_idx=local_round,
                model=model.module,
                num_samples=len(train_ds),
                val_loss_prev=val_loss_prev,
                val_loss_curr=val_loss_curr,
            )

        prev_val_loss = val_loss_curr
        dist.barrier()

        logging.info(
            f"✅ Finished round {local_round}, waiting for next round..."
        )

        while True:
            st = get_server_status()
            if st and st.get("current_round", 0) >= local_round + 1:
                local_round = st["current_round"]
                break
            time.sleep(max(2, POLL_INTERVAL // 3))

    if writer:
        writer.close()
    dist.destroy_process_group()
    logging.info("🏁 All federated rounds completed")


# ===============================
# 🧭 Entry
# ===============================
if __name__ == "__main__":
    main()
