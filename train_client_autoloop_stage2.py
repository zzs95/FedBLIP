"""
 Federated client training (LAVIS-style, DDP + HTTP server)
 ---------------------------------------------------------
 - 使用 common.dist_utils.init_distributed_mode 初始化分布式
 - 通过 HTTP 与服务器交互：
    /status        获取当前 round
    /global        下载全局权重
    /submit_update 上传本地更新
 - 每一轮：
    1) 从 server 下载最新全局模型并广播到各 GPU
    2) 本地训练 EPOCHS_LOCAL 个 epoch
    3) rank0 上传本地模型权重和样本数
    4) 等待 server 进入下一轮
"""

import argparse
import os
import io
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import requests

from common.dist_utils import init_distributed_mode, get_rank, is_main_process
from datasets_utils.imgfeat_cap import ImgFeatCapDataset
from trainer_fed import Trainer
from models.fedblip_style_qformer import Blip2Qformer
from datasets_utils.abnormality_list_56 import abnormality_dict
organ_num = [len(abnormality_dict[k]) for k in abnormality_dict.keys()]
ABN_NUM = np.sum(organ_num)

# ======================================================
# 🌐 联邦相关配置
# ======================================================
SERVER_URL     = os.environ.get("SERVER_URL", "http://127.0.0.1:8008")
CLIENT_NAME    = os.environ.get("CLIENT_NAME", "INSPECT")
MAX_ROUND      = int(os.environ.get("MAX_ROUND", 10))
EPOCHS_LOCAL   = int(os.environ.get("EPOCHS_LOCAL", 1))
POLL_INTERVAL  = int(os.environ.get("POLL_INTERVAL", 20))

# ---- LR-related global controls ----
LR             = float(os.environ.get("LR", 1e-5)) # 3e-5 -> 1e-5
MIN_LR         = float(os.environ.get("MIN_LR", 1e-6))
WARMUP_RATIO   = float(os.environ.get("WARMUP_RATIO", 0.2))    # 0.1 -> 0.2
WEIGHT_DECAY   = float(os.environ.get("WD", 0.1)) # 1e-4 ->0.1
BATCH_SIZE     = int(os.environ.get("BATCH_SIZE", 17))

EXP_DIR        = os.environ.get("EXP_DIR", "debug_exps_logs")
C_DIR = EXP_DIR + f"/{CLIENT_NAME}"
os.makedirs(EXP_DIR, exist_ok=True)

# ---- Model Global Config ----
NUM_CLIENTS  = int(os.environ.get("NUM_CLIENTS", 3))
STYLE_LEARN  = bool(int(os.environ.get("STYLE_LEARN", 1)))   # 用 0/1 控制

# ======================================================
# ✅ Argument parser
# ======================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Federated LAVIS Training Client")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--log_freq", type=int, default=50)
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--accum_grad_iters", type=int, default=1)
    
    # ========== Dataset (client specific) ==========
    parser.add_argument("--client_id", type=int, default=1)
    parser.add_argument("--vis_root", type=str, default="./stage1_img_feat/INSPECT")
    parser.add_argument("--train_ann", type=str, default="./stage1_img_feat/INSPECT/anno_file_train.json")
    parser.add_argument("--val_ann", type=str, default="./stage1_img_feat/INSPECT/anno_file_train.json")
    args = parser.parse_args()

    local_client2cid = {"INSPECT": 1, "Brown": 0, "JHU": 2}
    args.client_id = local_client2cid.get(CLIENT_NAME, args.client_id)
    # =======================================================
    # 🌐 环境变量
    # =======================================================
    args.batch_size   = BATCH_SIZE
    args.init_lr      = LR
    args.min_lr       = MIN_LR
    args.warmup_ratio = WARMUP_RATIO
    args.weight_decay = WEIGHT_DECAY
    args.epochs       = EPOCHS_LOCAL
    args.output_dir   = C_DIR
    args.abn_num      = ABN_NUM
    args.num_clients  = NUM_CLIENTS
    args.style_learn  = STYLE_LEARN
    return args


# ======================================================
# ✅ Reproducibility
# ======================================================
def setup_seeds(seed):
    seed += get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


# ======================================================
# 🌐 与服务器交互的工具函数
# ======================================================
def get_server_status():
    """轮询服务器状态（当前 round、是否有全局权重等）"""
    try:
        r = requests.get(f"{SERVER_URL}/status", timeout=15)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"⚠️ [status] failed: {e}")
    return None


def download_client_weights(client_name: str):
    """
    从服务器下载“本客户端的完整权重”（server 已经把共享 query/style 写回每个 client_weights 里）
    """
    try:
        # ✅ 关键变化：带 client_name
        r = requests.get(
            f"{SERVER_URL}/global",
            params={"client_name": client_name},
            timeout=120
        )
        if r.status_code == 200:
            sd = torch.load(io.BytesIO(r.content), map_location="cpu")
            print(f"✓ Downloaded weights for client={client_name} from server.")
            return sd
        else:
            try:
                msg = r.json()
            except Exception:
                msg = {"ok": False, "msg": r.text[:200]}
            print(f"ℹ️ /global not ready or failed: status={r.status_code}, resp={msg}")
            return None
    except Exception as e:
        print(f"⚠️ Failed to download client weights: {e}")
        return None


def upload_local_update(round_idx, model_state_dict, num_samples, val_loss_prev, val_loss_curr):
    """上传本地模型更新到服务器"""
    buf = io.BytesIO()
    torch.save(model_state_dict, buf)
    buf.seek(0)
    files = {"weights_file": ("weights.pth", buf, "application/octet-stream")}
    data = {
        "client_name": CLIENT_NAME,
        "round_idx": str(round_idx),
        "num_samples": str(num_samples),
        'val_loss_prev': val_loss_prev,
        'val_loss_curr': val_loss_curr,
    }
    try:
        r = requests.post(f"{SERVER_URL}/submit_update", files=files, data=data, timeout=120)
        r.raise_for_status()
        print(f"📤 Uploaded update for round {round_idx} ({num_samples} samples).")
        return True
    except Exception as e:
        print(f"⚠️ Upload failed: {e}")
        return False


def broadcast_state_dict_from_rank0(model):
    """
    在 DDP 多卡间同步 rank0 的最新模型权重
    model: 可能是 DDP 包裹后的模型
    """
    if not dist.is_available() or not dist.is_initialized():
        return

    obj_list = [None]
    rank = dist.get_rank()

    # 只在 rank0 序列化 state_dict
    if rank == 0:
        module = model.module if hasattr(model, "module") else model
        buf = io.BytesIO()
        torch.save(module.state_dict(), buf)
        obj_list[0] = buf.getvalue()

    dist.broadcast_object_list(obj_list, src=0)

    # 其它 rank 反序列化并加载
    if rank != 0 and obj_list[0] is not None:
        module = model.module if hasattr(model, "module") else model
        sd = torch.load(io.BytesIO(obj_list[0]), map_location="cpu")
        module.load_state_dict(sd, strict=False)


# ======================================================
# ✅ Main federated client logic
# ======================================================
def main():
    args = parse_args()

    # ===== 1. 初始化分布式环境 =====
    init_distributed_mode(args)
    setup_seeds(args.seed)

    if is_main_process():
    # 输出最终配置（rank0 显示）
        print("🚀 Federated distributed training client initialized.")
        print("========== Federated Config ==========")
        print(f"World size: {os.environ.get('WORLD_SIZE', 1)} | Rank: {get_rank()}")
        print(f"Using GPU: {args.gpu}")
        print(f"Output directory: {args.output_dir}")
        print(f"SERVER_URL = {SERVER_URL}")
        print(f"CLIENT_NAME = {CLIENT_NAME}")
        print(f"MAX_ROUND = {MAX_ROUND} | EPOCHS_LOCAL = {EPOCHS_LOCAL}")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
        print("========== LR Config ==========")
        print(f"BATCH_SIZE     = {args.batch_size}")
        print(f"LR(init)       = {args.init_lr}")
        print(f"MIN_LR         = {args.min_lr}")
        print(f"WARMUP_RATIO   = {args.warmup_ratio}")
        print("========== Model Config =========")
        print(f"ABN num        = {args.abn_num}")
        print(f"client_id      = {args.client_id}")
        print(f"NUM_CLIENTS    = {args.num_clients}")
        print(f"do style learn = {args.style_learn}")
        print("================================================")


    # ===== 2. 构建 Dataset =====
    if is_main_process():
        print("📦 Loading datasets...")
    train_ds = ImgFeatCapDataset(vis_root=args.vis_root, ann_path=args.train_ann)
    val_ds = ImgFeatCapDataset(vis_root=args.vis_root, ann_path=args.val_ann)

    # ===== 3. 构建模型 + Trainer =====
    if is_main_process():
        print("🧠 Building model...")

    model = Blip2Qformer(abn_num=args.abn_num, organ_num=organ_num, client_id=args.client_id, num_clients=args.num_clients, style_learn=bool(args.style_learn))

    trainer = Trainer(model, train_ds, val_ds, args)

    if is_main_process():
        print("🏁 Entering federated training loop...\n")

    local_round = 0
    val_loss_prev = 1
    # ==================================================
    # 🔁 联邦主循环
    # ==================================================
    while local_round < MAX_ROUND:
        # --- 1️⃣ 查询服务器状态 ---
        status = get_server_status()
        if not status:
            time.sleep(POLL_INTERVAL)
            continue

        server_round = status.get("current_round", 0)

        # 根据 server 进度更新本地轮数
        if server_round > local_round:
            local_round = server_round
        elif server_round < local_round:
            if is_main_process():
                print(f"⏸ Waiting for server (server={server_round}, local={local_round})...")
            time.sleep(POLL_INTERVAL)
            continue

        if is_main_process():
            print(f"\n================= 🌍 Federated Round {local_round} =================")

        # --- 2️⃣ 下载全局模型（仅 rank0）并广播 ---
        if is_main_process():
            sd = download_client_weights(CLIENT_NAME)
            if sd is not None:
                module = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
                module.load_state_dict(sd, strict=False)

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            broadcast_state_dict_from_rank0(trainer.model)
            dist.barrier()

        # --- 4️⃣ 本地训练若干 epoch ---
        train_stats, val_loss_dict = trainer.train_local_round(
            round_idx=local_round,
            epochs_local=EPOCHS_LOCAL,
        )
        # val_loss = val_loss_dict['loss']
        val_cls_loss = val_loss_dict['loss_cls']
        
        # --- 5️⃣ 上传本地更新（仅 rank0） ---
        if is_main_process():
            module = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
            success = upload_local_update(local_round, module.state_dict(), num_samples=len(train_ds), val_loss_prev=val_loss_prev, val_loss_curr=val_cls_loss)
            val_loss_prev = val_cls_loss
            
            if not success:
                print("⚠️ Upload failed, will retry in next polling cycle.")
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        if is_main_process():
            print(f"✅ Completed local training for round {local_round}. Waiting for server to enter next round...")

        # --- 6️⃣ 等待服务器进入下一轮 ---
        while True:
            status_now = get_server_status()
            server_round_now = status_now.get("current_round", 0) if status_now else 0

            if is_main_process():
                print(f"🔁 check server_round: {server_round_now}")

            if status_now and server_round_now >= local_round + 1:
                local_round = server_round_now
                if is_main_process():
                    print(f"🚀 Server entered round {local_round}, continuing training...")
                break
            time.sleep(max(2, POLL_INTERVAL // 3))

    # ===== 结束 =====
    if is_main_process() and trainer.writer is not None:
        trainer.writer.close()
        print("🏁 All federated rounds completed.")


if __name__ == "__main__":
    main()
