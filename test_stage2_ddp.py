
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DDP Multi-GPU Federated-style Testing (Abn-BLIP style)

Usage:
  CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 test_fed_ddp.py \
    --client_name JHU --client_id 2 --global_round 30 \
    --vis_root /media/brownradx/ssd_data2/VLM_PE_feat/fed_feat/JHU \
    --test_ann /media/brownradx/ssd_data2/VLM_PE_feat/fed_feat/JHU/anno_file_test.json \
    --exp_dir file_fedCls_fedBLIP_CR_EMA_no_avg/

Notes:
- Each rank writes: output_results_abn_text_probs.jsonl.part{rank}
- Rank0 merges into: output_results_abn_text_probs.jsonl
"""

import argparse
import io
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from datasets_utils.imgfeat_cap import ImgFeatCapDataset
from models.fedblip_style_qformer import Blip2Qformer


# ----------------------------
# Generation hyper-params
# ----------------------------
max_len = 40
min_len = 8
num_beams = 1
repetition_penalty = 1.15
length_penalty = 0.0
top_p = 0.9
temperature = 1.0


# ======================================================
# ✅ Argument parser
# ======================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Federated LAVIS Testing (DDP multi-GPU)")

    # Basic
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--use_amp", action="store_true", default=True)

    # ---------------------------
    # Dataset (client specific)
    # ---------------------------
    parser.add_argument("--client_id", type=int, default=1)
    parser.add_argument("--client_name", type=str, default="INSPECT")
    parser.add_argument("--vis_root", type=str, default="/media/brownradx/ssd_data2/VLM_PE_feat/fed_feat/INSPECT", )
    parser.add_argument( "--test_ann", type=str, default="/media/brownradx/ssd_data2/VLM_PE_feat/fed_feat/INSPECT/anno_file_test.json", )
    

    # FL settings
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--abn_num", type=int, default=56)
    parser.add_argument("--num_clients", type=int, default=3)
    parser.add_argument("--global_round", type=int, default=30)

    # Checkpoint settings (per-client ckpt)
    parser.add_argument("--style_learn", type=int, default=1)  # 0/1
    parser.add_argument("--exp_dir", type=str, default="./")
    parser.add_argument("--results_file", type=str, default="output_results_abn_text_probs.jsonl")

    args = parser.parse_args()

    # Derived: checkpoint path uses per-client naming: {client}_round_{r}.pth
    ckpt_name = f"{args.client_name}_round_{args.global_round}.pth"
    args.ckpt = os.path.join(args.exp_dir, "stage2_fed_server_store", ckpt_name)

    # Output dir
    args.output_dir = os.path.join(args.exp_dir, "stage2_test_result", f"round_{args.global_round}", args.client_name)
    os.makedirs(args.output_dir, exist_ok=True)

    args.style_learn = bool(args.style_learn)
    
    return args


# ======================================================
# ✅ Reproducibility
# ======================================================
def setup_seeds(seed: int, rank: int = 0):
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


# ======================================================
# ✅ DDP init
# ======================================================
def init_distributed():
    """
    torchrun provides env vars: RANK, WORLD_SIZE, LOCAL_RANK
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        return True, local_rank, rank, world_size
    return False, 0, 0, 1


def is_rank0(ddp_enabled: bool) -> bool:
    return (not ddp_enabled) or (ddp_enabled and dist.get_rank() == 0)


# ======================================================
# ✅ Testing loop
# ======================================================
@torch.no_grad()
def run_test_generate(model, ds_accNums, data_loader, device, save_file, show_pbar: bool):
    model.eval()

    iterator = tqdm(data_loader) if show_pbar else data_loader

    with open(save_file, "a", encoding="utf-8") as f:
        for batch in iterator:
            # move tensors to device
            samples = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    samples[k] = v.to(device, non_blocking=True)
                else:
                    # keep your original handling for non-tensor fields
                    samples[k] = list(map(list, zip(*v)))

            captions = model.generate(
                samples,
                use_nucleus_sampling=False,
                num_beams=num_beams,
                max_length=max_len,
                min_length=min_len,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                top_p=top_p,
                temperature=temperature,
            )

            # dataset indices (unique across ranks due to DistributedSampler)
            image_ids = samples["image_id"].detach().cpu().numpy().tolist()
            img_accNums = [ds_accNums[i] for i in image_ids]

            for caption, ds_idx, accNum in zip(captions, image_ids, img_accNums):
                item = {
                    "image_id": str(ds_idx),  # use dataset index
                    "AccessionNumber_md5": accNum,
                    "abn_text": caption[0],
                    "abn_probs": caption[1],
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


def merge_parts(base_save: str, world_size: int):
    """
    Merge .part{rank} files into base_save. Rank0 only.
    """
    if os.path.exists(base_save):
        os.remove(base_save)

    with open(base_save, "a", encoding="utf-8") as fout:
        for r in range(world_size):
            part = base_save + f".part{r}"
            if not os.path.exists(part):
                print(f"⚠️ Missing part file: {part}")
                continue
            with open(part, "r", encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)


# ======================================================
# ✅ Main
# ======================================================
def main():
    args = parse_args()

    ddp, local_rank, rank, world_size = init_distributed()
    setup_seeds(args.seed, rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_rank0(ddp):
        print("🚀 Federated-style testing initialized (DDP).")
        print(f"WORLD_SIZE      = {world_size}")
        print(f"CLIENT_NAME     = {args.client_name}")
        print(f"Output directory= {args.output_dir}")
        print(f"Using device    = {device}")
        print("========== Data / Model Config =========")
        print(f"BATCH_SIZE      = {args.batch_size}")
        print(f"ABN num         = {args.abn_num}")
        print(f"client_id       = {args.client_id}")
        print(f"NUM_CLIENTS     = {args.num_clients}")
        print(f"do style learn  = {args.style_learn}")
        print(f"Test anno       = {args.test_ann}")
        print(f"Checkpoint      = {args.ckpt}")
        print("================================================")

    # Dataset
    if is_rank0(ddp):
        print("📦 Loading dataset...")
    test_ds = ImgFeatCapDataset(vis_root=args.vis_root, ann_path=args.test_ann)
    if is_rank0(ddp):
        print(f"Test samples: {len(test_ds)}")

    # AccNum map: ds_idx -> accession
    ds_accNums = {v: k for k, v in test_ds.img_ids.items()}

    sampler = None
    if ddp:
        sampler = DistributedSampler(test_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Model
    if is_rank0(ddp):
        print("🧠 Building model...")
    model = Blip2Qformer(
        abn_num=args.abn_num,
        client_id=args.client_id,
        num_clients=args.num_clients,
        style_learn=bool(args.style_learn),
    )

    # Load checkpoint
    if is_rank0(ddp):
        print(f"📥 Loading checkpoint from {args.ckpt} ...")
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    if is_rank0(ddp):
        print(f"📥 Loaded state_dict. missing_keys={len(missing)}, unexpected_keys={len(unexpected)}")

    model.to(device)

    # (Optional) DDP wrap for consistency
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    # Output files
    base_save = os.path.join(args.output_dir, args.results_file)
    part_save = base_save + f".part{rank}"

    # Clear old part file
    if os.path.exists(part_save):
        os.remove(part_save)

    # Run generate (use underlying module if wrapped)
    gen_model = model.module if hasattr(model, "module") else model
    show_pbar = is_rank0(ddp)
    run_test_generate(gen_model, ds_accNums, test_loader, device, part_save, show_pbar=show_pbar)

    # Merge
    if ddp:
        dist.barrier()

    if is_rank0(ddp):
        merge_parts(base_save, world_size)
        print(f"✅ Merged results saved to: {base_save}")

    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

