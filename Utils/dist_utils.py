import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import random
from datetime import timedelta
# --------------------------
# 实用函数
# --------------------------
def setup_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def is_main_process() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

def ddp_print(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)
        
import socket   

def find_free_port():
    """返回一个未被占用的TCP端口"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

def init_distributed(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    else:
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0

    # 获取端口号（优先环境变量，否则自动检测）
    args.MASTER_PORT = os.environ.get("MASTER_PORT", str(find_free_port()))
    print(f"[INIT] Using MASTER_PORT: {args.MASTER_PORT}")

    torch.cuda.set_device(args.local_rank)

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{args.MASTER_PORT}",
        world_size=args.world_size,
        rank=args.rank,
        timeout=timedelta(seconds=1800),
    )
    dist.barrier()

    print(f"[DDP] Initialized: rank={args.rank}, world_size={args.world_size}, local_rank={args.local_rank}")

def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

def reduce_tensor(t: torch.Tensor, op=dist.ReduceOp.SUM):
    """
    在所有进程上聚合张量，并返回平均值（常用于loss）
    """
    rt = t.clone()
    dist.all_reduce(rt, op=op)
    rt /= dist.get_world_size()
    return rt

def all_gather_variable_len_tensor(t: torch.Tensor) -> torch.Tensor:
    """
    在DDP中收集可变长度的张量（按 batch 拼接）
    """
    world_size = dist.get_world_size()
    local_size = torch.tensor([t.size(0)], device=t.device)
    size_list = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    sizes = [int(s.item()) for s in size_list]
    max_size = max(sizes)

    # pad 到相同大小
    if t.size(0) < max_size:
        pad_shape = (max_size - t.size(0),) + tuple(t.shape[1:])
        pad = torch.zeros(pad_shape, dtype=t.dtype, device=t.device)
        t_padded = torch.cat([t, pad], dim=0)
    else:
        t_padded = t

    gather_list = [torch.zeros_like(t_padded) for _ in range(world_size)]
    dist.all_gather(gather_list, t_padded)

    # 去掉 pad，再拼接
    chunks = []
    for g, s in zip(gather_list, sizes):
        chunks.append(g[:s])
    return torch.cat(chunks, dim=0)