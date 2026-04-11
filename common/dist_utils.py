"""
 Copyright (c) 2022, Salesforce
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
"""

import datetime
import functools
import os
import torch
import torch.distributed as dist
import timm.models.hub as timm_hub


# ============================================================
# 🔹 打印控制：仅主进程输出
# ============================================================
def setup_for_distributed(is_master: bool):
    """Disable printing when not in master process."""
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


# ============================================================
# 🔹 基本分布式状态查询
# ============================================================
def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def get_rank() -> int:
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def is_main_process() -> bool:
    return get_rank() == 0


# ============================================================
# 🔹 初始化分布式模式
# ============================================================
def init_distributed_mode(args):
    """
    Initialize torch distributed environment.
    Support torchrun / slurm / single-node setups.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ.get("LOCAL_RANK", 0))
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("⚠️ Not using distributed mode (single process)")
        args.distributed = False
        args.rank = 0
        args.world_size = 1
        args.gpu = 0
        return

    args.distributed = True
    torch.cuda.set_device(args.gpu)

    # 默认 URL
    if not hasattr(args, "dist_url"):
        args.dist_url = "env://"

    args.dist_backend = "nccl"
    print(
        f"| distributed init (rank {args.rank}, world {args.world_size}, url {args.dist_url})",
        flush=True,
    )

    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
        timeout=datetime.timedelta(days=365),  # allow auto-downloads to complete
    )

    dist.barrier()  # 同步所有进程
    setup_for_distributed(args.rank == 0)


# ============================================================
# 🔹 获取 rank/world 信息
# ============================================================
def get_dist_info():
    """Return (rank, world_size), safe for both dist and non-dist modes."""
    if is_dist_avail_and_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


# ============================================================
# 🔹 仅主进程执行装饰器
# ============================================================
def main_process(func):
    """Decorator: run only on rank 0."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)
    return wrapper


# ============================================================
# 🔹 下载缓存文件（分布式安全）
# ============================================================
def download_cached_file(url, check_hash=True, progress=False):
    """
    Download a file from URL and cache it locally (via timm).
    Only rank-0 process downloads; others wait at barrier.
    """

    def get_cached_file_path():
        # sync local path across processes
        from urllib.parse import urlparse
        parts = urlparse(url)
        filename = os.path.basename(parts.path)
        return os.path.join(timm_hub.get_cache_dir(), filename)

    if is_main_process():
        timm_hub.download_cached_file(url, check_hash, progress)

    if is_dist_avail_and_initialized():
        dist.barrier()  # ensure file downloaded before others proceed

    return get_cached_file_path()
