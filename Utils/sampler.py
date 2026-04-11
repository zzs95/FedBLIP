import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
import math

# ========= 假设 reverse_samples_weight 已计算完毕 =========
# reverse_samples_weight: [N]，dtype=torch.float32
# weights = reverse_samples_weight.float()

# ========= 分布式初始化（示例） =========
# dist.init_process_group(backend='nccl')  # 若已初始化可略

# ========= 定义 WeightedDistributedSampler =========
class WeightedDistributedSampler(torch.utils.data.Sampler):
    def __init__(self, weights, num_samples=None, replacement=True,
                 num_replicas=None, rank=None, shuffle=True, seed=0):
        if num_replicas is None:
            num_replicas = dist.get_world_size()
            # print('get_world_size', num_replicas)
        if rank is None:
            rank = dist.get_rank()

        self.weights = weights.float()
        self.replacement = replacement
        self.shuffle = shuffle
        self.seed = seed
        self.num_replicas = num_replicas
        self.rank = rank

        self.num_samples = int(math.ceil(len(weights) / num_replicas)) 
        self.total_size = self.num_samples * num_replicas
        self.epoch = 0

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # 全局采样索引（带替换）
        indices = torch.multinomial(self.weights, self.total_size, self.replacement, generator=g).tolist()

        # 当前 rank 的子集（不重叠）
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
