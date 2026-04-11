import math

# --------------------------
# Warmup + Cosine
# --------------------------
class WarmupCosine:
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, eta_min=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.eta_min = eta_min
        self.last_epoch = 0

    def step(self):
        self.last_epoch += 1
        lr = self.get_lr(self.last_epoch)
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        return lr

    def get_lr(self, epoch):
        if epoch <= self.warmup_epochs:
            return self.base_lr * epoch / max(1, self.warmup_epochs)
        # 余弦衰减（不重启）
        t = epoch - self.warmup_epochs
        T = max(1, self.total_epochs - self.warmup_epochs)
        cos_out = 0.5 * (1.0 + math.cos(math.pi * t / T))
        return self.eta_min + (self.base_lr - self.eta_min) * cos_out