# trainer_fed.py

import os
import json
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from common.dist_utils import (
    get_rank,
    is_main_process,
    main_process,
    is_dist_avail_and_initialized
)


# =========================================================
# ✅ 设置统一 logging
# =========================================================
def setup_logger(output_dir: str):
    """设置 logging 格式并输出到文件 + 控制台"""
    log_file = os.path.join(output_dir, "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a"),
            logging.StreamHandler(),
        ],
    )
    if is_main_process():
        logging.info(f"Logging initialized. Output → {log_file}")


# =========================================================
# ✅ BaseTask：定义统一的训练、验证接口
# =========================================================
class BaseTask:
    def train_step(self, model, samples):
        """一次前向传播，返回主 loss 和子项"""
        output = model(samples)
        loss_dict = {k: v for k, v in output.items()}
        return output["loss"], loss_dict

    def valid_step(self, model, samples):
        """默认与 train_step 相同，可被子类重写"""
        return self.train_step(model, samples)

    @torch.no_grad()
    def evaluation(self, model, data_loader, device="cuda"):
        model.eval()
        losses = []
        losses_itc = []
        losses_cls = []
        losses_lm = []
        losses_style = []
        loss_sep = []
        loss_align = []
        center_cos = []
        center_delta_l2= []
        inter_cos_max= []
        inter_cos_mean= []
        inter_cos_min= []
        # losses_gt_txt_cls = []
        # losses_pred_txt_cls = []
        for samples in data_loader:
            if not isinstance(samples, dict):
                continue
            samples = {k: v.to(device) if torch.is_tensor(v) else v for k, v in samples.items()}
            loss, loss_dict = self.valid_step(model, samples)
            losses.append(float(loss.detach().cpu()))
            losses_itc.append(float(loss_dict['loss_itc'].detach().cpu()))
            losses_cls.append(float(loss_dict['loss_cls'].detach().cpu()))
            losses_lm.append(float(loss_dict['loss_lm'].detach().cpu()))
            losses_style.append(float(loss_dict['loss_style'].detach().cpu()))
            loss_sep.append(float(loss_dict['loss_sep'].detach().cpu()))
            loss_align.append(float(loss_dict['loss_align'].detach().cpu()))
            center_cos.append(float(loss_dict['center_cos'].detach().cpu()))
            center_delta_l2.append(float(loss_dict['center_delta_l2'].detach().cpu()))
            inter_cos_max.append(float(loss_dict['inter_cos_max'].detach().cpu()))
            inter_cos_mean.append(float(loss_dict['inter_cos_mean'].detach().cpu()))
            inter_cos_min.append(float(loss_dict['inter_cos_min'].detach().cpu()))
            # # losses_gt_txt_cls.append(float(loss_dict['loss_gt_txt_cls'].detach().cpu()))
            # # losses_pred_txt_cls.append(float(loss_dict['loss_pred_txt_cls'].detach().cpu()))
        return {
            'loss': sum(losses) / max(len(losses), 1), 
            'loss_itc': sum(losses_itc) / max(len(losses_itc), 1), 
            'loss_cls': sum(losses_cls) / max(len(losses_cls), 1), 
            'loss_lm': sum(losses_lm) / max(len(losses_lm), 1), 
            'loss_style': sum(losses_style) / max(len(losses_style), 1), 
            'loss_sep': sum(loss_sep) / max(len(loss_sep), 1), 
            'loss_align': sum(loss_align) / max(len(loss_align), 1), 
            'center_cos': sum(center_cos) / max(len(center_cos), 1), 
            'center_delta_l2': sum(center_delta_l2) / max(len(center_delta_l2), 1), 
            'inter_cos_max': sum(inter_cos_max) / max(len(inter_cos_max), 1), 
            'inter_cos_mean': sum(inter_cos_mean) / max(len(inter_cos_mean), 1), 
            'inter_cos_min': sum(inter_cos_min) / max(len(inter_cos_min), 1), 
            # # # 'loss_gt_txt_cls': sum(losses_gt_txt_cls) / max(len(losses_gt_txt_cls), 1), 
            # # # 'loss_pred_txt_cls': sum(losses_pred_txt_cls) / max(len(losses_pred_txt_cls), 1), 
                }

    def train_epoch(
        self,
        epoch,
        round_idx,
        model,
        data_loader,
        optimizer,
        lr_scheduler=None,
        scaler=None,
        device="cuda",
        log_freq=50,
        accum_grad_iters=1,
        writer=None,
        global_step_offset=0,
    ):
        model.train()
        total_loss = 0.0
        use_amp = scaler is not None
        global_step = global_step_offset

        for i, batch in enumerate(data_loader):
            if not isinstance(batch, dict):
                continue
            # samples = {k: v.to(device) if torch.is_tensor(v) else v for k, v in samples.items()} 
            samples = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    samples[k] = v.to(device, non_blocking=True)
                else:
                    samples[k] = list(map(list, zip(*v)))

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss, loss_dict = self.train_step(model, samples)
                # if round_idx > 10: # update 12262025
                if round_idx > 30: # update 12312025
                    loss = loss + loss_dict['loss_style']
                loss = loss / max(accum_grad_iters, 1)

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (i + 1) % max(accum_grad_iters, 1) == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            if lr_scheduler is not None:
                lr_scheduler.step()

            total_loss += float(loss.detach().cpu())
            global_step += 1

            # === TensorBoard ===
            if writer is not None and is_main_process():
                cur_lr = optimizer.param_groups[0]["lr"]
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/loss_itc", loss_dict['loss_itc'].item(), global_step)
                writer.add_scalar("train/loss_cls", loss_dict['loss_cls'].item(), global_step)
                writer.add_scalar("train/loss_lm", loss_dict['loss_lm'].item(), global_step)
                writer.add_scalar("train/loss_style", loss_dict['loss_style'].item(), global_step)
                writer.add_scalar("train/loss_sep", loss_dict['loss_sep'].item(), global_step)
                writer.add_scalar("train/loss_align", loss_dict['loss_align'].item(), global_step)
                writer.add_scalar("train/center_cos", loss_dict['center_cos'].item(), global_step)
                writer.add_scalar("train/center_delta_l2", loss_dict['center_delta_l2'].item(), global_step)
                writer.add_scalar("train/inter_cos_max", loss_dict['inter_cos_max'].item(), global_step)
                writer.add_scalar("train/inter_cos_mean", loss_dict['inter_cos_mean'].item(), global_step)
                writer.add_scalar("train/inter_cos_min", loss_dict['inter_cos_min'].item(), global_step)
                # # writer.add_scalar("train/loss_gt_txt_cls", loss_dict['loss_gt_txt_cls'].item(), global_step)
                # # writer.add_scalar("train/loss_pred_txt_cls", loss_dict['loss_pred_txt_cls'].item(), global_step)
                writer.add_scalar("train/lr", cur_lr, global_step)

            # === Logging ===
            if (i + 1) % log_freq == 0 and is_main_process():
                cur_lr = optimizer.param_groups[0]["lr"]
                logging.info(
                    f"[Rank {get_rank()} | Epoch {epoch} | Step {i+1}/{len(data_loader)}] "
                    f"Loss={loss.item():.4f} | LR={cur_lr:.6e}"
                )

        avg_loss = total_loss / max(len(data_loader), 1)
        if is_main_process():
            logging.info(f"✅ Epoch {epoch} finished. Avg loss={avg_loss:.4f}")
        return {"loss": avg_loss, "steps": global_step}


# =========================================================
# ✅ Trainer
# =========================================================
class Trainer:
    def __init__(self, model, train_dataset, val_dataset, args):
        self.args = args
        self.task = BaseTask()

        # ====== 输出目录 + Logging ======
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if is_main_process():
            setup_logger(str(self.output_dir))
            self.writer = SummaryWriter(log_dir=self.output_dir / "runs")
        else:
            self.writer = None

        # ====== 设备 ======
        self.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

        # ====== 分布式信息（外部已初始化） ======
        self.distributed = getattr(args, "distributed", False)
        self.rank = getattr(args, "rank", 0)

        # ====== 模型 + DDP ======
        self.model = model.to(self.device)
        if self.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True,
                # broadcast_buffers=False,   # ✅ 如果buffer报错
            )

        # ====== 数据加载 ======
        if self.distributed:
            train_sampler = DistributedSampler(train_dataset, shuffle=True)
            val_sampler = DistributedSampler(val_dataset, shuffle=False)
        else:
            train_sampler = None
            val_sampler = None

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            sampler=val_sampler,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        # ====== 优化器 / Scheduler / AMP 在单独函数中构建 ======
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.global_step = 0  # 用于 TensorBoard 递增

        # ⚠️ 只在 __init__ 中初始化一次，不要在每个 round 重建
        self.build_optim_sched()

    # =====================================================
    # 🔧 全局一次性构建 optimizer + scheduler + scaler
    # =====================================================
    def build_optim_sched(self):
        """全局一次性构建 optimizer / scheduler / scaler（不在 federated round 重置）"""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.init_lr,
            weight_decay=self.args.weight_decay,
        )

        steps_per_epoch = max(len(self.train_loader), 1)

        # 🔥 全局总训练步数 = steps_per_epoch × 每轮本地 epoch × 全局round数量
        max_round = int(os.environ.get("MAX_ROUND", 150))
        epochs_local = getattr(self.args, "epochs", 1)   # ← 注意这里改成 epochs
        total_steps = steps_per_epoch * epochs_local * max_round
        
        warmup_ratio = getattr(self.args, "warmup_ratio", 0.1)
        warmup_steps = int(total_steps * warmup_ratio)

        # 这里直接用一个固定 start_factor 即可（比如从 1% LR 推到 full LR）
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=max(warmup_steps, 1),
        )

        # 🔥 Cosine 衰减总长度 = total_steps - warmup_steps
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(total_steps - warmup_steps, 1),
            eta_min=self.args.min_lr,
        )

        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.args.use_amp)

        if is_main_process():
            logging.info(
                f"⚙️ Scheduler initialized: total_steps={total_steps}, "
                f"warmup_steps={warmup_steps}, warmup_ratio={warmup_ratio}"
            )

    # =====================================================
    # 🔁 联邦模式下的一轮本地训练（多 epoch）
    # =====================================================
    def train_local_round(self, round_idx: int, epochs_local: int):
        """
        用于联邦学习：在指定的 round 内，本地训练若干 epoch。
        - 不做 checkpoint 保存（由 server 负责聚合全局模型）
        - 仍然可以跑一次 validation + TensorBoard 记录
        """
        if is_main_process():
            logging.info(f"🌐 [Round {round_idx}] Local training for {epochs_local} epoch(s).")

        global_step = self.global_step
        last_train_stats = None

        for e in range(epochs_local):
            epoch_global = round_idx * epochs_local + e

            if self.distributed and hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(epoch_global)

            # === 训练 ===
            train_stats = self.task.train_epoch(
                epoch=epoch_global,
                round_idx=round_idx,
                model=self.model,
                data_loader=self.train_loader,
                optimizer=self.optimizer,
                lr_scheduler=self.scheduler,
                scaler=self.scaler,
                device=self.device,
                log_freq=self.args.log_freq,
                accum_grad_iters=self.args.accum_grad_iters,
                writer=self.writer,
                global_step_offset=global_step,
            )
            global_step = train_stats["steps"]
            self.global_step = global_step
            last_train_stats = train_stats

            if is_main_process():
                self.log_stats(train_stats, "train", epoch_global)

        # === Round 结束时跑一次验证 ===
        val_loss_dict = self.task.evaluation(self.model, self.val_loader, device=self.device)
        if is_main_process():
            logging.info(f"🔍 [Round {round_idx}] Validation Loss = {val_loss_dict['loss']:.4f}")
            if self.writer is not None:
                self.writer.add_scalar("val/loss_round", val_loss_dict['loss'], round_idx)
                self.writer.add_scalar("val/loss_itc_round", val_loss_dict['loss_itc'], round_idx)
                self.writer.add_scalar("val/loss_cls_round", val_loss_dict['loss_cls'], round_idx)
                self.writer.add_scalar("val/loss_lm_round", val_loss_dict['loss_lm'], round_idx)
                self.writer.add_scalar("val/loss_style_round", val_loss_dict['loss_style'], round_idx)
                self.writer.add_scalar("val/center_cos_round", val_loss_dict['center_cos'], round_idx)
                self.writer.add_scalar("val/center_delta_l2_round", val_loss_dict['center_delta_l2'], round_idx)
                self.writer.add_scalar("val/inter_cos_max", val_loss_dict['inter_cos_max'], round_idx)
                self.writer.add_scalar("val/inter_cos_mean", val_loss_dict['inter_cos_mean'], round_idx)
                self.writer.add_scalar("val/inter_cos_min", val_loss_dict['inter_cos_min'], round_idx)
                # # self.writer.add_scalar("val/loss_gt_txt_cls_round", val_loss_dict['loss_gt_txt_cls'], round_idx)
                # # self.writer.add_scalar("val/loss_pred_txt_cls_round", val_loss_dict['loss_pred_txt_cls'], round_idx)

        return last_train_stats if last_train_stats is not None else {"loss": 0.0, "steps": global_step}, val_loss_dict

    # =====================================================
    @main_process
    def log_stats(self, stats, split_name, epoch):
        stats = {f"{split_name}_{k}": float(v) if isinstance(v, torch.Tensor) else v for k, v in stats.items()}
        stats["epoch"] = int(epoch)
        log_file = self.output_dir / "log.txt"
        with open(log_file, "a") as f:
            f.write(json.dumps(stats) + "\n")
        logging.info(f"📘 Logged {split_name} stats for epoch {epoch}")
