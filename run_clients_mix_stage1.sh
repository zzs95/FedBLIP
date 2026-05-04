#!/bin/bash
# ===============================================================
# 🚀 Federated Learning Client Launcher (Multi-site DDP)
# ===============================================================
# 功能：分别启动 Brown / INSPECT / JHU 三个中心的多卡 DDP 训练进程
# 每个中心独立连接到中央 FedAvg Server（FastAPI）
# ===============================================================

# === 全局配置 ===
EXP_DIR="stage1_local"

# ===  GPU 数量配置 ===
MIX_GPUS=8

# === 启动各中心 ===
echo "============================================"
echo "🚀 Launching Single Client Training"
echo "============================================"

echo "[Mix] Starting ${MIX_GPUS} GPU client..."
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
  --nproc_per_node=${MIX_GPUS} \
  --master_port=29500 \
  train_stage1_local.py \
  --setname Mix \
  --exp-path "${EXP_DIR}" 

# === 监控所有进程 ===
wait
echo "✅ All clients finished local training and uploaded updates."
