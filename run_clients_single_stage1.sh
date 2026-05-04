#!/bin/bash
# ===============================================================
# 🚀 Local Multi-Center Training Launcher
# 目标：按联邦训练的 GPU 分配方式，分别启动 Brown / INSPECT / JHU
# ===============================================================

EXP_DIR="stage1_local/Single"
BASE_PORT=29500
SCRIPT="train_stage1_local.py"
LOG_DIR="${EXP_DIR}/local_logs"
mkdir -p "${LOG_DIR}"

# 格式: CENTER_NAME:GPU_START:GPU_COUNT:PORT_OFFSET:EPOCHS
CENTERS=(
  "Brown:0:5:0:50"
  "INSPECT:5:2:10:50"
  "JHU:7:1:20:100"
)

echo "============================================"
echo "🚀 Launching 3-center local DDP training"
echo "============================================"

for entry in "${CENTERS[@]}"; do
  IFS=":" read -r NAME GPU_START GPU_COUNT PORT_OFFSET EPOCHS <<< "$entry"

  CUDA_VISIBLE_DEVICES=$(seq -s, ${GPU_START} $((GPU_START + GPU_COUNT - 1)))
  MASTER_PORT=$((BASE_PORT + PORT_OFFSET))
  LOGFILE="${LOG_DIR}/${NAME}_$(date +%Y%m%d_%H%M%S).log"

  echo "▶️ 启动中心: ${NAME}"
  echo "   → GPUs: ${CUDA_VISIBLE_DEVICES}"
  echo "   → GPU_COUNT: ${GPU_COUNT}"
  echo "   → EPOCHS: ${EPOCHS}"
  echo "   → MASTER_PORT: ${MASTER_PORT}"
  echo "   → 日志: ${LOGFILE}"
  echo "------------------------------------------"

  nohup bash -c "
    export OMP_NUM_THREADS=4
    export TORCH_DISTRIBUTED_DEBUG=OFF
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}

    torchrun --nproc_per_node=${GPU_COUNT} \
             --master_port=${MASTER_PORT} \
             ${SCRIPT} \
             --setname ${NAME} \
             --exp-path ${EXP_DIR} \
             --epochs ${EPOCHS}
  " > "${LOGFILE}" 2>&1 &
done

wait
echo "✅ Brown / INSPECT / JHU 三个中心 local training 全部完成。"
echo "📄 查看日志: tail -f ${LOG_DIR}/*.log"