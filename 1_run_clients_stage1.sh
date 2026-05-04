#!/bin/bash
# ================================================================
# 🚀 Local Multi-Client Federated Launcher (8 GPUs, Brown=5 GPUs)
# 目标：让 Brown / INSPECT / JHU 每轮训练时间相近
# ================================================================

SERVER_URL="http://127.0.0.1:8008"
MAX_ROUND=50
BASE_PORT=29500

# 格式: CLIENT_NAME:GPU_START:GPU_COUNT:PORT_OFFSET:EPOCHS_LOCAL
CLIENTS=(
  "Brown:0:5:0:1"       # 55,600 样本 → 5 GPU, 1 epochs
  "INSPECT:5:2:10:1"    # 23,000 样本 → 2 GPU, 1 epochs
  "JHU:7:1:20:2"        # 5,700  样本 → 1 GPU, 2 epochs
)

# Brown：5 GPU，EPOCHS_LOCAL=1 → 55,600×1/5 = 11,120
# INSPECT：2 GPU，EPOCHS_LOCAL=1 → 23,000×1/2 = 11,500
# JHU：1 GPU，EPOCHS_LOCAL=2 → 5,700×2/1 = 11,400

SCRIPT="train_client_autoloop_stage1.py"
LOG_DIR="stage1_fed_logs"
mkdir -p "${LOG_DIR}"

for entry in "${CLIENTS[@]}"; do
  IFS=":" read -r NAME GPU_START GPU_COUNT PORT_OFFSET EPOCHS_LOCAL <<< "$entry"

  # === 分配 CUDA 设备 ===
  CUDA_VISIBLE_DEVICES=$(seq -s, ${GPU_START} $((GPU_START + GPU_COUNT - 1)))
  
  # === 分配唯一端口 ===
  MASTER_PORT=$((BASE_PORT + PORT_OFFSET))
  LOGFILE="${LOG_DIR}/${NAME}_$(date +%Y%m%d_%H%M%S).log"

  echo "▶️ 启动客户端: ${NAME}"
  echo "   → GPUs: ${CUDA_VISIBLE_DEVICES}"
  echo "   → EPOCHS_LOCAL: ${EPOCHS_LOCAL}"
  echo "   → MASTER_PORT: ${MASTER_PORT}"
  echo "   → 日志: ${LOGFILE}"
  echo "------------------------------------------"

  nohup bash -c "
    export SERVER_URL=${SERVER_URL}
    export CLIENT_NAME=${NAME}
    export MAX_ROUND=${MAX_ROUND}
    export EPOCHS_LOCAL=${EPOCHS_LOCAL}
    export OMP_NUM_THREADS=4
    export TORCH_DISTRIBUTED_DEBUG=OFF
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}

    torchrun --nproc_per_node=${GPU_COUNT} \
             --master_port=${MASTER_PORT} \
             ${SCRIPT}
  " > "${LOGFILE}" 2>&1 &
done

echo "✅ 所有客户端已启动。查看日志: tail -f ${LOG_DIR}/*.log"
echo "🛑 终止所有客户端可执行: pkill -f ${SCRIPT}"
