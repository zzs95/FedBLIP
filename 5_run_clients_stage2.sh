#!/bin/bash
# ================================================================
# 🚀 Local Multi-Client Federated Launcher (with Dataset Settings)
# ================================================================

SERVER_URL="http://127.0.0.1:8008"
MAX_ROUND=5
BASE_PORT=29500

SCRIPT="train_client_autoloop_stage2.py"
LOG_DIR="stage2_fed_logs"
EXP_DIR="stage2_exps_logs"
STYLE_LEARN=1 
mkdir -p $LOG_DIR
mkdir -p $EXP_DIR

# ================================================================
# CLIENT SETTINGS
# 格式:
# CLIENT_NAME:CLIENT_ID:GPU_START:GPU_COUNT:PORT_OFFSET:EPOCHS_LOCAL:VIS_ROOT
# ================================================================
# CLIENTS=(
#   "Brown:0:0:5:0:1:./stage1_img_feat/Brown"
#   "INSPECT:1:5:2:10:1:./stage1_img_feat/INSPECT"
#   "JHU:2:7:1:20:2:./stage1_img_feat/JHU"
# )
CLIENTS=(
  "Brown:0:0:5:0:1:./stage1_img_feat/INSPECT"
  "INSPECT:1:5:2:10:1:./stage1_img_feat/INSPECT"
  "JHU:2:7:1:20:2:./stage1_img_feat/INSPECT"
)
echo ""
echo "=============================================================="
echo "🔥 启动 Federated Clients（包含数据集路径设置）"
echo "=============================================================="
echo ""

for entry in "${CLIENTS[@]}"; do
  IFS=":" read -r NAME CLIENT_ID GPU_START GPU_COUNT PORT_OFFSET EPOCHS_LOCAL VIS_ROOT <<< "$entry"

  CUDA_VISIBLE_DEVICES=$(seq -s, $GPU_START $((GPU_START + GPU_COUNT - 1)))
  MASTER_PORT=$((BASE_PORT + PORT_OFFSET))

  # Build JSON paths
  TRAIN_JSON="${VIS_ROOT}/anno_file_train.json"
  VALID_JSON="${VIS_ROOT}/anno_file_valid.json"

  LOGFILE="${LOG_DIR}/${NAME}_$(date +%Y%m%d_%H%M%S).log"

  echo "▶️ 启动客户端: ${NAME}"
  echo "   → GPUs:            ${CUDA_VISIBLE_DEVICES}"
  echo "   → CLIENT_ID:      ${CLIENT_ID}"
  echo "   → VIS_ROOT:       ${VIS_ROOT}"
  echo "   → TRAIN_JSON:     ${TRAIN_JSON}"
  echo "   → VALID_JSON:     ${VALID_JSON}"
  echo "   → EPOCHS_LOCAL:   ${EPOCHS_LOCAL}"
  echo "   → STYLE_LEARN:    ${STYLE_LEARN}"
  echo "   → 日志:            ${LOGFILE}"
  echo "-----------------------------------------------------"

  nohup bash -c "
    export SERVER_URL=$SERVER_URL
    export CLIENT_NAME=$NAME
    export MAX_ROUND=$MAX_ROUND
    export EPOCHS_LOCAL=$EPOCHS_LOCAL

    export CLIENT_ID=$CLIENT_ID
    export VIS_ROOT=$VIS_ROOT
    export TRAIN_JSON=$TRAIN_JSON
    export VALID_JSON=$VALID_JSON

    export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
    export OMP_NUM_THREADS=8
    export LOG_DIR=$LOG_DIR
    export EXP_DIR=$EXP_DIR
    export STYLE_LEARN=$STYLE_LEARN

    torchrun --nproc_per_node=$GPU_COUNT \
             --master_port=$MASTER_PORT \
             $SCRIPT \
             --client_id $CLIENT_ID \
             --vis_root $VIS_ROOT \
             --train_ann $TRAIN_JSON \
             --val_ann $VALID_JSON 

  " > $LOGFILE 2>&1 &

done

echo "✅ 所有客户端已启动。查看日志: tail -f ${LOG_DIR}/*.log"
echo "🛑 终止所有客户端可执行: pkill -f ${SCRIPT}"

