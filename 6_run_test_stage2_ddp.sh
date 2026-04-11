#!/bin/bash
# ================================================================
# 🚀 Multi-GPU DDP Testing Launcher (per-client checkpoint)
#    - Each client runs its own DDP inference job
#    - Writes part files per rank and merges on rank0 (inside script)
# ================================================================

set -e

BASE_PORT=29600

# ✅ your DDP testing python
SCRIPT="test_stage2_ddp.py"

EXP_DIR="./"   # <-- your exp_dir
LOG_DIR="${EXP_DIR}/stage2_test_logs"
GLOBAL_ROUND=50                              # <-- which round to test
STYLE_LEARN=1  
ABN_NUM=56
NUM_CLIENTS=3
BATCH_SIZE=20
NUM_WORKERS=10

mkdir -p "$LOG_DIR"

# ================================================================
# CLIENT SETTINGS
# Format:
# NAME:CLIENT_ID:GPU_START:GPU_COUNT:PORT_OFFSET:VIS_ROOT:TEST_JSON
# ================================================================
CLIENTS=(
  # "Brown:0:0:5:0:./stage1_img_feat/Brown"
  "INSPECT:1:5:2:10:./stage1_img_feat/INSPECT"
  # "JHU:2:7:1:20:./stage1_img_feat/JHU"
)

echo ""
echo "=============================================================="
echo "🔥 Launch DDP Testing Jobs (per client)"
echo "=============================================================="
echo "SCRIPT       = $SCRIPT"
echo "EXP_DIR      = $EXP_DIR"
echo "GLOBAL_ROUND = $GLOBAL_ROUND"
echo "LOG_DIR      = $LOG_DIR"
echo "=============================================================="
echo ""

for entry in "${CLIENTS[@]}"; do
  IFS=":" read -r NAME CLIENT_ID GPU_START GPU_COUNT PORT_OFFSET VIS_ROOT <<< "$entry"

  CUDA_VISIBLE_DEVICES=$(seq -s, $GPU_START $((GPU_START + GPU_COUNT - 1)))
  MASTER_PORT=$((BASE_PORT + PORT_OFFSET))
  LOGFILE="${LOG_DIR}/TEST_${NAME}_R${GLOBAL_ROUND}_$(date +%Y%m%d_%H%M%S).log"
  TEST_JSON="${VIS_ROOT}/anno_file_test.json"

  echo "▶️ Client: ${NAME}"
  echo "   → GPUs:         ${CUDA_VISIBLE_DEVICES} (nproc=${GPU_COUNT})"
  echo "   → MASTER_PORT:  ${MASTER_PORT}"
  echo "   → CLIENT_ID:    ${CLIENT_ID}"
  echo "   → VIS_ROOT:     ${VIS_ROOT}"
  echo "   → TEST_JSON:    ${TEST_JSON}"
  echo "   → ROUND:        ${GLOBAL_ROUND}"
  echo "   → STYLE_LEARN:  ${STYLE_LEARN}"
  echo "   → LOG:          ${LOGFILE}"
  echo "-----------------------------------------------------"

  nohup bash -c "
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
    export OMP_NUM_THREADS=8

    torchrun --nproc_per_node=${GPU_COUNT} \
             --master_port=${MASTER_PORT} \
             ${SCRIPT} \
             --client_name ${NAME} \
             --client_id ${CLIENT_ID} \
             --vis_root ${VIS_ROOT} \
             --test_ann ${TEST_JSON} \
             --exp_dir ${EXP_DIR} \
             --global_round ${GLOBAL_ROUND} \
             --style_learn ${STYLE_LEARN} \
             --abn_num ${ABN_NUM} \
             --num_clients ${NUM_CLIENTS} \
             --batch_size ${BATCH_SIZE} \
             --num_workers ${NUM_WORKERS}
  " > "${LOGFILE}" 2>&1 &

done

echo "✅ 所有客户端已启动。查看日志: tail -f ${LOG_DIR}/*.log"
echo "🛑 终止所有客户端可执行: pkill -f ${SCRIPT}"
