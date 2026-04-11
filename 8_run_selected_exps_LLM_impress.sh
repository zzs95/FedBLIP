#!/bin/bash

# ===============================
# 固定参数
# ===============================
GPUS="0,1,2,3,4,5,6,7"
MODEL_ID="gpt-oss:20b"
RESULT_FILE="study_findings_pred.jsonl"

# ===============================
# 手动列出多个 exp_path
# ===============================

EXP_PATHS=(
"./stage2_test_result/round_50"
)

SETS=("INSPECT" )
echo "Starting selected experiment runs..."
echo ""

# ===============================
# 循环执行
# ===============================

for EXP_PATH in "${EXP_PATHS[@]}"; do

    if [ ! -d "$EXP_PATH" ]; then
        echo "Skipping (not found): $EXP_PATH"
        continue
    fi
    
    for SETNAME in "${SETS[@]}"; do
        echo "================================="
        echo "EXP: $EXP_PATH"
        echo "SET: $SETNAME"
        echo "================================="

        python LLM_study_impression_writing_ddp.py \
            --setname ${SETNAME} \
            --gpus ${GPUS} \
            --exp_path "${EXP_PATH}" \
            --findings_results_file ${RESULT_FILE} \
            --model_id ${MODEL_ID}

        echo ""
        echo "Finished: $EXP_PATH | $SETNAME"
        echo ""
    done

done   

echo "All selected experiments finished."
