#!/bin/bash

# ===============================
# 固定参数
# ===============================
GPUS="0,1,2,3,4,5,6,7"
MODEL_ID="gpt-oss:20b"
RESULT_FILE="output_results_abn_text_probs.jsonl"

# ===============================
# 手动列出多个 exp_path
# ===============================
/media/brownradx/ssd_code/Projects_zhusi/abn_blip
EXP_PATHS=(
"./stage2_test_result/round_50"
)

# SETS=("Brown" "JHU" "INSPECT" )
SETS=( "INSPECT" )
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

        python LLM_study_findings_writing_ddp.py \
            --setname ${SETNAME} \
            --gpus ${GPUS} \
            --exp_path "${EXP_PATH}" \
            --results_file ${RESULT_FILE} \
            --model_id ${MODEL_ID}

        echo ""
        echo "Finished: $EXP_PATH | $SETNAME"
        echo ""
    done

done  

echo "All selected experiments finished."
