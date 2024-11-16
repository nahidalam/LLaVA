#!/bin/bash

pip install prettytable openai==0.28

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="maya_toxicity_free_finetuned"
# CKPT="maya_full_ft"
SPLIT="llava-hallusionbench"
EVAL_DIR="./playground/data/eval/hallusion-bench"

mkdir -p ${EVAL_DIR}/answers/${CKPT}

# Convert HallusionBench format to LLaVA format
python scripts/convert_hallusionbench_for_eval.py \
    --src ${EVAL_DIR}/HallusionBench.json \
    --dst ${EVAL_DIR}/${SPLIT}.jsonl

# Run model on chunks
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_hallusionbench \
        --model-path nahidalam/${CKPT} \
        --model-base CohereForAI/aya-23-8B \
        --question-file ${EVAL_DIR}/${SPLIT}.jsonl \
        --image-folder ${EVAL_DIR}/images \
        --answers-file ${EVAL_DIR}/answers/${CKPT}/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode aya &
done

wait

# Merge results
output_file=${EVAL_DIR}/answers/${CKPT}/merge.jsonl
> "$output_file"
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${EVAL_DIR}/answers/${CKPT}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

mkdir -p ${EVAL_DIR}/results/${CKPT}
# Convert results back to HallusionBench format
python scripts/convert_hallusionbench_result_for_eval.py \
    --data ${EVAL_DIR}/HallusionBench.json \
    --src ${EVAL_DIR}/answers/${CKPT}/merge.jsonl \
    --dst ${EVAL_DIR}/results/${CKPT}/HallusionBench_result.json

# Run evaluation
cd ${EVAL_DIR}
python evaluation.py --model ${CKPT}
