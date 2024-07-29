#!/bin/bash

IMAGES=$1
MODEL_BASE=$2
MODEL_PATH=$3
PROJECTOR_PATH=$4
QUESTIONS=$5
ANSWERS=$6
CONTEXT=$7
OUTPUT_DIR=$8
OUTPUT_FILE=$9


python llava/eval/model_vqa_maya.py \
    --model-base "$MODEL_BASE" \
    --model-path "$MODEL_PATH" \
    --projector-path "$PROJECTOR_PATH" \
    --question-file "$QUESTIONS" \
    --image-folder "$IMAGES" \
    --answers-file "$OUTPUT_DIR/$OUTPUT_FILE" \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p "$OUTPUT_DIR/reviews"

python llava/eval/eval_gpt_review_bench.py \
    --question "$QUESTIONS" \
    --context "$CONTEXT" \
    --rule llava/eval/table/rule.json \
    --answer-list \
        "$ANSWERS" \
        "$OUTPUT_DIR/$OUTPUT_FILE" \
    --output \
        "$OUTPUT_DIR/reviews/$OUTPUT_FILE"

python llava/eval/summarize_gpt_review.py -f "$OUTPUT_DIR/reviews/$OUTPUT_FILE"
