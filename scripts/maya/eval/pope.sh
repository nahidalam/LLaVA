#!/bin/bash
python -m llava.eval.model_vqa_loader \
    --model-path nahidalam/maya_full_ft \
    --model-base CohereForAI/aya-23-8B \
    --question-file ./playground/data/eval/pope/maya_pope_test.jsonl \
    --image-folder ./playground/data/eval/pope/val2014 \
    --answers-file ./playground/data/eval/pope/answers/maya_full_ft.jsonl \
    --temperature 0 \
    --conv-mode aya

python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/maya_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/maya_full_ft.jsonl
