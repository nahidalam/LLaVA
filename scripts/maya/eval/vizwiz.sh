#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-path nahidalam/maya_full_ft \
    --model-base CohereForAI/aya-23-8B \
    --question-file ./playground/data/eval/vizwiz/test.json \
    --image-folder ./playground/data/eval/vizwiz/test \
    --answers-file ./playground/data/eval/vizwiz/answers/llava-v1.5-13b.jsonl \
    --temperature 0 \
    --conv-mode aya

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./playground/data/eval/vizwiz/test.json \
    --result-file ./playground/data/eval/vizwiz/answers/llava-v1.5-13b.jsonl \
    --result-upload-file ./playground/data/eval/vizwiz/answers_upload/maya_full_ft.json
