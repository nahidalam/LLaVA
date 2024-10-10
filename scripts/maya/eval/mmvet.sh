#!/bin/bash

python -m llava.eval.model_vqa \
    --model-path nahidalam/maya_full_ft \
    --model-base CohereForAI/aya-23-8B \
    --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ./playground/data/eval/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/maya_full_ft.jsonl \
    --temperature 0 \
    --conv-mode aya

mkdir -p ./playground/data/eval/mm-vet/results_maya

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/maya_full_ft.jsonl \
    --dst ./playground/data/eval/mm-vet/results_maya/maya_full_ft.json

