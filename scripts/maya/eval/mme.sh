#!/bin/bash

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"

# LOAD_MME_PY="$PROJECT_ROOT/llava/eval/maya/load_mme.py"
MODEL_VQA_LOADER_PY="$PROJECT_ROOT/llava/eval/model_vqa_loader.py"
CONVERT_ANSWER_TO_MME_PY="$PROJECT_ROOT/llava/eval/maya/convert_answer_to_mme.py"
MME_CALCULATE_PY="$PROJECT_ROOT/llava/eval/maya/calculate_mme.py"


cd "$PROJECT_ROOT"

# Run this line once if you want to load the data from Hugging Face (caches to disk). load_mme.sh is the equivalent
# python "$LOAD_MME_PY


python -m llava.eval.model_vqa_loader \
    --model-path "nahidalam/maya_full_ft" \
    --model-base "CohereForAI/aya-23-8B" \
    --question-file "./playground/data/eval/MME/llava_mme.jsonl" \
    --image-folder "./playground/data/eval/MME/MME_Benchmark_release_version" \
    --answers-file "./playground/data/eval/MME/answers/maya_full_ft.jsonl" \
    --temperature "0" \
    --conv-mode "aya"


cd "$PROJECT_ROOT/playground/data/eval/MME"

python "$CONVERT_ANSWER_TO_MME_PY" \
    --experiment "maya_full_ft"


cd "$PROJECT_ROOT/llava/eval/maya"

python "$MME_CALCULATE_PY" \
    --results_dir "../../../playground/data/eval/MME/eval_tool/answers/maya_full_ft"

