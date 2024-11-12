#!/bin/bash
#
# Evaluates Maya on the PALO Multilingual-LLaVA Bench-In-The-Wild dataset.
#
# Usage:
#   bash scripts/maya/eval/palo_multilingual_llava_bench.sh \
#     <MODEL_BASE> <MODEL_PATH> <MODE> <OPENAI_API_KEY> [PROJECTOR_PATH]
#
# Note:
#   - MODE: Choose either 'pretrained' or 'finetuned' (without quotes). Example: finetuned
#   - PROJECTOR_PATH: required if MODE=pretrained
#   - See Readme for more details 



export MULTILINGUAL_LLAVA_BENCH_PATH="playground/data/eval/multilingual-llava-bench-in-the-wild"
export OUTPUT_DIR="evaluation"
export IMAGES="$MULTILINGUAL_LLAVA_BENCH_PATH/images"
export PYTHONPATH="./:$PYTHONPATH"


MODEL_BASE=$1
MODEL_PATH=$2
MODE=$3
export OPENAI_API_KEY=$4
PROJECTOR_PATH=$5


evaluate_language() {
    local LANG=$1
    local QUESTIONS=$2
    local ANSWERS=$3
    local CONTEXT=$4
    local OUTPUT_FILE="Maya_${LANG}.jsonl"

    echo "******** Evaluating Maya on $LANG ********"
    
    cmd="python llava/eval/model_vqa_maya.py \
        --model-base "$MODEL_BASE" \
        --model-path "$MODEL_PATH" \
        --mode $MODE \
        --conv-mode aya \
        --question-file "$QUESTIONS" \
        --image-folder "$IMAGES" \
        --answers-file "$OUTPUT_DIR/$OUTPUT_FILE" \
        --temperature 0"

    # Add projector path if provided
    if [ ! -z "$PROJECTOR_PATH" ]; then
        cmd+=" --projector-path \"$PROJECTOR_PATH\""
    fi

    # Execute the command
    eval $cmd

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

}

# Evaluate each language
# 1.English
evaluate_language "English" "$MULTILINGUAL_LLAVA_BENCH_PATH/english/questions.jsonl" "$MULTILINGUAL_LLAVA_BENCH_PATH/english/answers_gpt4.jsonl" "$MULTILINGUAL_LLAVA_BENCH_PATH/english/context.jsonl"

# 2.Chinese
evaluate_language "Chinese" "$MULTILINGUAL_LLAVA_BENCH_PATH/chinese/question.jsonl" "$MULTILINGUAL_LLAVA_BENCH_PATH/chinese/answers.jsonl" "$MULTILINGUAL_LLAVA_BENCH_PATH/chinese/context.jsonl"

# 3.Spanish
evaluate_language "Spanish" "$MULTILINGUAL_LLAVA_BENCH_PATH/spanish/question.jsonl" "$MULTILINGUAL_LLAVA_BENCH_PATH/spanish/answers.jsonl" "$MULTILINGUAL_LLAVA_BENCH_PATH/spanish/context_spanish.jsonl"

# 4.French
evaluate_language "French" "$MULTILINGUAL_LLAVA_BENCH_PATH/french/question.jsonl" "$MULTILINGUAL_LLAVA_BENCH_PATH/french/answers.jsonl" "$MULTILINGUAL_LLAVA_BENCH_PATH/french/context.jsonl"

# 5.Russian
evaluate_language "Russian" "$MULTILINGUAL_LLAVA_BENCH_PATH/russian/question.jsonl" "$MULTILINGUAL_LLAVA_BENCH_PATH/russian/answers.jsonl" "$MULTILINGUAL_LLAVA_BENCH_PATH/russian/context.jsonl"

# 6.Arabic
evaluate_language "Arabic" "$MULTILINGUAL_LLAVA_BENCH_PATH/arabic/question.jsonl" "$MULTILINGUAL_LLAVA_BENCH_PATH/arabic/answers.jsonl" "$MULTILINGUAL_LLAVA_BENCH_PATH/arabic/context.jsonl"

# 7.Bengali -- Not currently supported
evaluate_language "Bengali" "$MULTILINGUAL_LLAVA_BENCH_PATH/bengali/question.jsonl" "$MULTILINGUAL_LLAVA_BENCH_PATH/bengali/answers.jsonl" "$MULTILINGUAL_LLAVA_BENCH_PATH/bengali/context.jsonl"

# 8.Hindi
evaluate_language "Hindi" "$MULTILINGUAL_LLAVA_BENCH_PATH/hindi/question.jsonl" "$MULTILINGUAL_LLAVA_BENCH_PATH/hindi/answers.jsonl" "$MULTILINGUAL_LLAVA_BENCH_PATH/hindi/context.jsonl"

# 9.Urdu -- Not currently supported
evaluate_language "Urdu" "$MULTILINGUAL_LLAVA_BENCH_PATH/urdu/question.jsonl" "$MULTILINGUAL_LLAVA_BENCH_PATH/urdu/answers.jsonl" "$MULTILINGUAL_LLAVA_BENCH_PATH/urdu/context.jsonl"

# 10.Japanese
evaluate_language "Japanese" "$MULTILINGUAL_LLAVA_BENCH_PATH/japanese/question.jsonl" "$MULTILINGUAL_LLAVA_BENCH_PATH/japanese/answers.jsonl" "$MULTILINGUAL_LLAVA_BENCH_PATH/japanese/context.jsonl"
