#!/bin/bash

export MULTILINGUAL_LLAVA_BENCH_PATH="playground/data/eval/multilingual-llava-bench-in-the-wild"
export OUTPUT_DIR="evaluation"
export IMAGES="$MULTILINGUAL_LLAVA_BENCH_PATH/images"

#export MODEL="/path/to/palo-v1.5-7b-665en_150K_of_arr_chi_hin_spa_ben_fr_jap_rus_ur"
#export MODEL_NAME="palo-v1.5-7b-665en_150K_of_arr_chi_hin_spa_ben_fr_jap_rus_ur"
#export OPENAI_API_KEY="write your open-ai key"

MODEL_BASE=$1
MODEL_PATH=$2
PROJECTOR_PATH=$3
MODEL_NAME=$4
export OPENAI_API_KEY=$5

export PYTHONPATH="./:$PYTHONPATH"

# 1.English
bash scripts/v1_5/eval/llavabench_palo.sh "$IMAGES" "$MODEL_BASE" "$MODEL_PATH" "$PROJECTOR_PATH" "$MULTILINGUAL_LLAVA_BENCH_PATH"/english/questions.jsonl "$MULTILINGUAL_LLAVA_BENCH_PATH"/english/answers_gpt4.jsonl "$MULTILINGUAL_LLAVA_BENCH_PATH"/english/context.jsonl "$OUTPUT_DIR" "$MODEL_NAME"_English.jsonl

# 2.Chinese
bash scripts/v1_5/eval/llavabench_palo.sh "$IMAGES" "$MODEL_BASE" "$MODEL_PATH" "$PROJECTOR_PATH" "$MULTILINGUAL_LLAVA_BENCH_PATH"/chinese/question.jsonl "$MULTILINGUAL_LLAVA_BENCH_PATH"/chinese/answers.jsonl "$MULTILINGUAL_LLAVA_BENCH_PATH"/chinese/context.jsonl "$OUTPUT_DIR" "$MODEL_NAME"_Chinese.jsonl

# 3.Spanish
bash scripts/v1_5/eval/llavabench_palo.sh "$IMAGES" "$MODEL_BASE" "$MODEL_PATH" "$PROJECTOR_PATH" "$MULTILINGUAL_LLAVA_BENCH_PATH"/spanish/question.jsonl "$MULTILINGUAL_LLAVA_BENCH_PATH"/spanish/answers.jsonl "$MULTILINGUAL_LLAVA_BENCH_PATH"/spanish/context.jsonl "$OUTPUT_DIR" "$MODEL_NAME"_Spanish.jsonl

# 4.French
bash scripts/v1_5/eval/llavabench_palo.sh "$IMAGES" "$MODEL_BASE" "$MODEL_PATH" "$PROJECTOR_PATH" "$MULTILINGUAL_LLAVA_BENCH_PATH"/french/question.jsonl "$MULTILINGUAL_LLAVA_BENCH_PATH"/french/answers.jsonl "$MULTILINGUAL_LLAVA_BENCH_PATH"/french/context.jsonl "$OUTPUT_DIR" "$MODEL_NAME"_French.jsonl

# 6.Russian
bash scripts/v1_5/eval/llavabench_palo.sh "$IMAGES" "$MODEL_BASE" "$MODEL_PATH" "$PROJECTOR_PATH" "$MULTILINGUAL_LLAVA_BENCH_PATH"/russian/question.jsonl "$MULTILINGUAL_LLAVA_BENCH_PATH"/russian/answers.jsonl "$MULTILINGUAL_LLAVA_BENCH_PATH"/russian/context.jsonl "$OUTPUT_DIR" "$MODEL_NAME"_Russian.jsonl

# 7.Arabic
bash scripts/v1_5/eval/llavabench_palo.sh "$IMAGES" "$MODEL_BASE" "$MODEL_PATH" "$PROJECTOR_PATH" "$MULTILINGUAL_LLAVA_BENCH_PATH"/arabic/question.jsonl "$MULTILINGUAL_LLAVA_BENCH_PATH"/arabic/answers.jsonl "$MULTILINGUAL_LLAVA_BENCH_PATH"/arabic/context.jsonl "$OUTPUT_DIR" "$MODEL_NAME"_Arabic.jsonl

# 8.Bengali
# bash scripts/v1_5/eval/llavabench_palo.sh "$IMAGES" "$MODEL_BASE" "$MODEL_PATH" "$PROJECTOR_PATH" "$MULTILINGUAL_LLAVA_BENCH_PATH"/bengali/question.jsonl "$MULTILINGUAL_LLAVA_BENCH_PATH"/bengali/answers.jsonl "$MULTILINGUAL_LLAVA_BENCH_PATH"/bengali/context.jsonl "$OUTPUT_DIR" "$MODEL_NAME"_Bengali.jsonl

# 9.Hindi
bash scripts/v1_5/eval/llavabench_palo.sh "$IMAGES" "$MODEL_BASE" "$MODEL_PATH" "$PROJECTOR_PATH" "$MULTILINGUAL_LLAVA_BENCH_PATH"/hindi/question.jsonl "$MULTILINGUAL_LLAVA_BENCH_PATH"/hindi/answers.jsonl "$MULTILINGUAL_LLAVA_BENCH_PATH"/hindi/context.jsonl "$OUTPUT_DIR" "$MODEL_NAME"_Hindi.jsonl

# 10.Urdu
# bash scripts/v1_5/eval/llavabench_palo.sh "$IMAGES" "$MODEL_BASE" "$MODEL_PATH" "$PROJECTOR_PATH" "$MULTILINGUAL_LLAVA_BENCH_PATH"/urdu/question.jsonl "$MULTILINGUAL_LLAVA_BENCH_PATH"/urdu/answers.jsonl "$MULTILINGUAL_LLAVA_BENCH_PATH"/urdu/context.jsonl "$OUTPUT_DIR" "$MODEL_NAME"_Urdu.jsonl

# 11.Japanese
bash scripts/v1_5/eval/llavabench_palo.sh "$IMAGES" "$MODEL_BASE" "$MODEL_PATH" "$PROJECTOR_PATH" "$MULTILINGUAL_LLAVA_BENCH_PATH"/japanese/question.jsonl "$MULTILINGUAL_LLAVA_BENCH_PATH"/japanese/answers.jsonl "$MULTILINGUAL_LLAVA_BENCH_PATH"/japanese/context.jsonl "$OUTPUT_DIR" "$MODEL_NAME"_Japanese.jsonl
