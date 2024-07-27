# LLaVA Bench In-the-Wild Multilingual Benchmark

Instructions for running the LLaVA Bench In-the-Wild multilingual benchmark.

## Setup

1. Install Git LFS:
   ```
   brew install git-lfs
   git lfs install
   ```

2. Download the PALO evaluation dataset:
   ```
   cd /path/to/LLaVA/playground/data/eval
   git clone https://huggingface.co/datasets/MBZUAI/multilingual-llava-bench-in-the-wild
   ```

## Running the Benchmark

To run the benchmark, use the following command:

bash scripts/v1_5/eval/eval_all_languages.sh "model-path" "model-name" "openai-api-key"

Example:

bash scripts/v1_5/eval/eval_all_languages.sh "liuhaotian/llava-v1.5-13b" "llava-v1.5-13b" "your-openai-api-key"

## Note

Ensure you have LLaVA/Maya and necessary dependencies installed and the required datasets downloaded before running the benchmark.