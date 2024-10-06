# LLaVA Bench In-the-Wild Multilingual (PALO) Benchmark

Instructions for running the LLaVA Bench In-the-Wild multilingual benchmark.

## Setup

1. Install Git LFS:
   Please refer to [official Git LFS Installation guide](https://github.com/git-lfs/git-lfs#installing)
   
   Examples:
   - For macOS (using Homebrew):
      ```
      brew install git-lfs
      git lfs install
      ```
   - For Ubuntu/Debian:
      ```
      sudo apt-get install git-lfs
      git lfs install
      ```

2. Download the PALO evaluation dataset:
   Create the following directory structure if it doesn't exist.
   ```
   cd /path/to/LLaVA/playground/data/eval
   git clone https://huggingface.co/datasets/MBZUAI/multilingual-llava-bench-in-the-wild
   ```

3. Maya Model Components:
   Maya is a multimodal model based on the Aya architecture with custom modifications. It is currently composed of:
   - Base model: "CohereForAI/aya-23-8B"
   - Model configuration: "nahidalam/Maya" (used for loading the config.json file)
   - Projector weights: found at https://huggingface.co/nahidalam/Maya/tree/main

   Downloading projector weights (one possible method): 
   Create the following directory structure if it doesn't exist.
   ```
   playground/data/eval/maya_projector
   wget https://huggingface.co/nahidalam/Maya/resolve/main/mm_projector.bin -O playground/data/eval/maya_projector/mm_projector.bin
   ```

## Running the Benchmark

4. Running the Evaluation:
   To run the evaluation, use the following command:
   ```
   bash scripts/v1_5/eval/eval_all_languages.sh \
       "model_base" \
       "model_path" \
       "projector_path" \
       "model_name" \
       "your-openai-api-key"
   ```

Example:
   ```
   bash scripts/v1_5/eval/eval_all_languages.sh \
       "CohereForAI/aya-23-8B" \
       "nahidalam/Maya" \
       "playground/data/eval/maya_projector/mm_projector.bin" \
       "Maya-8B" \
       "your-openai-api-key"
   ```

5. Eval Results: 
After running the evaluation, you can examine the following files:
```
Maya response file: LLaVA/evaluation/Maya-8B_English.jsonl
(Contains Maya's raw responses to the evaluation questions)

Judge-LLM response file: LLaVA/evaluation/reviews/Maya-8B_English.jsonl
(Contains the judge LLM's evaluation of Maya's responses)
```
