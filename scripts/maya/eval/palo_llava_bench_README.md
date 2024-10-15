# PALO Multilingual LLaVA Bench In-the-Wild Benchmark

Instructions for running the multilingual LLaVA Bench In-the-Wild benchmark.

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
   LLaVA/playground/data/eval
   git clone https://huggingface.co/datasets/MBZUAI/multilingual-llava-bench-in-the-wild
   ```

3. Maya Model:
   - model_base: Path to base LLM model (e.g., 'CohereForAI/aya-23-8B')
   - model_path: Path to trained model repo (e.g., 'nahidalam/maya_full_ft')
   - projector_path: Path to mm_projector.bin file (for pretrained Maya model)
   - mode: 'pretrained' or 'finetuned'
   - Projector weight/bin file found at (need access): https://huggingface.co/nahidalam/Maya-Pretrain-8lang/tree/main
   - For most accurate results, run benchmark on the finetuned model (mode: finetuned)

   Downloading projector weights (one possible method): 
   Create the following directory structure if it doesn't exist.
   ```
   LLaVA/playground/data/eval/maya_projector
   wget https://huggingface.co/nahidalam/Maya-Pretrain-8lang/resolve/main/mm_projector.bin -O playground/data/eval/maya_projector/mm_projector.bin
   ```

## Running the Benchmark

4. Running the Evaluation:
   To run the evaluation, use the following command:
   ```
   bash scripts/maya/eval/palo_multilingual_llava_bench.sh \
       "model_base" \
       "model_path" \
       mode \
       "openai-api-key" \
       "projector_path" 
   ```
   Note:
   - mode: Choose either 'pretrained' or 'finetuned' (<u>without quotes</u>). Example: finetuned
   - projector_path: required if mode=pretrained
   - openai-api-key: required

Examples:
   (1) Finetuned Maya Model:
   ```
   bash scripts/maya/eval/palo_multilingual_llava_bench.sh \ 
         "CohereForAI/aya-23-8B" \
         "nahidalam/maya_full_ft" \
         finetuned \
         "openai-api-key"
   ```

   (2) Pretrained Maya Model:
   ```
   bash scripts/maya/eval/palo_multilingual_llava_bench.sh \
        "CohereForAI/aya-23-8B" \
        "nahidalam/Maya-Pretrain-8lang" \
        pretrained \
        "openai-api-key" \
        "playground/data/eval/maya_projector/mm_projector.bin"
   ```

5. Evaluation Results: 
   After running the eval, you can examine the following files:
   ```
   Maya response file: LLaVA/evaluation/Maya_English.jsonl
   (Contains Maya's raw responses to the evaluation questions)

   Judge-LLM response file: LLaVA/evaluation/reviews/Maya_English.jsonl
   (Contains the judge LLM's evaluation of Maya's responses)
   ```