# Evaluation

In LLaVA-1.5, we evaluate models on a diverse set of 12 benchmarks. To ensure the reproducibility, we evaluate the models with greedy decoding. We do not evaluate using beam search to make the inference process consistent with the chat demo of real-time outputs.

Currently, we mostly utilize the official toolkit or server for the evaluation.

## Setup
The evaluation scripts in `scripts/maya/eval` will auto-download the fine-tuned Maya from Hugging Face. Make sure you have pip installed `huggingface_hub` and authenticate with an access token with the necessary access. To authenticate, run `huggingface-cli login` and paste in the token.


## Evaluate on Custom Datasets

You can evaluate LLaVA on your custom datasets by converting your dataset to LLaVA's jsonl format, and evaluate using [`model_vqa.py`](https://github.com/haotian-liu/LLaVA/blob/main/llava/eval/model_vqa.py).

Below we provide a general guideline for evaluating datasets with some common formats.

1. Short-answer (e.g. VQAv2, MME).

```
<question>
Answer the question using a single word or phrase.
```

2. Option-only for multiple-choice (e.g. MMBench, SEED-Bench).

```
<question>
A. <option_1>
B. <option_2>
C. <option_3>
D. <option_4>
Answer with the option's letter from the given choices directly.
```

3. Natural QA (e.g. LLaVA-Bench, MM-Vet).

No postprocessing is needed.

## Scripts

Before preparing task-specific data, **you MUST first download [eval.zip](https://drive.google.com/file/d/1atZSBBrAX54yYpxtVVW33zFvcnaHeFPy/view?usp=sharing)**. It contains custom annotations, scripts, and the prediction files with LLaVA v1.5. Extract to `./playground/data/eval`. This also provides a general structure for all datasets.

### VQAv2

1. Download [`test2015`](http://images.cocodataset.org/zips/test2015.zip) and put it under `./playground/data/eval/vqav2`.
2. Multi-GPU inference.
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/eval/vqav2.sh
```
3. Submit the results to the [evaluation server](https://eval.ai/web/challenges/challenge-page/830/my-submission): `./playground/data/eval/vqav2/answers_upload`.

### GQA

1. Download the [data](https://cs.stanford.edu/people/dorarad/gqa/download.html) and [evaluation scripts](https://cs.stanford.edu/people/dorarad/gqa/evaluate.html) following the official instructions and put under `./playground/data/eval/gqa/data`. You may need to modify `eval.py` as [this](https://gist.github.com/haotian-liu/db6eddc2a984b4cbcc8a7f26fd523187) due to the missing assets in the GQA v1.2 release.
2. Multi-GPU inference.
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/eval/gqa.sh
```

### VizWiz

1. Download [`test.json`](https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_data/Annotations.zip) and extract [`test.zip`](https://vizwiz.cs.colorado.edu/VizWiz_final/images/test.zip) to `test`. Put them under `./playground/data/eval/vizwiz`. You may ingore any vizwiz files (jsonl) from the LLaVA zip file.
2. Single-GPU inference.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/maya/eval/vizwiz.sh
```
3. Submit the results to the [evaluation server](https://eval.ai/web/challenges/challenge-page/2185/my-submission): `./playground/data/eval/vizwiz/answers_upload`.

### ScienceQA

1. Under `./playground/data/eval/scienceqa`, download `pid_splits.json`, `problems.json` from the `data/scienceqa` folder of the ScienceQA [repo](https://github.com/lupantech/ScienceQA).
2. In `./playground/data/eval/scienceqa`:
    - Create a directory `images`
    ```
    mkdir images
    ```
    - Download the data splits inside `images`:
    ```
    cd images
    
    wget https://scienceqa.s3.us-west-1.amazonaws.com/images/train.zip --no-check-certificate
    wget https://scienceqa.s3.us-west-1.amazonaws.com/images/val.zip --no-check-certificate
    wget https://scienceqa.s3.us-west-1.amazonaws.com/images/test.zip --no-check-certificate

    unzip -q train.zip
    unzip -q val.zip
    unzip -q test.zip

    rm train.zip
    rm val.zip
    rm test.zip
    ```
3. Single-GPU inference and evaluate.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/sqa.sh
```

### TextVQA

1. Download [`TextVQA_0.5.1_val.json`](https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json) and [images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip) and extract to `./playground/data/eval/textvqa`.
2. Single-GPU inference and evaluate.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/textvqa.sh
```

### POPE

<!-- 1. Download `coco` from [POPE](https://github.com/AoiDragon/POPE/tree/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco) and put under `./playground/data/eval/pope`. -->
1. Download `coco` from the [POPE repo](https://github.com/AoiDragon/POPE/tree/e3e39262c85a6a83f26cf5094022a782cb0df58d/output). Use the link to the specific version, as the latest version [has changed](https://github.com/haotian-liu/LLaVA/issues/626#issuecomment-1772025961).
2. Download `val2014` from the [COCO dataset](https://cocodataset.org/#download). [Direct download link](http://images.cocodataset.org/zips/val2014.zip). Unzip to `./playground/data/eval/pope`. Images should be in a folder at `./playground/data/eval/pope/val2014`
3. After unzipping `eval.zip`, rename `/playground/data/eval/pope/llava_pope_test.jsonl` to `maya_pope_test.jsonl`
4. Single-GPU inference and evaluate.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/maya/eval/pope.sh
```



### MME

<!-- 1. Download the data following the official instructions [here](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation).
    1. You will have to request the dataset authors to access.
    2. Taken from the [repo](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation#:~:text=leaderboards%20in%20time.%20%E2%9C%A8-,Download,-MME%20%F0%9F%8C%9F%F0%9F%8C%9F) as of Oct 2024: 
        > The benchmark dataset is collected by Xiamen University for academic research only. You can email yongdongluo@stu.xmu.edu.cn to obtain the dataset, according to the following requirement.Requirement: A real-name system is encouraged for better academic communication. Your email suffix needs to match your affiliation, such as xx@stu.xmu.edu.cn and Xiamen University. Otherwise, you need to explain why. Please include the information bellow when sending your application email.
        ```
        Name: (tell us who you are.)
        Affiliation: (the name/url of your university or company)
        Job Title: (e.g., professor, PhD, and researcher)
        Email: (your email address)
        How to use: (only for non-commercial use)
        ```
2. Download images to `MME_Benchmark_release_version`. -->
1. Run `scripts/maya/eval/load_mme.sh` to load data from Hugging Face.
2. put the official `eval_tool` [folder](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation/tools) under `./playground/data/eval/MME`.
3. Single-GPU inference and evaluate.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/maya/eval/mme.sh
```

### MMBench

1. Download [`mmbench_dev_20230712.tsv`](https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_20230712.tsv) and put under `./playground/data/eval/mmbench`.
2. Single-GPU inference.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/mmbench.sh
```
3. Submit the results to the [evaluation server](https://opencompass.org.cn/leaderboard-multimodal): `./playground/data/eval/mmbench/answers_upload/mmbench_dev_20230712`.

### MMBench-CN

1. Download [`mmbench_dev_cn_20231003.tsv`](https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_cn_20231003.tsv) and put under `./playground/data/eval/mmbench`.
2. Single-GPU inference.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/mmbench_cn.sh
```
3. Submit the results to the evaluation server: `./playground/data/eval/mmbench/answers_upload/mmbench_dev_cn_20231003`.


### SEED-Bench

1. Following the official [instructions](https://github.com/AILab-CVC/SEED-Bench/blob/main/DATASET.md) to download the images and the videos. Put images under `./playground/data/eval/seed_bench/SEED-Bench-image`.
2. Extract the video frame in the middle from the downloaded videos, and put them under `./playground/data/eval/seed_bench/SEED-Bench-video-image`. We provide our script `extract_video_frames.py` modified from the official one.
3. Multiple-GPU inference and evaluate.
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/eval/seed.sh
```
4. Optionally, submit the results to the leaderboard: `./playground/data/eval/seed_bench/answers_upload` using the official jupyter notebook.

### LLaVA-Bench-in-the-Wild

1. Extract contents of [`llava-bench-in-the-wild`](https://huggingface.co/datasets/liuhaotian/llava-bench-in-the-wild) to `./playground/data/eval/llava-bench-in-the-wild`.
2. Single-GPU inference and evaluate.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/llavabench.sh
```

### MM-Vet

1. Extract [`mm-vet.zip`](https://github.com/yuweihao/MM-Vet/releases/download/v1/mm-vet.zip) to `./playground/data/eval/mmvet`.
2. Single-GPU inference.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/mmvet.sh
```
3. Evaluate the predictions in `./playground/data/eval/mmvet/results` using the official jupyter notebook.

## More Benchmarks

Below are awesome benchmarks for multimodal understanding from the research community, that are not initially included in the LLaVA-1.5 release.

### Q-Bench

1. Download [`llvisionqa_dev.json`](https://huggingface.co/datasets/nanyangtu/LLVisionQA-QBench/resolve/main/llvisionqa_dev.json) (for `dev`-subset) and [`llvisionqa_test.json`](https://huggingface.co/datasets/nanyangtu/LLVisionQA-QBench/resolve/main/llvisionqa_test.json) (for `test`-subset). Put them under `./playground/data/eval/qbench`. 
2. Download and extract [images](https://huggingface.co/datasets/nanyangtu/LLVisionQA-QBench/resolve/main/images_llvisionqa.tar) and put all the images directly under `./playground/data/eval/qbench/images_llviqionqa`.
3. Single-GPU inference (change `dev` to `test` for evaluation on test set).
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/qbench.sh dev
```
4. Submit the results by instruction [here](https://github.com/VQAssessment/Q-Bench#option-1-submit-results): `./playground/data/eval/qbench/llvisionqa_dev_answers.jsonl`.

### Chinese-Q-Bench

1. Download [`质衡-问答-验证集.json`](https://huggingface.co/datasets/nanyangtu/LLVisionQA-QBench/resolve/main/%E8%B4%A8%E8%A1%A1-%E9%97%AE%E7%AD%94-%E9%AA%8C%E8%AF%81%E9%9B%86.json) (for `dev`-subset) and [`质衡-问答-测试集.json`](https://huggingface.co/datasets/nanyangtu/LLVisionQA-QBench/resolve/main/%E8%B4%A8%E8%A1%A1-%E9%97%AE%E7%AD%94-%E6%B5%8B%E8%AF%95%E9%9B%86.json) (for `test`-subset). Put them under `./playground/data/eval/qbench`. 
2. Download and extract [images](https://huggingface.co/datasets/nanyangtu/LLVisionQA-QBench/resolve/main/images_llvisionqa.tar) and put all the images directly under `./playground/data/eval/qbench/images_llviqionqa`.
3. Single-GPU inference (change `dev` to `test` for evaluation on test set).
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/qbench_zh.sh dev
```
4. Submit the results by instruction [here](https://github.com/VQAssessment/Q-Bench#option-1-submit-results): `./playground/data/eval/qbench/llvisionqa_zh_dev_answers.jsonl`.
