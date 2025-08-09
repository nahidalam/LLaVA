#!/usr/bin/env python
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
import llava.model.language_model.llava_llama

MODEL_PATH = "liuhaotian/llava-v1.6-vicuna-7b"

def main() -> None:
    model_name = get_model_name_from_path(MODEL_PATH)

    _, model, _, _ = load_pretrained_model(
        MODEL_PATH,
        None,
        model_name,
        False, False,
        device="cpu",
    )

    vision_tower_cls = model.get_vision_tower().__class__.__name__
    print("Vision tower class in use: ", vision_tower_cls)


if __name__ == "__main__":
    main()