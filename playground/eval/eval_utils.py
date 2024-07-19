import os
import sys
import torch
from transformers import AutoTokenizer, AutoConfig
from transformers.models.cohere.tokenization_cohere_fast import CohereTokenizerFast
from llava.model.language_model.llava_cohere import LlavaCohereForCausalLM, LlavaCohereConfig
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN



def get_projector_pretrained_cohere_model(model_base, model_path, projector_path):

    ## Instantiating tokenizer and model base
    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
    cfg_pretrained = LlavaCohereConfig.from_pretrained(model_path)
    model = LlavaCohereForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)


    ## Loading Projector layer weights
    mm_projector_weights = torch.load(projector_path, map_location='cpu')
    mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
    model.load_state_dict(mm_projector_weights, strict=False)


    ## Loading image processor
    image_processor = None

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model(device_map=device_map)
    if device_map != 'auto':
        vision_tower.to(device=device_map, dtype=torch.float16)
    image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return model, tokenizer, image_processor, context_len

