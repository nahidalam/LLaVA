import os
import sys
import torch
import requests
from io import BytesIO
from PIL import Image


from transformers import AutoTokenizer, AutoConfig, TextStreamer
from transformers.models.cohere.tokenization_cohere_fast import CohereTokenizerFast
from llava.model.language_model.llava_cohere import LlavaCohereForCausalLM, LlavaCohereConfig
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path




def get_projector_pretrained_cohere_model(model_base : str, model_path : str, projector_path : str) :
    """ Function that instantiates a pretrained Cohere/Aya model with only the adapter(projector) layer trained
    This is a replication of the load_pretrained_model function from llava.model.builder thats specific to Cohere/Maya

    Args:
        model_base : Path of the base LLM model in HF. Eg: 'CohereForAI/aya-23-8B', 'meta-llama/Meta-Llama-3-8B-Instruct'. 
                     This is used to instantiate the model and the tokenizer
        model_path : Path of the pretrained model in HF. Eg : 'nahidalam/Maya'
                     This is used to load the config file from the pretrained model. So this path/directory should have the config.json file
        projector_path : Path to the local directory which holds the mm_projector.bin file

    Returns:
       model: LlavaCohereForCausalLM object
       tokenizer: CohereTokenizerFast object
       image_processor:
       content_len:
    """

    device_map = 'auto'
    kwargs = {"device_map": device_map}
    kwargs['torch_dtype'] = torch.float16
    kwargs['attn_implementation'] = 'flash_attention_2'

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


def get_single_prompt_prediction(model, tokenizer, image_processor, image_file, user_question, temperature = 0.0, max_new_tokens = 100):
    """Generates the prediction for a single image-user question pair.

    Args:
        model (LlavaCohereForCausalLM): Pretrained cohere model
        tokenizer (CohereTokenizerFast): Tokenizer for pretrained model
        image_processor (): Image processor for pretrained model
        image_file (str): Local path to image file OR online link to image file
        user_question (str): Question to be shared with LLM
        temperature (float, optional): Temperature param for LLMs.  Defaults to 0.0.
        max_new_tokens (int, optional): Max new number of tokens generated. Defaults to 100.

    Returns:
        output (str): Model's response to user question  
    """
  

    conv_mode = "llava_v1"  # Need to verify this
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles

    ## Loading input image
    def load_image(image_file):
        if image_file.startswith('http://') or image_file.startswith('https://'):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(image_file).convert('RGB')
        return image

    image = load_image(image_file)
    image_size = image.size

    image_tensor = process_images([image], image_processor, model.config)
    if type(image_tensor) is list:
        image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    inp = user_question

    if image is not None:
        # first message
        if model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
        # image = None

    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image_size],
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            use_cache=True)

    outputs = tokenizer.decode(output_ids[0]).strip()

    return outputs

    

