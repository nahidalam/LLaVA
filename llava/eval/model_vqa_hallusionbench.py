import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.eval.maya.eval_utils import load_maya_model

from PIL import Image
import math

# TODO: fix answer generation, as all results are 2

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    if 'maya' not in model_name:
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    else:
        model, tokenizer, image_processor, context_len = load_maya_model(args.model_base, model_path, mode=args.mode)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line.get("image")  # May be None for text-only questions
        qs = line["text"]
        
        # Always create a conversation, but handle image token differently
        conv = conv_templates[args.conv_mode].copy()
        
        if image_file is None:
            # Text-only question - don't add image token
            conv.append_message(conv.roles[0], qs)
            image_tensor = None
        else:
            # Image question - add image token
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            conv.append_message(conv.roles[0], qs)
            
            # Process image
            image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
            image_tensor = process_images([image], image_processor, model.config)[0].unsqueeze(0).half().cuda()

        # Add response placeholder and get prompt
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Tokenize input
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        # Generate response
        with torch.inference_mode():
            outputs = model.generate(
                inputs=input_ids,
                images=image_tensor if image_file is not None else None,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=30,
                use_cache=True,
            )

        output_text = tokenizer.batch_decode(outputs[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        
        # Convert output to HallusionBench format (0=No, 1=Yes, 2=Uncertain)
        output_text = output_text.lower().strip()
        if "yes" in output_text:
            answer = "1"
        elif "no" in output_text:
            answer = "0"
        else:
            answer = "2"

        # Save result
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({
            "question_id": idx,
            "prompt": qs,
            "text": answer,
            "answer_id": ans_id,
            "model_id": model_name,
            "metadata": {
                "raw_response": output_text,
                "category": line.get("category"),
                "subcategory": line.get("subcategory"),
                "set_id": line.get("set_id"),
                "figure_id": line.get("figure_id"),
            }
        }) + "\n")
        ans_file.flush()

    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="nahidalam/maya_full_ft")
    parser.add_argument("--model-base", type=str, default="CohereForAI/aya-23-8B")
    parser.add_argument("--mode", type=str, default="finetuned")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    eval_model(args)
