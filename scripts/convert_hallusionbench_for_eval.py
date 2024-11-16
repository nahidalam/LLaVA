import json
import os
from tqdm import tqdm

def convert_for_eval(src_file, dst_file):
    """Convert HallusionBench format to LLaVA format."""
    with open(src_file, 'r') as f:
        data = json.load(f)
    
    converted = []
    for item in tqdm(data):
        # Get image path if visual input is required
        image_file = None
        if int(item['visual_input']) != 0:
            image_file = os.path.join(
                item['category'],
                item['subcategory'],
                f"{item['set_id']}_{item['figure_id']}.png"
            )
        
        # Convert to LLaVA format
        converted_item = {
            "question_id": f"{item['category']}_{item['subcategory']}_{item['set_id']}_{item['figure_id']}_{item['question_id']}",
            "image": image_file,
            "text": item['question'],
            "category": item['category'],
            "subcategory": item['subcategory'],
            "set_id": item['set_id'],
            "figure_id": item['figure_id'],
            "visual_input": item['visual_input'],
            "gt_answer": item['gt_answer'],
            "gt_answer_details": item['gt_answer_details']
        }
        converted.append(converted_item)
    
    # Write to JSONL format
    with open(dst_file, 'w') as f:
        for item in converted:
            f.write(json.dumps(item) + '\n')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default="HallusionBench.json")
    parser.add_argument("--dst", type=str, default="llava-hallusionbench.jsonl")
    args = parser.parse_args()
    
    convert_for_eval(args.src, args.dst)
