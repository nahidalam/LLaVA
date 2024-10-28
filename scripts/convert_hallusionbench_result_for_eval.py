import json
import argparse
from tqdm import tqdm
def convert_result(data_file, src_file, dst_file):
    """Convert LLaVA output format back to HallusionBench format."""
    # Read original HallusionBench data
    with open(data_file, 'r') as f:
        orig_data = json.load(f)
    
    # Create lookup dictionary
    results = {}
    with open(src_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            results[item['question_id']] = item['text']
    
    # Add model predictions to original data
    for item in tqdm(orig_data):
        qid = f"{item['category']}_{item['subcategory']}_{item['set_id']}_{item['figure_id']}_{item['question_id']}"
        response = results.get(qid, "")
        
        # Convert response to required format ("0", "1", or "2")
        response = response.lower().strip()
        if "yes" in response:
            item['model_prediction'] = "1"
        elif "no" in response:
            item['model_prediction'] = "0"
        else:
            item['model_prediction'] = "2"
    
    # Save results
    with open(dst_file, 'w') as f:
        json.dump(orig_data, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Original HallusionBench data")
    parser.add_argument("--src", type=str, help="Source file with model outputs")
    parser.add_argument("--dst", type=str, help="Destination file for evaluation")
    args = parser.parse_args()
    
    convert_result(args.data, args.src, args.dst)
