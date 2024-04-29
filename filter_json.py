import json

# Load the original JSON file
with open('./playground/data/llava_v1_5_mix665k.json', 'r') as f:
    data = json.load(f)

# Filter out items where the "image" field starts with "coco/train2017/"
filtered_data = [item for item in data if item.get('image', '').startswith('coco/train2017/')]

# Write the filtered data to a new JSON file
with open('./playground/data/filtered.json', 'w') as f:
    json.dump(filtered_data, f, indent=2)

