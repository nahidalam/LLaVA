import json
import requests

# Check the original model configuration
url = "https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b/raw/main/config.json"
response = requests.get(url)
config = json.loads(response.text)

print("Original vision tower configuration:")
print(f"mm_vision_tower: {config.get('mm_vision_tower', 'Not found')}")
print(f"vision_tower: {config.get('vision_tower', 'Not found')}")
print("\nFull config keys:")
for key in config.keys():
    if 'vision' in key.lower() or 'mm' in key.lower():
        print(f"{key}: {config[key]}")