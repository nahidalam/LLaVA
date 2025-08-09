import torch
from transformers import Aimv2VisionModel, AutoImageProcessor
from PIL import Image
import numpy as np

# Load the model and processor
model_name = "apple/aimv2-large-patch14-224"
model = Aimv2VisionModel.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

# Load and process a test image
image = Image.open("sample.jpg").convert("RGB")
print(f"Original image size: {image.size}")

# Process the image
inputs = processor(images=image, return_tensors="pt")
print(f"Processed tensor shape: {inputs['pixel_values'].shape}")

# Run through model
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)
    
print(f"Output last hidden state shape: {outputs.last_hidden_state.shape}")
print(f"Number of hidden states: {len(outputs.hidden_states)}")
print(f"Hidden state -2 shape: {outputs.hidden_states[-2].shape}")

# Check if there's a CLS token
if outputs.last_hidden_state.shape[1] == 257:  # 256 patches + 1 CLS
    print("Model includes CLS token at position 0")
else:
    print(f"Model has {outputs.last_hidden_state.shape[1]} tokens")