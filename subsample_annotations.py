import json
import random

# Define the path to the full annotation file
full_annotation_path = 'blip_laion_cc_sbu_558k.json'
# Define the path for our new, smaller annotation file
small_annotation_path = 'blip_laion_cc_sbu_55.8k.json' # 10% subset

# Define the fraction of the dataset you want to use (e.g., 0.1 for 10%)
fraction = 0.1

print("Loading full annotation file...")
with open(full_annotation_path, 'r') as f:
    full_data = json.load(f)

# Shuffle the data to get a representative sample
random.shuffle(full_data)

# Calculate the number of samples for the subset
num_small_samples = int(len(full_data) * fraction)

# Create the subset
small_data = full_data[:num_small_samples]

print(f"Created a subset with {len(small_data)} samples.")

# Save the new smaller annotation file
with open(small_annotation_path, 'w') as f:
    json.dump(small_data, f)

print(f"Saved smaller annotation file to: {small_annotation_path}")