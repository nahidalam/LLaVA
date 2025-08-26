import os
from transformers import AutoModel, AutoImageProcessor

# --- 1. Define your model paths ---
# The local directory where your gemma3_siglip_encoder is saved
LOCAL_MODEL_PATH = "gemma3_siglip_encoder"

# The name you want for your new repository on the Hugging Face Hub
# Format: "your-hf-username/your-model-name"
HUB_REPO_ID = "akshataa/gemma3-4b_it_siglip_encoder"


# --- 2. Load the model and its processor from the local directory ---
print(f"Loading model and processor from: {LOCAL_MODEL_PATH}")

# The processor handles image transformations (resizing, normalizing, etc.)
processor = AutoImageProcessor.from_pretrained(LOCAL_MODEL_PATH)

# The model itself
model = AutoModel.from_pretrained(LOCAL_MODEL_PATH)


# --- 3. Push the model and processor to the Hub ---
print(f"Pushing model and processor to: {HUB_REPO_ID}")

# It's highly recommended to upload as 'private' first to ensure everything is correct.
# You can make it public later from the website if you wish.
model.push_to_hub(
    HUB_REPO_ID,
    commit_message="Initial upload of gemma3_siglip_encoder model",
    private=True
)

processor.push_to_hub(
    HUB_REPO_ID,
    commit_message="Upload image processor",
    private=True
)

print("\nUpload complete!")
print(f"You can find your private model repository at: https://huggingface.co/{HUB_REPO_ID}")