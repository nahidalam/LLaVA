import os
from PIL import Image

# Define the directory path
directory = "/dev/data/instruction_tune_dataset/ocr_vqa/images"

def convert_to_jpg(file_path):
    """Convert the given image to JPG and save it."""
    try:
        # Open the image
        with Image.open(file_path) as img:
            # Convert image to RGB (if not already in this mode)
            img = img.convert("RGB")
            # Save the image as JPG
            new_file_path = os.path.splitext(file_path)[0] + ".jpg"
            img.save(new_file_path, "JPEG")
            print(f"Converted: {file_path} -> {new_file_path}")
            # Optionally delete the original file
            os.remove(file_path)
            print(f"Deleted original file: {file_path}")
    except Exception as e:
        print(f"Error converting {file_path}: {e}")

# Walk through the directory to find non-JPG images
for root, _, files in os.walk(directory):
    for file_name in files:
        file_path = os.path.join(root, file_name)
        # Skip if the file is already a JPG
        if not file_name.lower().endswith((".jpg", ".jpeg")):
            convert_to_jpg(file_path)

