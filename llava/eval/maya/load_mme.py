

"""
This script loads the MME dataset from Hugging Face, saves images and Y/N answers to the specified folder structure.
"""

import os
import json
import pandas as pd
from datasets import load_dataset
from PIL import Image

def main():
    # Define paths
    questions_file = "./playground/data/eval/MME/llava_mme.jsonl"
    answers_file = "./playground/data/eval/MME/answers/llava-v1.5-13b.jsonl"
    images_folder = "./playground/data/eval/MME/MME_Benchmark_release_version"
    
    # Load the MME dataset from disk
    mme_dataset = load_dataset("lmms-lab/MME")
    print("Dataset loaded successfully.\n")
    
    # Ensure base image save directory exists
    image_save_base_path = "./playground/data/eval/MME/MME_Benchmark_release_version"
    os.makedirs(image_save_base_path, exist_ok=True)

    # Categories that require an extra subdirectory for Y/N answers and images
    special_categories = ["artwork", "celebrity", "landmark", "scene", "posters"]
    
    # Save Y/N Answers to Folder
    def save_answer(example):
        question_id = example['question_id']  # Format: category/image.png
        question = example['question']
        answer = example['answer']
        
        category = question_id.split('/')[0]
        image_name = os.path.basename(question_id)
        image_basename = os.path.splitext(image_name)[0]
        
        # Special case handling – one entry on HuggingFace is different than the LLaVA repo
        if category == 'code_reasoning' and image_basename == '0020':
            question = "Is a python code shown in the picture? Please answer yes or no."
        
        # Determine the save directory
        if category in special_categories:
            full_save_dir = os.path.join(image_save_base_path, category, "questions_answers_YN")
        else:
            full_save_dir = os.path.join(image_save_base_path, category, "questions_answers_YN")
        
        # Create the directory if it doesn't exist
        os.makedirs(full_save_dir, exist_ok=True)
        
        # Define the full path for the answer file
        answer_save_path = os.path.join(full_save_dir, f"{image_basename}.txt")
        
        # Append the question and answer to the text file
        with open(answer_save_path, 'a') as answer_file:
            answer_file.write(f"{question}\t{answer}\n")
        
        return {'answer_save_path': answer_save_path}
    
    # Save Images to Folder
    def save_images(example):
        image = example['image']  # Assuming 'image' is a PIL Image object
        image_subdir = example['question_id']  # category/image.png
        category = image_subdir.split('/')[0]  # extract the category (first part of question_id)
        file_name = os.path.basename(image_subdir)  # extract the file name (including extension)

        # Get the image extension (e.g., ".png", ".jpg")
        _, extension = os.path.splitext(file_name)

        # Check if the category is one of the special ones that needs an extra 'images' folder
        if category in special_categories:
            # Save in the category/images/ structure (e.g., artwork/images/16006.jpg)
            full_save_dir = os.path.join(image_save_base_path, category, "images")
        else:
            # Save in the usual category/image.png structure (e.g., code_reasoning/0012.png)
            full_save_dir = os.path.join(image_save_base_path, category)

        # Create the subdirectory if it doesn't exist
        os.makedirs(full_save_dir, exist_ok=True)

        # Create the full image save path
        image_save_path = os.path.join(full_save_dir, file_name)

        # Save the image to the specified path in the correct format
        if extension.lower() == ".jpg" or extension.lower() == ".jpeg":
            image.save(image_save_path, format="JPEG")
        elif extension.lower() == ".png":
            image.save(image_save_path, format="PNG")
        else:
            # Handle other formats or default to PNG (though this shouldn't happen!)
            image.save(image_save_path, format="PNG")

        return {'image_save_path': image_save_path}
    
    # Apply the save_answer and save_images functions to the 'test' split of the dataset
    print("Saving Y/N answers to disk...")
    _ = mme_dataset['test'].map(save_answer)
    print("Answers sucessfully saved.")
    print("Saving images to disk. This might slow down around the 2000 image mark; be patient, it will speed up ")
    _ = mme_dataset['test'].map(save_images)
    print("Images sucessfully saved.")
    

if __name__ == "__main__":
    main()
