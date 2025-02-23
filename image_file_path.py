import os

def list_and_remove_non_jpg_images(root_dir):
    non_jpg_files = []
    
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if not file.lower().endswith(".jpg"):
                file_path = os.path.join(dirpath, file)
                non_jpg_files.append(file_path)
                os.remove(file_path)
    
    return non_jpg_files

if __name__ == "__main__":
    dataset_dir = "/dev/data/instruction_tune_dataset"
    non_jpg_images = list_and_remove_non_jpg_images(dataset_dir)
    
    print("Non-JPG images removed:")
    for file in non_jpg_images:
        print(file)

