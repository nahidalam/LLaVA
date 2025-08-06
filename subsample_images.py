import json
import os
import zipfile
import sys

# --- START OF CORRECTION ---
# This function checks if the script is running in an interactive environment like Jupyter.
def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

# Import the correct version of tqdm based on the environment
if is_notebook():
    from tqdm.notebook import tqdm
    print("Running in a notebook environment, using tqdm.notebook.")
else:
    from tqdm import tqdm
    print("Running in a terminal environment, using standard tqdm.")
# --- END OF CORRECTION ---


def create_subset_zip(small_json_path, large_zip_path, output_zip_path):
    """
    Creates a new, smaller zip file containing only the images listed in a JSON annotation file.

    Args:
        small_json_path (str): Path to the small JSON file with image filenames.
        large_zip_path (str): Path to the original large zip file.
        output_zip_path (str): Path where the new, smaller zip file will be saved.
    """
    # --- Step 1: Get the set of required image filenames from the small JSON ---
    print(f"Step 1: Loading required image filenames from '{os.path.basename(small_json_path)}'...")
    try:
        with open(small_json_path, 'r') as f:
            small_data = json.load(f)
        required_basenames = {item['image'] for item in small_data if 'image' in item}
        print(f"Found {len(required_basenames)} unique image filenames required.")
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return

    # --- Step 2: Build a map of (basename -> full_path) from the large zip ---
    print(f"Step 2: Building a file map from '{os.path.basename(large_zip_path)}'. This may take a moment...")
    path_map = {}
    try:
        with zipfile.ZipFile(large_zip_path, 'r') as large_zip:
            # Use the imported tqdm (either notebook or standard version)
            for file_path in tqdm(large_zip.namelist(), desc="Scanning large zip"):
                if file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                    basename = os.path.basename(file_path)
                    path_map[basename] = file_path
        print("File map created successfully.")
    except Exception as e:
        print(f"Error reading the large zip file: {e}")
        return

    # --- Step 3: Create the new zip by copying only the required files ---
    print(f"Step 3: Creating new zip file at '{output_zip_path}'...")
    not_found_count = 0
    with zipfile.ZipFile(large_zip_path, 'r') as large_zip:
        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as small_zip:
            # Use the imported tqdm
            for basename in tqdm(required_basenames, desc="Writing to new zip"):
                if basename in path_map:
                    full_path = path_map[basename]
                    image_data = large_zip.read(full_path)
                    small_zip.writestr(full_path, image_data)
                else:
                    print(f"Warning: Image '{basename}' from JSON not found in the large zip.")
                    not_found_count += 1
    
    print("-" * 30)
    print("✅ Process Complete!")
    print(f"New zip file created: {output_zip_path}")
    if not_found_count > 0:
        print(f"⚠️ Warning: {not_found_count} required images were not found in the source zip.")

# --- Main execution block ---
if __name__ == "__main__":
    # --- Define your file paths here ---
    # This should be the path to the small JSON you created
    small_json_path = 'blip_laion_cc_sbu_55.8k.json' 

    # This is the path to the large zip file
    large_zip_path = 'images.zip'

    # This is the desired output path for your new, smaller zip file
    output_zip_path = 'small_images.zip'

    # --- Run the function ---
    create_subset_zip(small_json_path, large_zip_path, output_zip_path)