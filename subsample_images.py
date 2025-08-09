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
    import posixpath

    # --- Step 1: Load required filenames exactly as in JSON ---
    print(f"Step 1: Loading required image filenames from '{os.path.basename(small_json_path)}'...")
    try:
        with open(small_json_path, 'r') as f:
            small_data = json.load(f)
        # Normalize to zip-style paths (forward slashes)
        required_paths = {posixpath.normpath(item['image']) for item in small_data if 'image' in item}
        print(f"Found {len(required_paths)} unique image filenames required.")
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return

    # --- Step 2: Build a map of (normalized_path -> actual_zip_path) ---
    print(f"Step 2: Building a file map from '{os.path.basename(large_zip_path)}'...")
    path_map = {}
    try:
        with zipfile.ZipFile(large_zip_path, 'r') as large_zip:
            for file_path in tqdm(large_zip.namelist(), desc="Scanning large zip"):
                if file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                    norm_path = posixpath.normpath(file_path)
                    path_map[norm_path] = file_path
        print("File map created successfully.")
    except Exception as e:
        print(f"Error reading the large zip file: {e}")
        return

    # --- Step 3: Create the new zip ---
    print(f"Step 3: Creating new zip file at '{output_zip_path}'...")
    not_found_count = 0
    with zipfile.ZipFile(large_zip_path, 'r') as large_zip:
        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as small_zip:
            for rel_path in tqdm(required_paths, desc="Writing to new zip"):
                if rel_path in path_map:
                    full_path_in_zip = path_map[rel_path]
                    image_data = large_zip.read(full_path_in_zip)
                    small_zip.writestr(full_path_in_zip, image_data)
                else:
                    print(f"Warning: Image '{rel_path}' from JSON not found in the large zip.")
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
    small_json_path = 'blip_laion_cc_sbu_558.json' 

    # This is the path to the large zip file
    large_zip_path = 'images1.zip'

    # This is the desired output path for your new, smaller zip file
    output_zip_path = 'smallest_images.zip'

    # --- Run the function ---
    create_subset_zip(small_json_path, large_zip_path, output_zip_path)