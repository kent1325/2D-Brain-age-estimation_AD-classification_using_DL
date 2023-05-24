import os
import shutil

# Get the path to the directory containing the data to be processed
data_folder = input("Enter full path to MRI slices: ")
data_folder = data_folder.strip()
print(f"You entered the path: '{data_folder}'")

dest_folder = input("Enter destination folder (e.g. 'data/MRI_slices'): ")
dest_folder = dest_folder.strip()
print(f"You entered the path: '{dest_folder}'")

# Loop through files and folders
for root, dirs, files in os.walk(data_folder):
    if "deeplearning_prepare_data" in root and "t1_linear" in root:
        for file in files:
            filepath = os.path.join(root, file)
            shutil.copy2(filepath, dest_folder)
