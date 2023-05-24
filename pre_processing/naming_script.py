import os

# Get the path to the directory containing the data to be processed
data_folder = input("Enter full path to unprocessed data (bids data dir): ")
data_folder = data_folder.strip()
print(f"You entered the path: '{data_folder}'")

# Loop through files and folders
for root, dirs, files in os.walk(data_folder):
    for file_name in files:
        # Check for 'sess' and replace with 'ses'
        if "sess" in file_name:
            new_file_name = file_name.replace("sess", "ses")
            os.rename(os.path.join(root, file_name), os.path.join(root, new_file_name))
            print(f"Renamed {file_name} to {new_file_name}")

    # Check if folder contains multiple runs and remove all except latest run
    if "anat" in root:
        if (
            len(
                [
                    entry
                    for entry in os.listdir(root)
                    if os.path.isfile(os.path.join(root, entry))
                ]
            )
            / 2
        ) > 1:
            is_first_valid_file = False
            for index, file_name in enumerate(files):
                if not is_first_valid_file and "echo" not in file_name:
                    run_num = file_name.split("run-")[1].split(".")[0]
                    is_first_valid_file = True
                if run_num not in file_name or "echo" in file_name:
                    run_path = os.path.join(root, file_name)
                    os.remove(run_path)
                    print(f"Deleted {file_name}")

print("Done fixing naming issues...")
