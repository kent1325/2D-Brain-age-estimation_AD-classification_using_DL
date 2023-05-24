import nibabel as nib
import os
import matplotlib.pyplot as plt
import datetime


file_extension = ".png"
date = datetime.date.today().strftime("%d-%m-%Y")
project_path = os.path.normpath(os.path.join(os.path.dirname(__file__), os.pardir))
base_path = os.path.join(project_path, "data", "processed")
print(f"Searching for t1_linear images in: {base_path}")


# Loop through files and folders
for root, dirs, files in os.walk(base_path):
    if "t1_linear" in root:
        for file_name in files:
            if ".nii.gz" in file_name:
                file_path = os.path.join(root, file_name)
                t1_img = nib.load(file_path)
                t1_data = t1_img.get_fdata()
                slices = [t1_data[60, :, :], t1_data[:, 155, :], t1_data[:, :, 118]]

                fig, axes = plt.subplots(1, len(slices))
                for i, s in enumerate(slices):
                    axes[i].imshow(s.T, cmap="gray", origin="lower")

                path = os.path.join(project_path, "reports", "brain_images", date)
                filename = file_name.split("_")[0] + "_" + file_name.split("_")[1]
                try:
                    if os.path.exists(path):
                        plt.savefig(
                            os.path.join(path, filename + file_extension),
                            bbox_inches="tight",
                        )

                    if not os.path.exists(path):
                        os.makedirs(path)
                        plt.savefig(
                            os.path.join(path, filename + file_extension),
                            bbox_inches="tight",
                        )
                    print(f"Successfully exported '{filename+file_extension}'")
                except Exception as e:
                    print("Error saving brain image: ", e)
print("Script finished saving all brain images.")
