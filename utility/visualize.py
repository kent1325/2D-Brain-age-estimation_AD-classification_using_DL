from math import floor, ceil, sqrt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys
import os
import datetime
from torch import Tensor

sys.path.append("..")
import utility.plot_settings


class PlotData:
    def __init__(self) -> None:
        pass

    def _export_figure(self, filename: str) -> None:
        """Exports the plot into a folder, containing all figures.

        Args:
            filename (str): The name of the figure.
        """
        file_extension = ".png"
        date = datetime.date.today().strftime("%d-%m-%Y")
        path = f"reports/figures/{date}/"

        if os.path.exists(path):
            plt.savefig(path + filename + file_extension, bbox_inches="tight")

        if not os.path.exists(path):
            os.makedirs(path)
            plt.savefig(path + filename + file_extension, bbox_inches="tight")

        print(f"Successfully exported '{filename+file_extension}'")

    def plot_training_patch_error_by_epoch(self, training_error: dict) -> None:
        """Plots the average error for each patch by epochs.

        Args:
            training_error (dict): The dictionary containing the average error.
        """
        # Create canvas
        fig, ax = plt.subplots()

        # Get the list of patches from the dictionary
        patches = list(training_error.keys())

        # Loop over all the patches and plot their average batch values by epoch
        for patch in patches:
            # Get the list of average batch values for the current patch
            avg_batch_values = training_error[patch]

            # Compute the list of errors for the current patch
            errors = [avg_batch_value for avg_batch_value in avg_batch_values]

            # Plot the errors by epoch for the current patch
            ax.plot(
                range(1, len(errors) + 1), errors, label=f"Patch #{patch}", marker="o"
            )

        # Set the title, x-axis label, and y-axis label
        ax.set_title("Development of Patch error by Epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Error")

        # Set legend
        fig.legend(
            loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes
        )

        # Styling
        ax.grid(True)
        ax.tick_params(left=False, bottom=False, color="black")

        # Export figure
        self._export_figure(filename="Training Patch Error by Epoch")

        # Show the plot
        # plt.show()

    def plot_validation_error(
        self, loss_dict: dict, epochs: int, filename: str
    ) -> None:
        """Plots the training and validation errors for a k-fold validation.

        Args:
            loss_dict (dictionery): the model dictionary with training errors
            epochs (int): the amount of epochs the model was trained on
        """
        # Create canvas
        fig, ax = plt.subplots()

        error_type_list = ["epoch_error", "val_error"]
        epoch_error_list = []
        for error in error_type_list:
            for fold in range(1, 6):
                for key in loss_dict["fold_{}".format(fold)]:
                    epoch_error_list.append(
                        loss_dict["fold_{}".format(fold)][key][error]
                    )
        error_list, validation_list = (
            epoch_error_list[: len(epoch_error_list) // 2],
            epoch_error_list[len(epoch_error_list) // 2 :],
        )

        average_epoch_error = []
        average_validation_error = []
        for j in range(epochs):
            error_sum = 0
            val_sum = 0
            for i in range(j, len(error_list), epochs):
                error_sum += error_list[i]
                val_sum += validation_list[i]
            average_epoch_error.append(error_sum)
            average_validation_error.append(val_sum)
        average_epoch_error = [x / 5 for x in average_epoch_error]
        average_validation_error = [x / 5 for x in average_validation_error]

        # Creating plot of training and validation error
        ax.plot(
            average_epoch_error,
            marker="o",
            markersize=3,
            linewidth=1.5,
            label="Training error",
        )
        ax.plot(
            average_validation_error,
            marker="o",
            markersize=3,
            linewidth=1.5,
            label="Validation error",
        )

        # ax.legend(labels=["Training error", "Validation error"])
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Error")
        ax.set_title("Training and validation error", size=20)

        # Set legend
        fig.legend(
            loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes
        )

        # Styling
        ax.grid(True)
        ax.tick_params(left=False, bottom=False, color="black")

        # Export figure
        self._export_figure(filename=f"{filename} - Training and validation error")

        # plt.show()

    def plot_training_error(self, loss_dict: dict, filename: str) -> None:
        """Plots the training error over epochs.

        Args:
            loss_dict (dictionery): the model dictionary with training errors

        """
        # Create canvas
        fig, ax = plt.subplots()

        epoch_error_list = []

        for key in loss_dict.keys():
            if "epoch" in key:
                epoch_error_list.append(loss_dict[key]["epoch_error"])

        ax.plot(epoch_error_list, marker="o", markersize=3, linewidth=1.5)

        ax.set_xlabel("Epochs")
        ax.set_ylabel("Error")
        ax.set_title("Training error", size=20)

        # Styling
        ax.grid(True)
        ax.tick_params(left=False, bottom=False, color="black")

        # Export figure
        self._export_figure(filename=f"{filename} - Training error")

        # plt.show()

    def plot_tensor_to_image(
        self, tensor_data: Tensor, filename: str, batch_idx: int, epoch: int
    ) -> None:
        for i in range(len(tensor_data)):
            plt.subplot(
                floor(sqrt(len(tensor_data))), ceil(sqrt(len(tensor_data))), i + 1
            )
            plt.imshow(tensor_data[i][0], cmap="gray")
        plt.plot()

        # Export figure
        self._export_figure(
            filename=f"{filename}_epoch_{epoch}_batch-idx_{batch_idx} - Tensor to image - {str(datetime.datetime.now())}"
        )


# region template_plot
# def template_plot(self, df):
#     # Create canvas
#     fig, ax1 = plt.subplots()

#     # Make plots
#     ax1.plot(df.index, df["Motor Active Power"], label="Motor Active Power")
#     ax1.plot(df.index, df["Motor Apparent Power"], label="Motor Apparent Power")
#     ax1.plot(df.index, df["Motor Reactive Power"], label="Motor Reactive Power")
#     ax1.plot(df.index, df["Motor Shaft Power"], label="Motor Shaft Power")

#     # Adjust date ticks
#     ax1.xaxis.set_major_locator(mdates.DayLocator(interval=1))
#     ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m-%Y"))

#     # Adjust y-ticks and limits
#     y1_min = 12
#     y1_max = 17

#     ax1.set_ylim([y1_min, y1_max])
#     ax1.set_yticks(np.arange(y1_min, y1_max + 1, 1))

#     # Set labels
#     ax1.set_title("Motor Power")
#     ax1.set_xlabel("Timestamp", color="black")
#     ax1.set_ylabel("Power (W)", color="black")

#     # Set legend
#     fig.legend(
#         loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes
#     )

#     # Styling
#     ax1.grid(True)
#     ax1.tick_params(left=False, bottom=False, color="black")

#     # Export figure
#     self._export_figure(filename="Motor Power (4x)")

#     # Show plot in interactive Python
#     plt.show()
# endregion
