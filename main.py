import datetime
import math
import os
import pickle
from pprint import pprint
import torch
import torch.nn as nn
import GlobalLocalTransformer as gl
from torch.utils.data import DataLoader
import data_loader as dl
import pre_processing.image_labeling as il
from utility.visualize import PlotData
import sklearn.model_selection as sk
import pandas as pd
import uuid

# Hyperparameters
IS_TRAINING = False
VALIDATION = False
IS_SICK = False
LEARNING_RATE = 0.0001
TRAIN_TEST_SPLIT = 0.9
BATCH_SIZE = 25
NUM_OF_EPOCHS = 100
TEST_BATCH_SIZE = 5
CURRENT_DATE = datetime.date.today().strftime("%d-%m-%Y")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_CSV = "data/image_labeling/label_list_train.csv"
TEST_CSV = "data/image_labeling/label_list_test.csv"
SICK_TEST_CSV = "data/image_labeling/sick_test.csv"
SICK_SUBJECT_CSV = "data/image_labeling/sick_subjects.csv"
HEALTHY_SUBJECT_CSV = "data/image_labeling/healthy_subjects.csv"
DATA_DIR = "data/MRI_slices"
SETTINGS = {
    "inplace": 3,
    "patch_size": 84,
    "step": 41,
    "nblock": 6,
    "drop_rate": 0.5,
    "backbone": "vgg8",
}


def model_saver(
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    filename: str,
    loss: dict,
    fold: int = -1,
):
    """Saves model parameters to a file.

    Arguments:
        epoch: The current epoch.
        model: The model to save.
        optimizer: The optimizer to save.
        filename: The name of the file to save to.
        loss: The loss of the model.
    """

    checkpoint = {
        "epoch": epoch,
        "model_name": filename,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
    }
    path = f"models/{CURRENT_DATE}/"
    if fold != -1:
        file_name = f"{filename}_fold_{fold}_epoch_{epoch}.pth.tar"
        checkpoint["fold"] = fold
    else:
        file_name = f"{filename}_epoch_{epoch}.pth.tar"
    try:
        if os.path.exists(path):
            torch.save(checkpoint, path + file_name)

        if not os.path.exists(path):
            os.makedirs(path)
            torch.save(checkpoint, path + file_name)
        print(f"Model '{file_name}' is saved")
    except Exception as e:
        print("Error saving model: ", e)
    _loss_saver(epoch=epoch, fold=fold, filename=filename, loss=loss)


def _loss_saver(
    epoch: int,
    filename: str,
    loss: dict,
    fold: int = -1,
):
    """Saves model parameters to a file.

    Arguments:
        epoch: The current epoch.
        filename: The name of the file to save to.
        loss: The loss of the model.
    """
    path = f"models/{CURRENT_DATE}/loss/"
    if fold != -1:
        file_name = f"{filename}_fold_{fold}_epoch_{epoch}_loss.pkl"
    else:
        file_name = f"{filename}_epoch_{epoch}_loss.pkl"
    try:
        if os.path.exists(path):
            with open(path + file_name, "wb") as f:
                pickle.dump(loss, f)

        if not os.path.exists(path):
            os.makedirs(path)
            with open(path + file_name, "wb") as f:
                pickle.dump(loss, f)
        print(f"Loss '{file_name}' is saved")
    except Exception as e:
        print("Error saving model: ", e)


def model_loader(
    filename: str,
    epoch: int,
    date: str,
    fold: int = -1,
    device: torch.device = DEVICE,
    learning_rate: float = LEARNING_RATE,
):
    """Loads model parameters from a file.

    Args:
        filename (str): The name of the file to load from.
        epoch (int): The epoch to load.
        learning_rate (float, optional): The learning rate for the optimizer. Defaults to LEARNING_RATE.

    Returns:
        model, optimizer, loss (tuple) : Returns a tuple containing the model, optimizer, and loss.
    """
    path = f"models/{date}/"
    if fold != -1:
        file_name = f"{filename}_fold_{fold}_epoch_{epoch}.pth.tar"
    else:
        file_name = f"{filename}_epoch_{epoch}.pth.tar"
    try:
        # Load model parameters
        checkpoint = torch.load(path + file_name, map_location=device)
        print(f"Model '{file_name}' is loaded")
    except Exception as e:
        print("Error loading model: ", e)
        return None, None, None

    # Initialize model
    model = gl.GlobalLocalBrainAge(
        inplace=SETTINGS["inplace"],
        patch_size=SETTINGS["patch_size"],
        step=SETTINGS["step"],
        nblock=SETTINGS["nblock"],
        backbone=SETTINGS["backbone"],
    )

    # Load model parameters
    model.load_state_dict(checkpoint["model_state"])

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Load optimizer parameters
    optimizer.load_state_dict(checkpoint["optim_state"])
    loss = _loss_loader(filename=filename, epoch=epoch, date=date, fold=fold)

    return model, optimizer, loss


def _loss_loader(filename: str, epoch: int, date: str, fold: int = -1):
    path = f"models/{date}/loss/"
    if fold != -1:
        file_name = f"{filename}_fold_{fold}_epoch_{epoch}_loss.pkl"
    else:
        file_name = f"{filename}_epoch_{epoch}_loss.pkl"
    try:
        with open(path + file_name, "rb") as f:
            loss = pickle.load(f)
        print(f"Loss '{file_name}' is loaded")
        return loss
    except Exception as e:
        print("Error loading model: ", e)


def model_trainer(
    model: torch.nn.Module,
    dataset: dl.LoadTensorImages,
    filename: str,
    saving_step: int,
    batch_size: int = BATCH_SIZE,
    epochs: int = NUM_OF_EPOCHS,
    learning_rate: float = LEARNING_RATE,
):
    """Trains a Neural Network model.

    Args:
        model (nn.Module): The model to train.
        dataset (dl.LoadTensorImages): The training dataset.
        filename (str): The filename used for when storing the model.
        saving_step (int): Number of steps where the models is saved.
        batch_size (int, optional): The size of the batch to enable parallelization. Defaults to BATCH_SIZE.
        epochs (int, optional): The number of epochs. Defaults to NUM_OF_EPOCHS.
        learning_rate (float, optional): The learning rate for the optimizer. Defaults to LEARNING_RATE.
    """
    print("Initialized model trainer...")
    model.to(DEVICE)
    if next(model.parameters()).is_cuda == True:
        print("This model is pushed to CUDA")
    if next(model.parameters()).is_cuda == False:
        print("This model is NOT pushed to CUDA")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.L1Loss()
    model_error = {}
    model_error["settings"] = SETTINGS
    plot = PlotData()

    if VALIDATION:
        validation_loop(
            model=model,
            train_set=dataset,
            filename=filename,
            saving_step=saving_step,
            dictionary=model_error,
            optimizer=optimizer,
            criterion=criterion,
            epochs=epochs,
            batch_size=batch_size,
        )
        model, optimizer, loss = model_loader(
            filename=filename, epoch=epochs, date=CURRENT_DATE, fold=5
        )
        plot.plot_validation_error(loss_dict=loss, epochs=epochs, filename=filename)
    else:
        training_loop(
            model=model,
            train_set=dataset,
            filename=filename,
            saving_step=saving_step,
            dictionary=model_error,
            optimizer=optimizer,
            criterion=criterion,
            epochs=epochs,
            batch_size=batch_size,
        )
        model, optimizer, loss = model_loader(
            filename=filename, epoch=epochs, date=CURRENT_DATE
        )
        plot.plot_training_error(loss_dict=loss, filename=filename)

    print("Finished training the model...")


def model_tester(
    model: torch.nn.Module,
    dataset: dl.LoadTensorImages,
    batch_size: int = TEST_BATCH_SIZE,
    device: torch.device = DEVICE,
) -> list:
    """Tests the trained model on unseen data.

    Args:
        model (torch.nn.Module): The model to use for testing.
        dataset (dl.LoadTensorImages): The tensor dataset.
        split_param (int): The split parameter that determines the size of test data.
        batch_size (int, optional): A batch size that must always be equal to the number of slices for each brain. Defaults to BATCH_SIZE.

    Returns:
        list: The average error for each batch.
    """
    print("Initializing model tester...")
    loss = nn.L1Loss()
    errors = []
    test_loader = DataLoader(dataset=dataset, shuffle=False, batch_size=batch_size)
    model.eval()
    model.to(device)
    df = pd.DataFrame(columns=["Age", "Predicted Age", "Error"])
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(test_loader):
            data, targets = data.to(device), targets.to(device)
            targets.unsqueeze_(1)
            patch_predictions = model(data)
            patch_predictions = patch_predictions[1:]
            patch_stack_tensor = torch.stack(patch_predictions, dim=1)
            prediction = torch.mean(patch_stack_tensor, dim=1)
            error = loss(prediction, targets)
            errors.append(error.item())
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        {
                            "Age": [targets[0].item()],
                            "Predicted Age": [torch.mean(prediction).item()],
                            "Error": [error.item()],
                        }
                    ),
                ],
                ignore_index=True,
            )
    print("Finished testing the model")
    error_scalar = sum(errors) / len(errors)

    return error_scalar, df


def model_validation(model: torch.nn.Module, val_loader: DataLoader):
    """Computes validation error

    Args:
        model (torch.nn.Module): A model for a given fold and epoch
        val_loader (DataLoader): Validation dataset
    """
    patch_val_dict = {}
    batch_error = []
    criterion = nn.L1Loss()
    # Start model evaluation
    model.eval()
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            target.unsqueeze_(1)
            predictions = model(data)
            # Remove the first prediction which is the global prediction
            predictions = predictions[1:]
            criteria = [criterion] * len(predictions)
            targets = [target] * len(predictions)
            val_loss = 0
            patch = 1
            for prediction, target, criterion in zip(predictions, targets, criteria):
                error = criterion(prediction, target)
                val_loss += error
                if not patch in patch_val_dict:
                    patch_val_dict[patch] = error.item()
                else:
                    patch_val_dict[patch] += error.item()
                patch += 1
            batch_error.append(val_loss)
    patch_val_dict = {
        key: value / (len(batch_error)) for key, value in patch_val_dict.items()
    }
    average_val_batch_error = (sum(batch_error) / len(batch_error)) / len(predictions)
    # Start training again
    model.train()

    return average_val_batch_error, patch_val_dict


def _generate_model_errors(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.L1Loss,
    epoch: int,
    epochs: int = NUM_OF_EPOCHS,
    learning_rate: float = LEARNING_RATE,
):
    """A utility function that computes the errors when looping through epochs and stores those values

    Args:
        model (torch.nn.Module): The model
        train_loader (DataLoader): The training set
        optimizer (torch.optim.Optimizer): The optimizer object
        criterion (torch.nn.L1Loss): The loss function
        epoch (int): The current epoch number
        epochs (int, optional): The total number of epochs. Defaults to NUM_OF_EPOCHS.
        learning_rate (float, optional): The learning rate. Defaults to LEARNING_RATE.

    Returns:
        The epoch key, the average batch error for a given epoch, and the patch dictionary
    """
    print(f"Initializing epoch {epoch} of {epochs}.....")
    patch_dict = {}
    batch_loss = []
    plot = PlotData()

    if epoch % 25 == 0:
        learning_rate *= 0.5
        optimizer.param_groups[0]["lr"] = learning_rate
        print(
            f"Learning rate is set to: {optimizer.state_dict()['param_groups'][0]['lr']}"
        )

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        if data.is_cuda == False and DEVICE.type != "cpu":
            print(f"Batch #{batch_idx} is NOT pushed to CUDA")

        if IS_TRAINING and not VALIDATION:
            plot.plot_tensor_to_image(
                tensor_data=data, filename=filename, batch_idx=batch_idx, epoch=epoch
            )
        target.unsqueeze_(1)
        optimizer.zero_grad()
        outputs = model(data)
        # Remove the first output which is the global prediction
        outputs = outputs[1:]
        criteria = [criterion] * len(outputs)
        targets = [target] * len(outputs)
        loss = 0
        patch = 1
        for output, target, criterion in zip(outputs, targets, criteria):
            loss = loss + criterion(output, target)
            error = criterion(output, target)
            if not patch in patch_dict:
                patch_dict[patch] = error.item()
            else:
                patch_dict[patch] += error.item()
            patch += 1
        batch_loss.append(loss)
        loss.backward()
        optimizer.step()

    average_batch_error = (sum(batch_loss) / (len(batch_loss))) / len(outputs)
    patch_dict = {key: value / (len(batch_loss)) for key, value in patch_dict.items()}
    epoch_dict_key = "epoch_{}".format(epoch)

    return epoch_dict_key, average_batch_error, patch_dict


def validation_loop(
    model: torch.nn.Module,
    train_set: dl.LoadTensorImages,
    filename: str,
    saving_step: int,
    dictionary: dict,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.L1Loss,
    epochs: int = NUM_OF_EPOCHS,
    batch_size: int = BATCH_SIZE,
):
    """Creates k-fold random samples from training set and stores a model for a given fold and given epoch

    Args:
        model (torch.nn.Module): The model
        train_set (dl.LoadTensorImages): The training set
        filename (str): The name of the model
        saving_step (int): The saving step determines how often to save the model
        dictionary (dict): The dictionary stores model information and error metrics
        optimizer (torch.optim.Optimizer): Optimizer object
        criterion (torch.nn.Module): The loss function
        epochs (int, optional): The number of epochs. Defaults to NUM_OF_EPOCHS
        batch_size (int, optional): The batch size. Defaults to BATCH_SIZE
        learning_rate (float, optional): _description_. Defaults to LEARNING_RATE.
    """
    # Configuration for k-fold cross validation
    kfold = sk.KFold(n_splits=5, shuffle=False)
    # Iterate through folds
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_set)):
        fold_dict_key = "fold_{}".format(fold + 1)
        if not fold_dict_key in dictionary:
            dictionary[fold_dict_key] = {}
        print("_________________.:fold no: {}:._________________".format(fold + 1))
        # Reset model parameters
        model.apply(reset_weights)
        print("Model parameters have been reset.")
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

        train_loader = DataLoader(
            dataset=train_set, batch_size=batch_size, sampler=train_subsampler
        )
        val_loader = DataLoader(
            dataset=train_set, batch_size=batch_size, sampler=val_subsampler
        )

        for epoch in range(1, epochs + 1):
            epoch_dict_key, average_batch_error, patch_dict = _generate_model_errors(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                epoch=epoch,
            )
            val_error, patch_val_dict = model_validation(
                model=model, val_loader=val_loader
            )
            dictionary[fold_dict_key][epoch_dict_key] = {
                "epoch_error": average_batch_error.item(),
                "patch_error": patch_dict,
                "val_error": val_error.item(),
                "patch_val_error": patch_val_dict,
            }
            print(
                f"Average patch loss for epoch {epoch}: {average_batch_error} \nValidation loss: {val_error}\n"
            )

            if ((epoch) % saving_step) == 0:
                model_saver(
                    epoch=epoch,
                    fold=fold + 1,
                    model=model,
                    optimizer=optimizer,
                    filename=filename,
                    loss=dictionary,
                )


def training_loop(
    model: torch.nn.Module,
    train_set: dl.LoadTensorImages,
    filename: str,
    saving_step: int,
    dictionary: dict,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.L1Loss,
    epochs: int = NUM_OF_EPOCHS,
    batch_size: int = BATCH_SIZE,
):
    """Trains model with set hyperparameters

    Args:
        model (torch.nn.Module): The model with the best hyperparameters
        train_set (dl.LoadTensorImages): The training set
        filename (str): The model name
        saving_step (int): Saving every 'saving_step' model
        dictionary (dict): A dictionary containing error metrics and model information
        optimizer (torch.optim.Optimizer): Optimizer object
        criterion (torch.nn.L1Loss): The loss function
        epochs (int, optional): Number of epochs. Defaults to NUM_OF_EPOCHS.
        batch_size (int, optional): Batch size. Defaults to BATCH_SIZE.
        learning_rate (float, optional): The initial learning rate. Defaults to LEARNING_RATE.
    """
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=False)

    for epoch in range(1, epochs + 1):
        epoch_dict_key, average_batch_error, patch_dict = _generate_model_errors(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            epoch=epoch,
        )
        dictionary[epoch_dict_key] = {
            "epoch_error": average_batch_error.item(),
            "patch_error": patch_dict,
        }

        if ((epoch) % saving_step) == 0:
            model_saver(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                filename=filename,
                loss=dictionary,
            )

        print(f"Average loss epoch {epoch}: {average_batch_error}\n")


def dataset_init(csv_dir: str, data_dir: str = DATA_DIR) -> dl.LoadTensorImages:
    """Initializes a dataset object of type LoadTensorImages.

    Args:
        csv_dir (str): The path to the csv file.
        data_dir (str): The path to the data directory. Defaults to DATA_DIR.

    Returns:
        dataset (LoadTensorImages): An instance of LoadTensorImages.
    """
    # Ensure that the csv directory exists
    if not os.path.exists(csv_dir):
        raise FileNotFoundError(
            f"The specified csv directory does not exist: {csv_dir}"
        )
    # Ensure that the data directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"The specified data directory does not exist: {data_dir}"
        )
    # Initialize the dataset
    dataset = dl.LoadTensorImages(csv_file=csv_dir, root_dir=data_dir)
    print("Dataset initialized.")
    return dataset


def reset_weights(m: torch.nn.Module):
    """Try resetting model weights to avoid weight leakage.

    Args:
        m (torch.nn.Module): The model to reset.
    """
    for layer in m.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()


def compute_tensor_size(
    image_dim: torch.Size,
    patch_size: int = SETTINGS["patch_size"],
    step_size: int = SETTINGS["step"],
) -> int:
    """Computes the number of patches in a tensor.

    Args:
        image_dim (torch.Size): The size of the tensor.
        patch_size (int, optional): The wanted patch size. Defaults to SETTINGS["patch_size"].
        step_size (int, optional): The wanted step size. Defaults to SETTINGS["step"].

    Returns:
        num_patches (int): The total number of patches.
    """
    # compute the number of patches in each dimension
    num_patches_height = math.floor((image_dim.size(2) - patch_size) / step_size) + 1
    num_patches_width = math.floor((image_dim.size(3) - patch_size) / step_size) + 1

    # compute the total number of patches
    num_patches = num_patches_height * num_patches_width + 1
    print(
        f"Created number of local patches: {num_patches}, when image dimensions are: {image_dim.size()}."
    )

    return num_patches


if __name__ == "__main__":
    torch.manual_seed(0)

    if IS_TRAINING:
        # Initialize the image labeling dictionary for train and test
        if not os.path.isfile(TEST_CSV) or not os.path.isfile(TRAIN_CSV):
            train_subset, test_subset = il.split_subject_csv(
                oasis_csv_path=HEALTHY_SUBJECT_CSV, split=TRAIN_TEST_SPLIT
            )
            il.image_labeling(
                oasis_path=DATA_DIR, subject_path=train_subset, dest_path=TRAIN_CSV
            )
            il.image_labeling(
                oasis_path=DATA_DIR, subject_path=test_subset, dest_path=TEST_CSV
            )
        dataset = dataset_init(csv_dir=TRAIN_CSV, data_dir=DATA_DIR)
        model = gl.GlobalLocalBrainAge(
            inplace=SETTINGS["inplace"],
            patch_size=SETTINGS["patch_size"],
            step=SETTINGS["step"],
            nblock=SETTINGS["nblock"],
            backbone=SETTINGS["backbone"],
        )
        filename = f"{model=}".split("=")[0]

        model_trainer(
            model=model,
            dataset=dataset,
            saving_step=NUM_OF_EPOCHS,  # A saving step > NUM_OF_EPOCHS results en no saves.
            batch_size=BATCH_SIZE,
            epochs=NUM_OF_EPOCHS,
            filename=filename,
        )
    else:
        filename = "tester_model"
        epoch = NUM_OF_EPOCHS
        date = "29-04-2023"
        model, optimizer, loss = model_loader(
            filename=filename, epoch=NUM_OF_EPOCHS, date=date
        )

        if IS_SICK:
            if not os.path.isfile(SICK_TEST_CSV):
                il.image_labeling(
                    oasis_path=DATA_DIR,
                    subject_path=SICK_SUBJECT_CSV,
                    dest_path=SICK_TEST_CSV,
                )
            dataset = dataset_init(csv_dir=SICK_TEST_CSV, data_dir=DATA_DIR)
            source = SICK_SUBJECT_CSV.split("/")[-1].split(".")[0]
        else:
            dataset = dataset_init(csv_dir=TEST_CSV, data_dir=DATA_DIR)
            source = HEALTHY_SUBJECT_CSV.split("/")[-1].split(".")[0]

        test_loss, df = model_tester(model=model, dataset=dataset)
        print(test_loss)

        path = f"models/{date}/loss/"
        df_filename = f"df_{source}_{filename}_epoch_{epoch}_{str(uuid.uuid4())}.pkl"
        try:
            if os.path.exists(path):
                df.to_pickle(path + df_filename)

            if not os.path.exists(path):
                os.makedirs(path)
                df.to_pickle(path + df_filename)
            print(f"Dataframe '{df_filename}' is saved")
        except Exception as e:
            print("Error saving dataframe: ", e)

        test_filename = "test_loss_" + filename
        _loss_saver(epoch=epoch, filename=test_filename, loss=test_loss)
