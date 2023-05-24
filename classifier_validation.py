import pickle
import torch
import torch.nn as nn
import GLT_Vector_Output_Single as gltvo
from torch.utils.data import DataLoader
import data_loader as dl
import MLP as mlp
import main as m
import sklearn.model_selection as sk
from sklearn.metrics import roc_auc_score
import numpy as np
torch.manual_seed(0) 

#Hyperparameters
LEARNING_RATE = 0.00001
BATCH_SIZE = 50
NUM_OF_EPOCHS = 100
INPUT_SIZE = 3072
HIDDEN_SIZE = INPUT_SIZE*2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "/home/student.aau.dk/ebuchb20/MRI_slices"
CSV_DIR = "data/image_labeling/05_split_balanced.csv"
FILENAME = "05_split_balanced_classifier"
PATH = f"models/classifier/{FILENAME}"
SENSITIVITY_METRIC = True #True AD minor class, False AD major class


SETTINGS = {
    "inplace": 3,
    "patch_size": 107,
    "step": 50,
    "nblock": 6,
    "drop_rate": 0.5,
    "backbone": "vgg8",
}

MLP_SETTINGS = {
    "input": INPUT_SIZE,
    "hidden": HIDDEN_SIZE,
    "learning_rate": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "epochs": NUM_OF_EPOCHS,
}


def RA_score(outputs: torch.tensor, target: torch.tensor):
    """computes the 

    Args:
        outputs (torch.tensor): _description_
        target (torch.tensor): _description_

    Returns:
        _type_: _description_
    """
    outputs = torch.sigmoid(outputs)
    score = roc_auc_score(target, outputs)
    return score


def sensitivity(outputs: torch.tensor, target: torch.tensor):
    """Computes the sensitivity metric for a batch of predictions

    Args:
        outputs (torch.tensor): the predictions
        target (torch.tensor): the ground truth

    Returns:
        torch.tensor: the sensitivity metric
    """
    outputs = torch.sigmoid(outputs)
    outputs = torch.round(outputs)
    tp = torch.sum(target * outputs)
    fn = torch.sum(target * (1 - outputs))
    sensitivity = tp / (tp + fn)
    return sensitivity

def specificity(outputs: torch.tensor, target: torch.tensor):
    """Computes the specificity metric for a batch of predictions

    Args:
        outputs (torch.tensor): the predictions
        target (torch.tensor): the ground truth

    Returns:
        torch.tensor: the specificity metric
    """
    outputs = torch.sigmoid(outputs)
    outputs = torch.round(outputs)
    tn = torch.sum((1 - target) * (1 - outputs))
    fp = torch.sum((1 - target) * outputs)
    specificity = tn / (tn + fp)
    return specificity

def accuracy_fn(outputs: torch.tensor, target: torch.tensor):
    """Computes the accuracy for a batch of predictions

    Args:
        outputs (torch.tensor): the predictions
        target (torch.tensor): the ground truth

    Returns:
        torch.tensor: returns the accuracy
    """
    outputs = torch.sigmoid(outputs)
    outputs = torch.round(outputs)
    correct = torch.eq(target, outputs).sum().item()
    acc = (correct / len(outputs)) * 100 
    return acc

def validation_epoch_average(model_dict:dict, epoch:int):
    """Calculates the average validation error for a k-fold validation.

    Args:
        model_dict (dictionery): the model dictionary with training errors
        epoch (int): the epoch for which the average validation error is calculated

    Returns:
        float: the average validation error for the epoch
    """
    error_type_list = ['val_accuracy', 'RA_score', 'sensitivity', 'specificity']
    accuracy_list, RA_list, sensitivity_list, specificity_list = [], [], [], []

    for error in error_type_list:
        for fold in range(1, 6):
            error_value = model_dict['fold_{}'.format(fold)]['epoch_{}'.format(epoch)][error]
            
            if error == 'val_accuracy':
                accuracy_list.append(error_value)
            elif error == 'RA_score':
                RA_list.append(error_value)
            elif error == 'sensitivity':
                sensitivity_list.append(error_value)
            elif error == 'specificity':
                specificity_list.append(error_value)


    accuracy_mean = np.mean(accuracy_list)
    accuracy_std = np.std(accuracy_list)
    ROC_AUC_mean = np.mean(RA_list)
    ROC_AUC_std = np.std(RA_list)
    sensitivity_mean = np.mean(sensitivity_list)
    sensitivity_std = np.std(sensitivity_list)
    specificity_mean = np.mean(specificity_list)
    specificity_std = np.std(specificity_list)


    return accuracy_mean, accuracy_std, ROC_AUC_mean, ROC_AUC_std, sensitivity_mean, sensitivity_std, specificity_mean, specificity_std

def score_calculator(dict:dict, epoch:int):
    """Finds the epoch with the best statistics

    Args:
        dict (dict): _description: The dictionary with the model information
        epoch (int): The maximum number of epochs
    """
    for epoch in range(1, epoch):
        if SENSITIVITY_METRIC:
            _,_,_,_,sensitivity,_,_,_ = validation_epoch_average(dict, epoch)
            if epoch == 1:
                best_score = (sensitivity, epoch)
            else:
                if sensitivity > best_score[0]:
                    best_score = (sensitivity, epoch)
        else:
            _,_,_,_,_,_,specificity,_ = validation_epoch_average(dict, epoch)
            if epoch == 1:
                best_score = (specificity, epoch)
            else:
                if specificity > best_score[0]:
                    best_score = (specificity, epoch)

    accuracy_mean, accuracy_std, ROC_AUC_mean, ROC_AUC_std, sensitivity_mean, sensitivity_std, specificity, specificity_std = validation_epoch_average(dict, best_score[1])
    print(f"Best epoch: {best_score[1]} \n \
            Validation accuracy: {accuracy_mean:.2f} \n \
            Validation accuracy sd: {accuracy_std:.2f} \n \
            ROC AUC score: {ROC_AUC_mean:.2f} \n \
            ROC AUC sd score: {ROC_AUC_std:.2f}\n \
            Sensitivity: {sensitivity_mean:.2f} \n \
            Sensitivity sd: {sensitivity_std:.2f} \n \
            Specificity: {specificity:.2f} \n \
            Specificity sd: {specificity_std:.2f}")


def _generate_model_errors(
    model: torch.nn.Module,
    feature_extractor: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.BCEWithLogitsLoss,
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

    epoch_loss = 0.0
    acc_list = []

    if epoch % 30 == 0:
        learning_rate *= 0.1 
        optimizer.param_groups[0]["lr"] = learning_rate
        print(
            f"Learning rate set to: {optimizer.state_dict()['param_groups'][0]['lr']}"
        )
    elif epoch % 15 == 0:
        learning_rate *= 0.5
        optimizer.param_groups[0]["lr"] = learning_rate
        print(
            f"Learning rate is set to: {optimizer.state_dict()['param_groups'][0]['lr']}\n"
        )


    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        if data.is_cuda == False and DEVICE.type != "cpu":
            print(f"Batch #{batch_idx} is NOT pushed to CUDA")

        optimizer.zero_grad()
        feature = feature_extractor(data)
        outputs = model(feature)
        target = target.unsqueeze(1)

        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        acc = accuracy_fn(outputs, target)
        acc_list.append(acc)
    epoch_accuracy = sum(acc_list)/len(acc_list)
    epoch_dict_key = "epoch_{}".format(epoch)

    return epoch_dict_key, epoch_accuracy

def model_validation(
        model: torch.nn.Module,
        feature_extractor: torch.nn.Module,
        val_loader: DataLoader
    ):
    """Computes the validation accuracy for a given model

    Args:
        model (torch.nn.Module): A model for a given fold and epoch
        val_loader (DataLoader): Validation dataset
    
    Returns:
        torch.tensor: The average validation accuracy
        toech.tensor: The average ROC-AUC score
        torch.tensor: The average sensitivity score
        torch.tensor: The average specificity score
    """
    
    val_accuracy = []
    sensitivity_list = []
    specificity_list = []
    model.eval()
    score_list = []
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            feature = feature_extractor(data)
            outputs = model(feature)
            target = target.unsqueeze(1)
            acc = accuracy_fn(outputs, target)
            val_accuracy.append(acc)
            
            model.to("cpu")
            feature_extractor.to("cpu")
            outputs, target = outputs.to("cpu"), target.to("cpu")
            try:
                score = RA_score(outputs, target)
                score_list.append(score)

            except Exception as e:
                print("Could not compute RU score", e)
            try:
                sens = sensitivity(outputs, target)
                spec = specificity(outputs, target)
                sensitivity_list.append(sens)
                specificity_list.append(spec)
            except Exception as e:
                print("Could not compute sens and spec", e)
            model.to(DEVICE)
            feature_extractor.to(DEVICE)

    average_score = (sum(score_list) / len(score_list))
    sensitivity_score = (sum(sensitivity_list) / len(sensitivity_list))
    specificity_score = (sum(specificity_list) / len(specificity_list))
    average_val_accuracy= (sum(val_accuracy) / len(val_accuracy))
    model.train()

    return average_val_accuracy, average_score, sensitivity_score, specificity_score


def validation_loop(
    model: torch.nn.Module,
    feature_extractor: torch.nn.Module,
    dataset: dl.LoadTensorImages,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.BCEWithLogitsLoss,
    epochs: int = NUM_OF_EPOCHS,
    batch_size: int = BATCH_SIZE,
):
    """Creates k-fold random samples from training set and stores a model for a given fold and given epoch

    Args:
        model (torch.nn.Module): The model
        dataset (dl.LoadTensorImages): The training set
        filename (str): The name of the model
        saving_step (int): The saving step determines how often to save the model
        dictionary (dict): The dictionary stores model information and error metrics
        optimizer (torch.optim.Optimizer): Optimizer object
        criterion (torch.nn.Module): The loss function
        epochs (int, optional): The number of epochs. Defaults to NUM_OF_EPOCHS
        batch_size (int, optional): The batch size. Defaults to BATCH_SIZE
        learning_rate (float, optional): _description_. Defaults to LEARNING_RATE.
    
    Returns:
        dict: The dictionary with model information and error metrics
    """
    dictionary = {}
    X = []
    y = []
    for item in dataset:
        features = item[0]  
        target = item[1]    
        X.append(features)
        y.append(target)
    kfold = sk.StratifiedKFold(n_splits=5)
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X,y)):
        #Sanity check if indeces are correctly distributed
        val_indices = [idx.item() for idx in val_idx]
        train_indices = [idx.item() for idx in train_idx]
        print("Elements in training set:", len(train_indices))
        print("Elements in validation set:", len(val_indices))
        print("Positive instances in validation set:", sum(y[idx.item()] for idx in val_idx))
        fold_dict_key = "fold_{}".format(fold + 1)
        if not fold_dict_key in dictionary:
            dictionary[fold_dict_key] = {}
    
        print("_________________.:fold no: {}:._________________".format(fold + 1))
        # Reset model parameters
        model.apply(m.reset_weights)
        model.train()
        print("Model parameters have been reset.")
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

        train_loader = DataLoader(
            dataset=dataset, batch_size=batch_size, sampler=train_subsampler
        )
        val_loader = DataLoader(
            dataset=dataset, batch_size=batch_size, sampler=val_subsampler
        )

        for epoch in range(1, epochs + 1):
            epoch_dict_key, epoch_accuracy = _generate_model_errors(
                model=model,
                feature_extractor=feature_extractor,
                train_loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                epoch=epoch,
            )
            val_accuracy, RA_score, sens_score, spec_score = model_validation(
                model=model,
                feature_extractor=feature_extractor, 
                val_loader=val_loader
            )
            dictionary[fold_dict_key][epoch_dict_key] = {
                "epoch_accuracy": epoch_accuracy,
                "val_accuracy": val_accuracy,
                "RA_score": RA_score,
                "sensitivity": sens_score,
                "specificity": spec_score,
            }
            print(
                f"Epoch {epoch} training accuracy: {epoch_accuracy}% \nValidation accuracy: {val_accuracy}%\nROC-AUC score: {RA_score}\nSensitivity: {sens_score}\nSpecificity: {spec_score}\n"
            )
    return dictionary
            

def model_trainer(
    model: torch.nn.Module,
    feature_extractor: torch.nn.Module,
    batch_size: int = BATCH_SIZE,
    epochs: int = NUM_OF_EPOCHS,
    learning_rate: float = LEARNING_RATE,
):
    """Trains a Classifier Neural Network model.

    Args:
        model (nn.Module): The model to train.
        dataset (dl.LoadTensorImages): The training dataset.
        filename (str): The filename used for when storing the model.
        saving_step (int): Number of steps where the models is saved.
        batch_size (int, optional): The size of the batch to enable parallelization. Defaults to BATCH_SIZE.
        epochs (int, optional): The number of epochs. Defaults to NUM_OF_EPOCHS.
        learning_rate (float, optional): The learning rate for the optimizer. Defaults to LEARNING_RATE.

    Returns:
        dict: The dictionary with model information and error metrics.
    """
    print("Initialized model trainer...")
    feature_extractor.to(DEVICE)
    if next(feature_extractor.parameters()).is_cuda == True:
        print("Feature extractor is pushed to CUDA")
    if next(feature_extractor.parameters()).is_cuda == False:
        print("Feature extractor is NOT pushed to CUDA")
    model.to(DEVICE)
    if next(model.parameters()).is_cuda == True:
        print("MLP is pushed to CUDA")
    if next(model.parameters()).is_cuda == False:
        print("MLP is NOT pushed to CUDA")
    dataset = m.dataset_init(CSV_DIR, DATA_DIR)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    #Validation loop
    model_accuracy_dict = validation_loop(
        model=model,
        feature_extractor=feature_extractor,
        dataset=dataset,
        optimizer=optimizer,
        criterion=criterion,
        epochs=epochs,
        batch_size=batch_size
    )
    model_accuracy_dict["settings"] = SETTINGS
    model_accuracy_dict['mlp_settings'] = MLP_SETTINGS
    with open(PATH, "wb") as file:
        pickle.dump(model_accuracy_dict, file)
    print("............................")
    print("Finished Validation on : {}".format(FILENAME))

    return model_accuracy_dict

#-----------------------------------------------------------#
if __name__ == "__main__":
    feature = gltvo.GlobalLocalBrainAge(
        inplace=SETTINGS["inplace"],
        patch_size=SETTINGS["patch_size"],
        step=SETTINGS["step"],
        nblock=SETTINGS["nblock"],
        backbone=SETTINGS["backbone"],
    )
    checkpoint = torch.load("models/classifier/split_05_model_epoch_100.pth.tar", map_location=DEVICE)
    feature.load_state_dict(checkpoint["model_state"])

    model = mlp.WideMLP(
        INPUT_SIZE, 
        HIDDEN_SIZE
    )
    print("\n\nStarting validation for: {}\n\n".format(FILENAME))
    model_dict = model_trainer(model, feature)
    score_calculator(model_dict, NUM_OF_EPOCHS)
    



