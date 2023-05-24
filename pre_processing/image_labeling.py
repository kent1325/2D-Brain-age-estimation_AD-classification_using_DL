import pandas as pd
from pprint import pprint
import os

SUBJECT_TEST = "data/image_labeling/subject_test.csv"
SUBJECT_TRAIN = "data/image_labeling/subject_train.csv"


def split_subject_csv(oasis_csv_path: str, split: float = 0.9) -> None:
    df = pd.read_csv(oasis_csv_path)

    # Calculate the number of samples in the training set
    split_param = create_train_test_split(dataset=df, split=split)

    # Split dataframe into training and test sets and save to csv
    df_train = df.iloc[:split_param]
    df_test = df.iloc[split_param:]

    df_train.to_csv(SUBJECT_TRAIN, index=False)
    df_test.to_csv(SUBJECT_TEST, index=False)

    print(f"Splitted the subject csv into {SUBJECT_TRAIN} and {SUBJECT_TEST}.")

    return SUBJECT_TRAIN, SUBJECT_TEST


def image_labeling(oasis_path: str, subject_path: str, dest_path: str) -> None:
    """
    Get image labeling data from csv files
    """
    oasis_frame = pd.read_csv(subject_path)
    oasis_frame["session"] = oasis_frame["MR ID"].str.split("MR_", n=1, expand=True)[1]

    # List to store the split file names
    split_file_names = []

    # Iterate through each file in the directory
    for file_name in os.listdir(oasis_path):
        # Split the file name on "-"
        split_name = file_name.split("-")
        subject = split_name[1].split("_")[0]
        session = split_name[2].split("_")[0]
        split_file_names.append([subject, session, file_name])

    # Create a pandas dataframe from the list of split names
    df = pd.DataFrame(split_file_names, columns=["Subject", "session", "filename"])
    cleaned_df = df.merge(oasis_frame, how="inner", on=["Subject", "session"])

    count_df = cleaned_df.groupby(["Subject", "session"]).count()

    if (
        count_df.loc[count_df["filename"] < 5].empty
        and count_df.loc[count_df["filename"] > 5].empty
    ):
        label_map = df.merge(oasis_frame, how="inner", on=["Subject", "session"])
        label_map = label_map[["filename", "Age"]].sort_values(by=["filename"])

        if not os.path.isfile(dest_path):
            label_map.to_csv(dest_path, index=False)
        else:
            label_map.to_csv(
                dest_path,
                mode="a",
                header=False,
                index=False,
            )
        print(f"Created/appended slices and labels to {dest_path}")
    else:
        pprint(count_df.loc[count_df["filename"] < 5])
        pprint(count_df.loc[count_df["filename"] > 5])


def create_train_test_split(dataset: pd.DataFrame, split: float) -> int:
    """Computes the training set sample size.

    Args:
        dataset (pd.DataFrame): The dataset to be split.
        split (float, optional): The split criteria given in a percentage (e.g. 0.7). Defaults to TRAIN_TEST_SPLIT.

    Returns:
        num_train_samples (int): The traning sample size.
    """
    # Calculate the number of samples in the training set
    num_train_samples = round((len(dataset) / 5) * split) * 5
    print(
        f"Created the split criteria" #\n\t train size: {num_train_samples},\n\t test size: {len(dataset) - num_train_samples}."
    )

    return num_train_samples
