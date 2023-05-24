import pandas as pd
import os
import numpy as np


SICK_CLASSIFIER_CSV = "sick_subjects_classifier.csv"
HEALTHY_CLASSIFIER_CSV = "healthy_subjects_classifier.csv"
MIXED_CLASSIFIER_TRAIN_CSV = "mixed_subjects_classifier_train.csv"
MIXED_CLASSIFIER_TEST_CSV = "mixed_subjects_classifier_test.csv"
TRAIN_TEST_SPLIT = 0.9
USE_BALANCED_DATASET = True


def balance_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()

    # create separate dataframes for filenames with and without AD
    df_ad = df_copy[df_copy["has_ad"] == 1]
    df_no_ad = df_copy[df_copy["has_ad"] == 0]

    # find the length of the dataframes
    len_ad = len(df_ad)
    len_no_ad = len(df_no_ad)

    # randomly sample a subset of the longer dataframe to make the lengths equal
    if len_ad > len_no_ad:
        df_ad = df_ad.iloc[:len_no_ad]
    else:
        df_no_ad = df_no_ad.iloc[:len_ad]

    # concatenate the two dataframes into a single dataframe
    df_balanced = pd.concat([df_ad, df_no_ad], ignore_index=True)

    return df_balanced


project_path = os.path.normpath(os.path.join(os.path.dirname(__file__), os.pardir))
base_path = os.path.join(project_path, "data", "image_labeling")

# Read the datasets
df_healthy = pd.read_csv(os.path.join(base_path, HEALTHY_CLASSIFIER_CSV))
df_sick = pd.read_csv(os.path.join(base_path, SICK_CLASSIFIER_CSV))

# Append the datasets
df_mixed = pd.concat([df_healthy, df_sick], ignore_index=True)

# Create a new column that combines the subject ID and session ID
df_mixed["subject_session"] = (
    df_mixed["filename"].str.split("_", n=2, expand=True)[0]
    + "_"
    + df_mixed["filename"].str.split("_", n=2, expand=True)[1]
)

if USE_BALANCED_DATASET:
    df_mixed = balance_dataset(df_mixed)

# Split the data into two groups randomly, while ensuring that all slices for a given subject/session are in the same group
unique_subject_sessions = df_mixed["subject_session"].unique()
np.random.shuffle(unique_subject_sessions)

# Split the dataset into train and test
split_param = round((len(unique_subject_sessions) / 5) * TRAIN_TEST_SPLIT) * 5
group1_subject_sessions = unique_subject_sessions[:split_param]
group2_subject_sessions = unique_subject_sessions[split_param:]

df_mixed_train = (
    df_mixed.set_index("subject_session").loc[group1_subject_sessions].reset_index()
)
df_mixed_test = (
    df_mixed.set_index("subject_session").loc[group2_subject_sessions].reset_index()
)

# Remove the subject_session column from the resulting dataframes
df_mixed_train = df_mixed_train.drop(columns=["subject_session"])
df_mixed_test = df_mixed_test.drop(columns=["subject_session"])

# Save the merged dataset
df_mixed_train.to_csv(os.path.join(base_path, MIXED_CLASSIFIER_TRAIN_CSV), index=False)
df_mixed_test.to_csv(os.path.join(base_path, MIXED_CLASSIFIER_TEST_CSV), index=False)

print(
    f"Created mixed train and test csv files {MIXED_CLASSIFIER_TRAIN_CSV} and {MIXED_CLASSIFIER_TEST_CSV}."
)
