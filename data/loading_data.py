import pandas as pd
import kagglehub


def load_dataset():
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    dataset = pd.read_csv(f"{path}/creditcard.csv")
    return dataset


def load_fraud_dataset():
    dataset = load_dataset()
    fraud_dataset = dataset[dataset['Class'] == 1]
    return fraud_dataset


def load_non_fraud_dataset():
    dataset = load_dataset()
    non_fraud_dataset = dataset[dataset['Class'] == 0]
    return non_fraud_dataset
