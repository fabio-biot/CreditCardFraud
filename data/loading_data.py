import pandas as pd


def load_dataset():
    return pd.read_csv("data/creditcard.csv")


def load_fraud_dataset():
    dataset = load_dataset()
    fraud_dataset = dataset[dataset['Class'] == 1]
    return fraud_dataset


def load_non_fraud_dataset():
    dataset = load_dataset()
    non_fraud_dataset = dataset[dataset['Class'] == 0]
    return non_fraud_dataset
