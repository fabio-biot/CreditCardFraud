from pathlib import Path

import pandas as pd


DATASET_PATH = Path(__file__).resolve().parent / "creditcard.csv"


def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            "Dataset introuvable. Lancez `make data` pour telecharger "
            f"creditcard.csv dans {DATASET_PATH}."
        )

    return pd.read_csv(DATASET_PATH)


def load_fraud_dataset():
    dataset = load_dataset()
    fraud_dataset = dataset[dataset['Class'] == 1]
    return fraud_dataset


def load_non_fraud_dataset():
    dataset = load_dataset()
    non_fraud_dataset = dataset[dataset['Class'] == 0]
    return non_fraud_dataset
