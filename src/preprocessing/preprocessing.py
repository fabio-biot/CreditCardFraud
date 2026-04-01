from src.data.loading_data import load_dataset, load_fraud_dataset, load_non_fraud_dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.compose import make_column_transformer
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from xgboost import XGBClassifier


def get_best_model(min_recall: float = 0.8, cv: int = 3):
    """
    Entraîne un modèle XGBoost sur le dataset et retourne le meilleur modèle GridSearchCV.
    Paramètres :
        min_recall : float, recall minimal sur la classe fraud pour choisir le threshold
        cv : int, nombre de folds pour GridSearchCV
    Retourne :
        best_model : Pipeline entraîné avec le meilleur estimator
        best_threshold : float, threshold optimal pour prédire la classe 1
    """
    # Chargement des données
    dataset = load_dataset()
    x_data = dataset.drop("Class", axis=1)
    y_data = dataset["Class"]

    # Split train/test
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.2, random_state=42, stratify=y_data
    )

    # Gérer le déséquilibre
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

    # Modèle XGBoost
    model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42
    )

    pipeline = Pipeline([
        ("model", model)
    ])

    # GridSearchCV
    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [3, 5],
        "model__learning_rate": [0.05, 0.1]
    }

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring="recall",
        n_jobs=-1,
        verbose=1
    )

    grid.fit(x_train, y_train)
    best_model = grid.best_estimator_

    # Probabilités sur test set
    y_proba = best_model.predict_proba(x_test)[:, 1]

    # Trouver le meilleur threshold pour min recall
    best_threshold = 0
    best_precision = 0
    for t in np.arange(0.1, 1.0, 0.01):
        y_pred = (y_proba > t).astype(int)
        report = classification_report(y_test, y_pred, output_dict=True)
        precision_fraude = report["1"]["precision"]
        recall_fraude = report["1"]["recall"]

        if recall_fraude >= min_recall and precision_fraude > best_precision:
            best_precision = precision_fraude
            best_threshold = t

    print(f"Threshold conseillé : {best_threshold:.2f}")
    print(f"Precision : {best_precision:.2f}")
    print("Confusion Matrix sur test set :")
    print(confusion_matrix(y_test, (y_proba > best_threshold).astype(int)))

    return best_model, best_threshold