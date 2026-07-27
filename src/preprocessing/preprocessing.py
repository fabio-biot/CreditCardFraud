import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from joblib import dump, load

from src.data.loading_data import load_dataset

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "xgboost_fraud_model.joblib"
THRESHOLD_PATH = Path(__file__).resolve().parent.parent / "models" / "threshold.txt"


def save_model(model, threshold: float):
    """Sauvegarde le modèle et le threshold dans le dossier models."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    dump(model, MODEL_PATH)
    with open(THRESHOLD_PATH, "w") as f:
        f.write(str(threshold))
    print(f"Modèle sauvegardé : {MODEL_PATH}")
    print(f"Threshold sauvegardé : {THRESHOLD_PATH}")


def load_model():
    """Charge le modèle et le threshold depuis le dossier models."""
    if not MODEL_PATH.exists() or not THRESHOLD_PATH.exists():
        raise FileNotFoundError(
            "Modèle ou threshold introuvable. "
            "Entraînez d'abord le modèle avec get_best_model()."
        )
    model = load(MODEL_PATH)
    with open(THRESHOLD_PATH, "r") as f:
        threshold = float(f.read())
    return model, threshold


def get_best_model(min_recall: float = 0.8, cv: int = 3, force_train: bool = False):
    """
    Entraîne un modele XGBoost et retourne le meilleur modele.
    Charge depuis le disque si le modèle existe déjà et force_train=False.

    Paramètres :
        min_recall : recall minimal sur la classe fraud
        cv : int, nombre de folds pour GridSearchCV
        force_train : bool, si True, réentraîne même si le modèle existe

    Retourne :
        best_model : Pipeline entraîné avec le meilleur estimator
        best_threshold : float, threshold optimal pour prédire la classe 1
    """
    # Vérifier si le modèle existe déjà
    if not force_train and MODEL_PATH.exists() and THRESHOLD_PATH.exists():
        print("Modèle existant trouvé. Chargement depuis le disque...")
        return load_model()
    
    print("Entraînement du modèle...")
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

    # Sauvegarder le modèle et le threshold
    save_model(best_model, best_threshold)

    return best_model, best_threshold
