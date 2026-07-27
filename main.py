import pandas as pd
from src.preprocessing.preprocessing import get_best_model
from src.data.loading_data import load_dataset


def get_sample_transactions(dataset, n_fraud: int = 1, n_normal: int = 1):
    """
    Sélectionne des échantillons aléatoires de transactions fraudeuses et normales.
    
    Args:
        dataset: DataFrame contenant le dataset
        n_fraud: nombre de transactions fraudeuses à sélectionner
        n_normal: nombre de transactions normales à sélectionner
    
    Returns:
        list: liste de tuples (transaction_features, true_label)
    """
    samples = []
    
    fraud_samples = dataset[dataset["Class"] == 1].sample(n=min(n_fraud, len(dataset[dataset["Class"] == 1])))
    for _, row in fraud_samples.iterrows():
        features = row.drop("Class").values.reshape(1, -1)
        samples.append((features, 1))
    
    normal_samples = dataset[dataset["Class"] == 0].sample(n=min(n_normal, len(dataset[dataset["Class"] == 0])))
    for _, row in normal_samples.iterrows():
        features = row.drop("Class").values.reshape(1, -1)
        samples.append((features, 0))
    
    return samples


def main():
    best_model, threshold = get_best_model()
    dataset = load_dataset()
    columns = dataset.drop("Class", axis=1).columns
    
    # Sélectionner des exemples réels du dataset
    samples = get_sample_transactions(dataset, n_fraud=1, n_normal=1)
    
    print("=" * 100)
    print("Exemples de prédictions sur des transactions réelles du dataset :")
    print("=" * 100)
    
    for i, (features, true_label) in enumerate(samples):
        x_to_predict = pd.DataFrame(features, columns=columns)
        y_proba = best_model.predict_proba(x_to_predict)[0][1]
        y_pred = int(y_proba > threshold)
        
        label_str = "FRAUDE" if true_label == 1 else "NORMALE"
        pred_str = "FRAUDE" if y_pred == 1 else "NORMALE"
        
        print(f"\nExemple {i + 1} :")
        print(f"  Type réel : {label_str}")
        print(f"  Prédiction : {pred_str}")
        print(f"  Probabilité de fraude : {y_proba:.4f}")
        print(f"  Seuil : {threshold:.4f}")
        print(f"  Correct : {'OUI' if y_pred == true_label else 'NON'}")


if __name__ == "__main__":
    main()
