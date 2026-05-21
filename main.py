import pandas as pd
from src.preprocessing.preprocessing import get_best_model
from src.data.loading_data import load_dataset


def main():
    y_result = 0
    best_model, threshold = get_best_model()
    dataset = load_dataset()
    columns = dataset.drop("Class", axis=1).columns
    x_topredict = pd.DataFrame([[
        0, -1.3, -0.08, 2.53,
        -1, -0.03, 0.3, 0.2, 0.01,
        0.4, 0.15, -0.551599533, -0.05,
        -1, -0.3, 1.5, -0.4, 0.5, 0.02579058,
        0.40399296, 0.251412098, -0.018306778,
        0.277837576, -0.11047391, 0.066928075, 0.2,
        -0.189114844, 0.133558377, -0.021053053, 149.62
    ]], columns=columns)

    data = pd.DataFrame([[
        472, -3.043540624, -3.157307121, 1.08846278, 2.288643618,
        1.35980513, -1.064822523, 0.325574266, -0.067793653,
        -0.270952836, -0.838586565, -0.414575448, -0.50314086,
        0.676501545, -1.692028933, 2.000634839, 0.666779696,
        0.599717414, 1.725321007, 0.28334483, 2.102338793,
        0.661695925, 0.435477209, 1.375965743, -0.293803153,
        0.279798032, -0.145361715, -0.252773123, 0.035764225, 529
    ]],  columns=columns)

    y_proba = best_model.predict_proba(x_topredict)[0][1]
    y_pred = int(y_proba > threshold)
    print(
        f"Prediction : {y_pred}, proba fraude : {y_proba:.4f}, "
        f"valeur réelle : {y_result}"
    )

    print("=" * 100)
    y_result = 1
    y_proba = best_model.predict_proba(data)[0][1]
    y_pred = int(y_proba > threshold)
    print(
        f"Prediction : {y_pred}, proba fraude : {y_proba:.4f}, "
        f"valeur réelle : {y_result}"
    )


if __name__ == "__main__":
    main()
