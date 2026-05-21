from pathlib import Path
from shutil import copy2

import kagglehub


DATASET = "mlg-ulb/creditcardfraud"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DESTINATION = PROJECT_ROOT / "src" / "data" / "creditcard.csv"


def download_dataset() -> Path:
    dataset_dir = Path(kagglehub.dataset_download(DATASET))
    source = dataset_dir / "creditcard.csv"

    if not source.exists():
        raise FileNotFoundError(
            f"Fichier creditcard.csv introuvable dans {dataset_dir}"
        )

    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    copy2(source, DESTINATION)
    return DESTINATION


if __name__ == "__main__":
    destination = download_dataset()
    print(f"Dataset disponible : {destination}")
