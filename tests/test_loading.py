"""Tests pour le module de chargement des données."""
import unittest
from pathlib import Path
import pandas as pd

from src.data.loading_data import load_dataset, load_fraud_dataset, load_non_fraud_dataset


class TestLoadingData(unittest.TestCase):
    """Test cases pour loading_data.py"""

    def setUp(self):
        """Vérifie que le dataset existe avant chaque test."""
        self.dataset_path = Path(__file__).resolve().parent.parent / "src" / "data" / "creditcard.csv"
        if not self.dataset_path.exists():
            self.skipTest(f"Dataset introuvable : {self.dataset_path}")

    def test_load_dataset_returns_dataframe(self):
        """Test que load_dataset retourne un DataFrame."""
        dataset = load_dataset()
        self.assertIsInstance(dataset, pd.DataFrame)

    def test_load_dataset_has_correct_columns(self):
        """Test que le dataset a les bonnes colonnes."""
        dataset = load_dataset()
        expected_columns = [
            'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8',
            'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16',
            'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24',
            'V25', 'V26', 'V27', 'V28', 'Amount', 'Class'
        ]
        self.assertListEqual(list(dataset.columns), expected_columns)

    def test_load_dataset_has_class_column(self):
        """Test que le dataset contient la colonne Class."""
        dataset = load_dataset()
        self.assertIn('Class', dataset.columns)

    def test_load_dataset_class_values(self):
        """Test que la colonne Class contient seulement 0 et 1."""
        dataset = load_dataset()
        unique_values = dataset['Class'].unique()
        self.assertListEqual(list(unique_values), [0, 1])

    def test_load_fraud_dataset_returns_only_fraud(self):
        """Test que load_fraud_dataset retourne seulement les fraudes."""
        fraud_dataset = load_fraud_dataset()
        self.assertTrue((fraud_dataset['Class'] == 1).all())

    def test_load_non_fraud_dataset_returns_only_normal(self):
        """Test que load_non_fraud_dataset retourne seulement les transactions normales."""
        non_fraud_dataset = load_non_fraud_dataset()
        self.assertTrue((non_fraud_dataset['Class'] == 0).all())


if __name__ == '__main__':
    unittest.main()