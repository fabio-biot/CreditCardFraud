"""Tests pour le module de preprocessing."""
import unittest
from pathlib import Path
import shutil

from src.preprocessing.preprocessing import get_best_model, save_model, load_model, MODEL_PATH, THRESHOLD_PATH


class TestPreprocessing(unittest.TestCase):
    """Test cases pour preprocessing.py"""

    def setUp(self):
        """Vérifie que le dataset existe avant chaque test."""
        self.dataset_path = Path(__file__).resolve().parent.parent / "src" / "data" / "creditcard.csv"
        if not self.dataset_path.exists():
            self.skipTest(f"Dataset introuvable : {self.dataset_path}")

    def tearDown(self):
        """Nettoie les fichiers de modèle créés pendant les tests."""
        models_dir = MODEL_PATH.parent
        if models_dir.exists():
            shutil.rmtree(models_dir)

    def test_get_best_model_returns_model_and_threshold(self):
        """Test que get_best_model retourne un modèle et un threshold."""
        model, threshold = get_best_model(cv=2)  # cv=2 pour plus rapide
        self.assertIsNotNone(model)
        self.assertIsInstance(threshold, float)
        self.assertGreaterEqual(threshold, 0.0)
        self.assertLessEqual(threshold, 1.0)

    def test_get_best_model_saves_files(self):
        """Test que get_best_model sauvegarde les fichiers."""
        get_best_model(cv=2)
        self.assertTrue(MODEL_PATH.exists())
        self.assertTrue(THRESHOLD_PATH.exists())

    def test_save_and_load_model(self):
        """Test que save_model et load_model fonctionnent ensemble."""
        from xgboost import XGBClassifier
        from sklearn.pipeline import Pipeline
        
        model = Pipeline([("model", XGBClassifier())])
        threshold = 0.5
        
        save_model(model, threshold)
        loaded_model, loaded_threshold = load_model()
        
        self.assertEqual(loaded_threshold, threshold)

    def test_load_model_raises_error_when_files_missing(self):
        """Test que load_model lève une erreur quand les fichiers n'existent pas."""
        models_dir = MODEL_PATH.parent
        if models_dir.exists():
            shutil.rmtree(models_dir)
        
        with self.assertRaises(FileNotFoundError):
            load_model()

    def test_get_best_model_uses_saved_model(self):
        """Test que get_best_model utilise le modèle sauvegardé si disponible."""
        # D'abord entraîner et sauvegarder
        model1, threshold1 = get_best_model(cv=2, force_train=True)
        
        # Ensuite appeler sans force_train
        model2, threshold2 = get_best_model(cv=2, force_train=False)
        
        # Devrait charger depuis le disque
        self.assertEqual(threshold1, threshold2)


if __name__ == '__main__':
    unittest.main()