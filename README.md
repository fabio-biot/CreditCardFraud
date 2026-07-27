# CardFraud

Portfolio de detection de fraude bancaire a partir du dataset public Kaggle
`mlg-ulb/creditcardfraud`. Le projet montre un workflow de data science simple :
exploration des donnees, preparation, entrainement d'un modele XGBoost et prediction
sur quelques transactions d'exemple.

## Objectif

Le dataset est fortement desequilibre : la classe fraude est rare. Le pipeline entraine
un classifieur XGBoost avec `scale_pos_weight`, puis choisit un seuil de decision en
fonction d'un rappel minimal sur la classe fraude.

## Structure

```text
.
|-- config/
|   `-- dataset_config.py        # Telechargement Kaggle du dataset
|-- src/
|   |-- data/loading_data.py     # Chargement du CSV local
|   `-- preprocessing/
|       `-- preprocessing.py     # Entrainement et selection du seuil
|-- tests/                       # Tests unitaires
|   |-- test_loading.py
|   `-- test_preprocessing.py
|-- EDA.ipynb                    # Analyse exploratoire
|-- model.ipynb                  # Experimentations modele
|-- main.py                      # Demo de prediction
|-- requirements.txt             # Dependances Python
`-- Makefile                     # Commandes utiles
```

## Installation

Pre-requis :

- Python 3.9 ou plus recent
- Un compte Kaggle configure si vous utilisez `make data`

```bash
make setup
```

La commande cree un environnement virtuel `.venv` et installe les dependances depuis
`requirements.txt`.

## Donnees

Le fichier `creditcard.csv` n'est pas versionne dans Git. Pour le telecharger depuis
Kaggle et le placer dans `src/data/creditcard.csv` :

```bash
make data
```

Si Kaggle n'est pas encore configure, placez votre fichier `kaggle.json` selon la
documentation Kaggle, ou ajoutez manuellement `creditcard.csv` dans `src/data/`.

## Utilisation

Lancer la demonstration de prediction :

```bash
make run
```

Ouvrir Jupyter pour explorer les notebooks :

```bash
make notebook
```

Verifier rapidement la syntaxe Python :

```bash
make check
```

## Commandes Makefile

```bash
make help       # Affiche les commandes disponibles
make setup      # Cree .venv et installe les dependances
make install    # Installe ou reinstalle les dependances
make data       # Telecharge le dataset Kaggle
make run        # Lance main.py
make notebook   # Lance Jupyter Notebook
make lint       # Lance flake8
make check      # Compile les fichiers Python
make test       # Lance les tests unitaires
make clean      # Supprime caches Python et artefacts locaux
```

## Resultat attendu

`main.py` entraine le modele (ou charge depuis le disque), calcule un seuil conseille,
affiche une matrice de confusion sur le jeu de test, puis affiche des predictions sur
des exemples réels du dataset avec leur classe réelle.

Le modèle est automatiquement sauvegardé dans `src/models/` et réutilisé aux
prochaines exécutions pour éviter un réentraînement (utilisez `force_train=True`
pour forcer le réentraînement).

## Améliorations implémentées

- ✅ Sauvegarde du modèle entraîné avec `joblib` dans `src/models/`
- ✅ Chargement automatique du modèle existant pour éviter le réentraînement
- ✅ Tests unitaires sur le chargement de données et le preprocessing
- ✅ Utilisation d'exemples réels du dataset dans main.py

## Pistes d'amelioration futures

- Exposer une petite API ou interface web pour présenter le projet en portfolio.
- Ajouter plus de métriques et visualisations dans les notebooks.
- Optimiser les hyperparamètres du modèle XGBoost.
