PYTHON ?= python3
VENV ?= .venv
BIN := $(VENV)/bin
PIP := $(BIN)/pip
PY := $(BIN)/python
INSTALL_STAMP := $(VENV)/.installed
PYTHONPYCACHEPREFIX ?= .pycache

.PHONY: help setup install data run notebook lint check clean

help:
	@echo "Commandes disponibles:"
	@echo "  make setup     Cree l'environnement virtuel et installe les dependances"
	@echo "  make install   Installe les dependances dans .venv"
	@echo "  make data      Telecharge le dataset Kaggle dans src/data/"
	@echo "  make run       Lance la demo de prediction"
	@echo "  make notebook  Lance Jupyter Notebook"
	@echo "  make lint      Lance flake8"
	@echo "  make check     Verifie la syntaxe Python"
	@echo "  make clean     Supprime les caches et artefacts locaux"

setup: $(INSTALL_STAMP)

$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)

$(INSTALL_STAMP): requirements.txt $(VENV)/bin/activate
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	touch $(INSTALL_STAMP)

install: $(INSTALL_STAMP)

data: $(INSTALL_STAMP)
	$(PY) config/dataset_config.py

run: $(INSTALL_STAMP)
	$(PY) main.py

notebook: $(INSTALL_STAMP)
	$(PY) -m notebook

lint: $(INSTALL_STAMP)
	$(PY) -m flake8 main.py loading.py config src

check:
	PYTHONPYCACHEPREFIX=$(PYTHONPYCACHEPREFIX) $(PYTHON) -m compileall main.py loading.py config src

clean:
	rm -rf .pycache
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -f matrix_analysis.png
