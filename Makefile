PYTHON ?= /usr/bin/python3
VENV ?= .venv
BIN := $(VENV)/bin
PIP := $(BIN)/pip
PY := $(BIN)/python
INSTALL_STAMP := $(VENV)/.installed
PYTHONPYCACHEPREFIX ?= .pycache

.PHONY: help setup install data run notebook lint check test clean

help:
	@echo "Commandes disponibles:"
	@echo "  make setup     Cree l'environnement virtuel et installe les dependances"
	@echo "  make install   Installe les dependances dans .venv"
	@echo "  make data      Telecharge le dataset Kaggle dans src/data/"
	@echo "  make run       Lance la demo de prediction"
	@echo "  make notebook  Lance Jupyter Notebook"
	@echo "  make lint      Lance flake8"
	@echo "  make check     Verifie la syntaxe Python"
	@echo "  make test      Lance les tests unitaires"
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

run:
	@if [ -f $(INSTALL_STAMP) ] || [ -d $(VENV)/bin ]; then $(PY) main.py; else echo "Venv non initialisé. Lancez 'make setup' d'abord."; exit 1; fi

notebook:
	@if [ -f $(INSTALL_STAMP) ] || [ -d $(VENV)/bin ]; then $(PY) -m notebook; else echo "Venv non initialisé. Lancez 'make setup' d'abord."; exit 1; fi

lint:
	@if [ -f $(INSTALL_STAMP) ] || [ -d $(VENV)/bin ]; then $(PY) -m flake8 main.py config src tests; else echo "Venv non initialisé. Lancez 'make setup' d'abord."; exit 1; fi

check:
	@if [ -f $(INSTALL_STAMP) ] || [ -d $(VENV)/bin ]; then PYTHONPYCACHEPREFIX=$(PYTHONPYCACHEPREFIX) $(PY) -m compileall main.py config src tests; else echo "Venv non initialisé. Lancez 'make setup' d'abord."; exit 1; fi

test:
	@if [ -f $(INSTALL_STAMP) ] || [ -d $(VENV)/bin ]; then $(PY) -m pytest tests -v; else echo "Venv non initialisé. Lancez 'make setup' d'abord."; exit 1; fi

clean:
	rm -rf .pycache
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -f matrix_analysis.png
	find . -type d -name "models" -prune -exec rm -rf {} +
