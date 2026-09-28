# Makefile — entry points reproducibles del repo.
# Convenciones:
#   - Todos los targets son idempotentes: podés correrlos varias veces.
#   - Sin autenticación en ningún script (cero tokens, cero keys).
#   - Verificá `make help` para ver el catálogo completo.

.PHONY: help setup data download translate-estimate translate-pilot translate eda baseline test coverage lint format clean zip all
.DEFAULT_GOAL := help

ifeq ($(OS),Windows_NT)
    PY ?= python
    VENV_BIN := .venv/Scripts
else
    VENV_BIN := .venv/bin
    ifneq ($(wildcard $(VENV_BIN)/python),)
        PY ?= $(VENV_BIN)/python
    else
        PY ?= python3
    endif
endif
PIP ?= $(PY) -m pip
VENV_PY := $(VENV_BIN)/python

DATA_DIR ?= ./data
SEED ?= 42

help:  ## Mostrar este catálogo de comandos.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup:  ## Crear venv, instalar deps runtime + dev + pre-commit.
	@echo ">> creando venv con $(PY)..."
	@$(PY) -m venv .venv
	@$(VENV_PY) -m pip install --upgrade pip && \
		$(VENV_PY) -m pip install -r requirements.txt && \
		$(VENV_PY) -m pip install -r requirements-dev.txt && \
		$(VENV_PY) -m pip install -r requirements-torch.txt
	@$(VENV_PY) -m pre_commit install
	@echo ">> setup OK. Activá con: source $(VENV_BIN)/activate"

download:  ## Descargar todas las fuentes (las gated fallan con instrucciones, sin cortar).
	@echo ">> descargando fuentes públicas..."
	@$(PY) -m src.data.download.download_coello_guilarte --out $(DATA_DIR)/raw/coello_guilarte
	@$(PY) -m src.data.download.download_kaggle_sdd --out $(DATA_DIR)/raw/kaggle_sdd
	@$(PY) -m src.data.download.download_reddit_mh_posts --out $(DATA_DIR)/raw/reddit_mh_posts
	@$(PY) -m src.data.download.download_depression_reddit --out $(DATA_DIR)/raw/depression_reddit
	@$(PY) -m src.data.download.download_prevenia_es --out $(DATA_DIR)/raw/prevenia_es
	@$(PY) -m src.data.download.download_synthetic --out $(DATA_DIR)/raw/synthetic
	@echo ">> fuentes gated (requieren acceso previo, ver DATA_CARD)..."
	-@$(PY) -m src.data.download.download_swmh --out $(DATA_DIR)/raw/swmh
	-@$(PY) -m src.data.download.download_redsm5_sample --out $(DATA_DIR)/raw/redsm5_sample
	-@$(PY) -m src.data.download.download_mentalriskes_github --out $(DATA_DIR)/raw/mentalriskes_github

data: download  ## Descargar + limpiar + mergear + splitear todos los corpus.
	@echo ">> [2/4] limpiando + anonimizando → data/interim/..."
	@$(PY) -m src.data.make_dataset --out $(DATA_DIR)/interim --seed $(SEED)
	@echo ">> [3/4] mergeando corpus → data/processed/corpus_v1.parquet..."
	@$(PY) -m src.data.merge_corpora --in $(DATA_DIR)/interim --out $(DATA_DIR)/processed
	@echo ">> [4/4] split user-level estratificado 70/10/20..."
	@$(PY) -m src.data.build_splits --in $(DATA_DIR)/processed --out $(DATA_DIR)/processed/splits --seed $(SEED)
	@echo ">> data OK."

# --- Traducción EN→ES (ver configs/translation.yaml y README) ---
SOURCE ?= kaggle_sdd
BACKEND ?= nvidia
LIMIT ?= 0

translate-estimate:  ## Volumen (docs/caracteres/tokens) de los corpus a traducir.
	@$(PY) -m src.translation.translate estimate

translate-pilot:  ## Piloto de 200 docs: make translate-pilot SOURCE=kaggle_sdd BACKEND=nvidia
	@$(PY) -m src.translation.translate run --source $(SOURCE) --backend $(BACKEND) \
		--limit 200 --out-name pilot_$(BACKEND).parquet

translate:  ## Traducir una fuente: make translate SOURCE=kaggle_sdd BACKEND=nvidia LIMIT=30000
	@$(PY) -m src.translation.translate run --source $(SOURCE) --backend $(BACKEND) --limit $(LIMIT)

eda:  ## Ejecutar notebooks 01-04 con papermill (outputs en reports/).
	@mkdir -p reports/figures reports/tables
	@$(PY) -m papermill notebooks/01_eda/01_exploracion_inicial.ipynb \
		reports/eda_01_exploracion_inicial.ipynb \
		-p DATA_DIR $(DATA_DIR) -p SEED $(SEED) -p OUT_DIR reports
	@$(PY) -m papermill notebooks/01_eda/02_distribuciones_y_balance.ipynb \
		reports/eda_02_distribuciones_y_balance.ipynb \
		-p DATA_DIR $(DATA_DIR) -p SEED $(SEED) -p OUT_DIR reports
	@$(PY) -m papermill notebooks/01_eda/03_marcadores_linguisticos.ipynb \
		reports/eda_03_marcadores_linguisticos.ipynb \
		-p DATA_DIR $(DATA_DIR) -p SEED $(SEED) -p OUT_DIR reports
	@$(PY) -m papermill notebooks/01_eda/04_comparacion_entre_corpora.ipynb \
		reports/eda_04_comparacion_entre_corpora.ipynb \
		-p DATA_DIR $(DATA_DIR) -p SEED $(SEED) -p OUT_DIR reports
	@echo ">> eda OK. Outputs en reports/."

baseline:  ## Entrenar baseline clásico (TF-IDF + LogReg y SVM).
	@$(PY) -m src.models.train_baseline --config configs/models.yaml --data-dir $(DATA_DIR)/processed/splits --out-dir models --reports-dir reports/tables

coverage:  ## Correr pytest con reporte de cobertura sobre src/models y src/evaluation.
	@$(PY) -m pytest tests/ --cov=src/models --cov=src/evaluation --cov-report=term-missing -v

test:  ## Correr pytest.
	@$(PY) -m pytest tests/ -v --tb=short

lint:  ## ruff + black --check.
	@$(PY) -m ruff check src/ tests/ --line-length 100
	@$(PY) -m black --check --line-length 100 src/ tests/

format:  ## auto-fix con ruff + black + isort.
	@$(PY) -m ruff check --fix --line-length 100 src/ tests/
	@$(PY) -m black --line-length 100 src/ tests/
	@$(PY) -m isort --profile black --line-length 100 src/ tests/

clean:  ## Borrar caches, venv, manifests; conserva data/.
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf .pytest_cache .ruff_cache .mypy_cache mlruns mlruns.db
	@echo ">> clean OK."

zip:  ## Empaquetar el repo en /workspace/repo_tesis_depresion.zip.
	@cd $(CURDIR) && \
		zip -r /workspace/repo_tesis_depresion.zip . \
			-x '*/__pycache__/*' \
			-x '*/.ipynb_checkpoints/*' \
			-x '*/.git/*' \
			-x '*/.pytest_cache/*' \
			-x '*/.ruff_cache/*' \
			-x '*/.mypy_cache/*' \
			-x '*/__pycache__/*' \
			-x 'data/raw/*' \
			-x 'data/interim/*' \
			-x 'data/processed/*' \
			-x '*.pyc' \
			-x '*.egg-info/*'
	@echo ">> zip OK."
	@ls -lh /workspace/repo_tesis_depresion.zip

all: setup data eda test lint  ## Pipeline completo.
	@echo ">> all OK."
