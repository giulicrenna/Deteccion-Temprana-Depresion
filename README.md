# Detección temprana de depresión mediante PLN y aprendizaje automático

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Autores:** Giuliano Crenna, Bruno Emmanuel Pace
**Institución:** Universidad del gran Rosario (UGR)
**Carrera:** Licenciatura Ciencias de Datos
**Director:** (a confirmar)

---

## ¿Qué es este repo?

Es el esqueleto reproducible para la tesis de detección temprana de
depresión en texto en español, usando Procesamiento de Lenguaje Natural
(PLN) y aprendizaje automático. Abarca las **etapas 1-3** del cronograma
(descarga → corpus → EDA) y deja plantada la base para las etapas
4-6 (baseline, BETO, explicabilidad).

## Motivación

La depresión es una de las principales causas de discapacidad a nivel
mundial (WHO 2023). Detectarla temprano a partir de marcadores en el
lenguaje (1ra persona singular, vocabulario absolutista, polaridad
negativa) podría ayudar a canalizar pacientes a intervención
profesional. Este trabajo se enfoca en español rioplatense/peninsular
— un idioma con pocos recursos comparados con inglés.

## Quickstart

```bash
# 1) Setup (crea venv, instala deps, pre-commit)
make setup

# 2) Descargar + limpiar + mergear + splitear
make data

# 3) (Opcional) Traducir al español los corpus en inglés — ver sección "Traducción"
make translate-estimate
make translate SOURCE=kaggle_sdd BACKEND=nvidia LIMIT=30000
make data   # re-mergea incluyendo lo traducido

# 4) Ejecutar los 4 notebooks de EDA (con papermill)
make eda

# 5) Correr tests
make test
```

> **No requiere tokens ni API keys para descargar datos públicos.** Si
> una fuente está gated, los scripts de descarga levantan
> `NotImplementedError` con instrucciones de a quién pedirle acceso
> (`make data` sigue con el resto).

> **GPU:** `requirements-torch.txt` usa CUDA 12.8 (torch 2.7.1), necesario
> para GPUs Blackwell (RTX 50xx / RTX PRO Blackwell) y compatible con
> RTX 30xx/40xx.

## Estructura del repo

```
.
├── data/                        # crudo (raw), interim (anonimizado), processed (corpus)
│   ├── raw/                     # ← no se sube a git
│   ├── interim/                 # ← no se sube a git
│   ├── processed/               # ← corpus final + splits
│   └── DATA_CARD.md             # trazabilidad ética
├── notebooks/
│   ├── 01_eda/                  # 4 notebooks de EDA
│   ├── 02_preprocessing/        # 1 notebook de tokenización
│   ├── 03_baseline/             # stubs para etapa 4
│   ├── 04_beto/                 # stubs para etapa 4
│   └── 05_xai/                  # stubs para etapa 6
├── src/
│   ├── data/
│   │   ├── download/            # scripts de descarga de corpus
│   │   ├── make_dataset.py      # raw → interim (limpieza + anonimización)
│   │   ├── merge_corpora.py     # interim → processed (esquema unificado)
│   │   └── build_splits.py      # user-level split 70/10/20
│   ├── translation/             # traducción EN→ES (MarianMT/NLLB local o API LLM)
│   ├── features/                # LIWC, temporales, polaridad
│   ├── models/                  # stubs para etapa 4
│   ├── evaluation/              # métricas
│   ├── xai/                     # stubs para etapa 6
│   └── utils/                   # seeds, logging
├── configs/                     # YAMLs desacoplados del código
├── tests/                       # pytest
├── reports/figures/             # PNGs exportados de los notebooks
├── references/                  # PDFs y papers de referencia
├── Makefile                     # entry points reproducibles
├── requirements.txt             # runtime core (sin torch)
├── requirements-torch.txt       # torch separado (CUDA 12.1)
├── requirements-dev.txt         # linters, tests, jupyter
├── environment.yml              # conda equivalent
└── README.md (este archivo)
```

## Datasets

| Nombre | Fuente | Licencia | Idioma | Posts | Acceso | Estado |
|---|---|---|---|---|---|---|
| Coello-Guilarte 2019 | INAOE | Research use | es | 1.047.194 tweets | público | ✅ incluido |
| Suicide and Depression Detection **v13** | Kaggle (Komati) | CC BY-SA 4.0 | en → traducir | 348.103 | público | ✅ descargado |
| Reddit Mental Health Posts | HF `solomonk/...` | sin declarar | en → traducir | 151.274 | público | ✅ descargado |
| Depression Reddit Cleaned | HF `mrjunos/...` | CC BY 4.0 | en → traducir | 7.731 | público | ✅ descargado |
| SWMH (Ji et al. 2021) | HF `AIMH/SWMH` | CC BY-NC 4.0 | en → traducir | ~54k | gated (auto) | ⚠️ falta login HF |
| ReDSM5 | HF `irlab-udc/redsm5` | Apache-2.0 | en → traducir | 1.484 | gated (manual) | ⚠️ pedir acceso |
| PrevenIA spanish-suicide-intent | HF | CC BY 4.0 | es | 189.064 | público | ✅ descargado (no se mergea: constructo suicidio) |
| Mini-corpus sintético | local | Generated | es | 60 | — | ✅ dev only |
| MentalRiskES (muestra / completo) | UJA | Gated | es | 45k | formulario | ⚠️ stub |
| Leis 2019 | F. Ronzano | Gated | es | — | mail | ⚠️ stub |
| DAIC-WOZ | USC ICT | DUA | en | — | DUA | ⚠️ stub |
| EmoEvalEs / SWMH-ES | — | — | es | — | el ID de HF no existe | ❌ desactivados |
| RSDD | — | N/AV | — | — | — | ❌ N/AV |

La v13 de Kaggle separa `depression` / `SuicideWatch` / `teenagers`
(la v14 los fusiona en `suicide`). Ver `data/DATA_CARD.md` para el mapeo
de etiquetas y el detalle completo.

## Traducción EN → ES

Los corpus en inglés quedan en `data/interim/<fuente>/data.parquet`
(`lang=en`, ya anonimizados). `src/translation/translate.py` genera
`data_es.parquet` al lado; `merge_corpora` sólo incluye una fuente en
inglés cuando existe su traducción, deja el español en `text_clean` y
conserva el original en `text_orig` / `lang_orig` (+ `is_translated`,
`mt_system`).

Backends (`configs/translation.yaml`):

| Backend | Tipo | Notas |
|---|---|---|
| `opus_mt` | local GPU (MarianMT) | rápido (~4,2 posts/s medido en RTX 3060 Ti 8GB, batch=32, num_beams=4), calidad floja con jerga |
| `nllb` | local GPU (NLLB-200 600M) | licencia CC-BY-NC |
| `nvidia` | API NVIDIA NIM (Nemotron) | free tier ~40 RPM; key en `.env` (`NVIDIA_API_KEY`) |
| `ollama` | LLM local vía Ollama | cualquier servidor OpenAI-compatible sirve (vLLM, OpenRouter...) |

```bash
cp .env.example .env                    # completar NVIDIA_API_KEY
python -m src.translation.translate list-models --backend nvidia   # copiar el id a translation.yaml
make translate-estimate                 # volumen a traducir
make translate-pilot SOURCE=kaggle_sdd BACKEND=nvidia               # 200 docs → pilot_nvidia.parquet
make translate SOURCE=kaggle_sdd BACKEND=nvidia LIMIT=30000         # muestra estratificada
```

- Reanudable: caché en `data/interim/<fuente>/translations/<sistema>.jsonl`.
- `LIMIT` toma una muestra estratificada por label con seed fija.
- Cada traducción trae `qc_flags` (posible rechazo/nota agregada por el
  LLM, ratio de longitud anómalo, restos de inglés) para revisión manual.
- Se traduce el texto anonimizado: ningún handle/URL/email sale a la API.

## Pipeline multi-corpus (Etapa 2b)

Pipeline end-to-end para unificar **Coello-Guilarte + 3 corpus en inglés
traducidos** (`kaggle_sdd`, `reddit_mh_posts`, `depression_reddit`) y
re-entrenar BETO sobre el corpus multi. **Todos los pasos se ejecutan
desde notebooks** en `notebooks/06_translate/`:

| # | Notebook | Qué hace | Tiempo |
|---|---|---|---|
| 1 | [01_project_user_csvs.ipynb](notebooks/06_translate/01_project_user_csvs.ipynb) | Anonimiza los 3 CSV crudos → `data/interim/<fuente>/data.parquet` | ~2 min |
| 2 | [02_translate_multi.ipynb](notebooks/06_translate/02_translate_multi.ipynb) | MarianMT EN→ES con GPU, cache append-only | ~7h |
| 3 | [03_merge_multi.ipynb](notebooks/06_translate/03_merge_multi.ipynb) | Inner-join corpus + traducciones → `corpus_v1.parquet` | ~5 min |
| 4 | [04_build_splits_multi.ipynb](notebooks/06_translate/04_build_splits_multi.ipynb) | Splits user-level 70/10/20 → `splits_multi/` | ~2 min |
| 5 | [04_beto/03_finetune_beto_multi.ipynb](notebooks/04_beto/03_finetune_beto_multi.ipynb) | Re-entrena BETO multi (~50 min GPU) + eval batched | ~50 min |

> **Pre-requisito del paso 2**: el notebook usa el Python del sistema
> con CUDA (`C:\Users\giuli\AppData\Local\Programs\Python\Python312\python.exe`),
> no el `.venv`.

### Estimaciones de traducción (RTX 3060 Ti 8GB)

Piloto con 200 docs estratificados: **4.20 docs/s**.

- **kaggle_sdd**: 80k docs (default) → ~5h · 232k completos → ~15h
- **reddit_mh_posts**: 24k (completo, todos label=2) → ~1.6h
- **depression_reddit**: 7.7k (completo, 50/50) → ~0.5h
- **Total default**: ~7h para los 3 corpus.

> `reddit_mh_posts` solo aporta positivos (filtrado a `r/depression`
> puro). Si querés recortar para no desbalancear, agregá `--limit N`
> en la celda correspondiente de `02_translate_multi.ipynb`.

### Leaderboard final (test)

| Modelo | Feature set | F1-macro | AUC depressive | F1 depressive |
|---|---|---|---|---|
| `logreg_full_balanced` | tfidf | ~0,655 | ~0,755 | … |
| `logreg_downsampled` | tfidf | … | … | … |
| `logreg_combined` | tfidf+handcrafted | ~0,644 | … | … |
| `beto` (mono) | beto_transformer | ~0,717 | … | … |
| `beto_multi` (multi) | beto_transformer_multi | TBD | TBD | TBD |

> Los números del mono ya están en `reports/tables/all_models_metrics.csv`.
> El multi se appendea al ejecutar `notebooks/04_beto/03_finetune_beto_multi.ipynb`.

## Ética

- **Solo datos públicos** (Twitter, Reddit-like).
- **Anonimización previa** al análisis (URLs, menciones, emails,
  teléfonos, hashtags). Documentado en `src/data/make_dataset.py`.
- **No se suben datos crudos a git** (`data/raw/`, `interim/`,
  `processed/` en `.gitignore`).
- Limitaciones reconocidas: el NER para nombres propios no se aplica en
  v1 (riesgo de falsos negativos). Ver `DATA_CARD.md` sección
  "Anonimización / Limitaciones".
- El modelo final **no debe usarse como screening clínico unilateral**
  (sección 8 de la tesis lo discute).

## Roadmap

| Etapa | Qué | Estado |
|---|---|---|
| 1 | Setup del repo + descargas | ✅ v0.1 |
| 2 | Corpus unificado + anonimización | ✅ v0.1 |
| 2b | Traducción EN→ES de corpus Reddit | 🔧 pipeline listo, falta correr |
| 3 | EDA (4 notebooks) | ✅ v0.1 |
| 4 | Baseline (LogReg) + BETO fine-tuning | ⏳ próximo |
| 5 | Evaluación comparativa + métricas | ⏳ |
| 6 | XAI (SHAP, LIME, attention) | ⏳ |

## Cómo contribuir

1. Fork + branch con prefijo `feat/`, `fix/`, `docs/`.
2. Antes de commitear, corré `make format` (black + ruff + isort).
3. Abrí un PR con descripción clara del cambio.
4. El CI local (pre-commit) valida formato, linting y que no haya
   secretos commiteados.

**Código de conducta:** respeto y profesionalismo. No se tolera acoso
de ningún tipo.

## Cita sugerida (BibTeX)

```bibtex
@thesis{crenna_pace_2026,
  author = {Crenna, Giuliano and Pace, Bruno Emmanuel},
  title  = {Detección temprana de depresión mediante PLN y aprendizaje automático},
  school = {Universidad de Granada},
  year   = {2026},
  type   = {{Tesis de grado}}
}
```

## Licencia

MIT — ver [LICENSE](LICENSE).
