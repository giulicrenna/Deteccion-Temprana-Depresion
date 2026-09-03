# Detección temprana de depresión mediante PLN y aprendizaje automático

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Autores:** Giuliano Crenna, Juan Ignacio Pace
**Institución:** Universidad de Granada (UGR)
**Carrera:** Ingeniería en Tecnología / Ciencias de Datos
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

# 3) Ejecutar los 4 notebooks de EDA (con papermill)
make eda

# 4) Correr tests
make test
```

> **No requiere tokens ni API keys para descargar datos.** Si una
> fuente está gated, los scripts de descarga levantan
> `NotImplementedError` con instrucciones de a quién pedirle acceso.

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

| Nombre | Fuente | Licencia | Idioma | Wget-able | Estado |
|---|---|---|---|---|---|
| Coello-Guilarte 2019 | INAOE | Research use | es | sí | ✅ incluido |
| MentalRiskES (muestra) | GitHub UJA | Gated | es | zip cifrado | ⚠️ stub |
| ReDSM5 paraphrase | HF Hub | MIT | en/es | `datasets` lib | ✅ incluido |
| EmoEvalEs | HF Hub | Research use | es | `datasets` lib | ✅ best-effort |
| SWMH-ES | HF Hub | Mixed | es | `datasets` lib | ✅ best-effort |
| Mini-corpus sintético | local | Generated | es | sí | ✅ dev only |
| MentalRiskES completo | autores | Gated | es | no | ⚠️ stub |
| Leis 2019 | Kaggle / F. Ronzano | Gated | es | no | ⚠️ stub |
| DAIC-WOZ | USC ICT | DUA | en | no | ⚠️ stub |
| RSDD | — | N/AV | es | — | ❌ N/AV |

Ver `data/DATA_CARD.md` para el detalle completo.

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
  author = {Crenna, Giuliano and Pace, Juan Ignacio},
  title  = {Detección temprana de depresión mediante PLN y aprendizaje automático},
  school = {Universidad de Granada},
  year   = {2026},
  type   = {{Tesis de grado}}
}
```

## Licencia

MIT — ver [LICENSE](LICENSE).
