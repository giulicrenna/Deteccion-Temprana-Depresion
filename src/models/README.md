# src/models/

Módulo de modelado (etapa 4 del cronograma de la tesis).

## Estado actual

**Stubs únicamente.** Los scripts de entrenamiento (baseline, BETO) se
implementarán en la etapa 4, una vez que el corpus procesado esté validado
por los notebooks de EDA.

## Plan de implementación

| Archivo | Función | Etapa |
|---|---|---|
| `train_baseline.py` | TF-IDF + Logistic Regression / SVM. Baseline rápido. | 4 |
| `train_beto.py` | Fine-tuning de `dccuchile/bert-base-spanish-wwm-cased` con `transformers.Trainer`. | 4 |
| `predict.py` | Inferencia batch sobre un parquet con el modelo entrenado. | 4 |
| `registry.py` | Versionado + logueo en MLflow local. | 4 |

## Convenciones

- Todos los modelos deben llamar `set_seed(42)` al inicio (de `src/utils/seeds.py`).
- El split usado está en `data/processed/splits/{train,val,test}.parquet`.
- Los artefactos (modelo, tokenizer, métricas) se guardan en `mlruns/`
  vía MLflow (SQLite local, no servidor).
- Reporte final en `reports/tables/` (CSV con métricas por experimento).

## Decisiones pendientes (resolver en etapa 4)

1. ¿Grid search vs Optuna? (mi recomendación: Optuna para BETO, grid
   chiquito para baseline).
2. ¿Class weights para el desbalance? (sí, recomendado).
3. ¿Mixed-precision training? (sí si hay GPU).
4. ¿Validación cruzada o hold-out? (hold-out 70/10/20 ya hecho).
