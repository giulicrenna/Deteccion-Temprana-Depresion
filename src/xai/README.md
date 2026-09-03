# src/xai/

Módulo de explicabilidad (etapa 6 del cronograma de la tesis).

## Estado actual

**Stubs únicamente.** Los análisis de SHAP / LIME se implementarán en la
etapa 6, después de tener modelos entrenados y validados.

## Plan de implementación

| Archivo | Técnica | Output |
|---|---|---|
| `shap_analysis.py` | SHAP KernelExplainer (baseline) o SHAP sobre transformers | `reports/figures/shap_*.png` |
| `lime_analysis.py` | LIME TextExplainer para ejemplos individuales | `reports/figures/lime_*.html` |
| `attention_viz.py` | Visualización de attention de BETO (BertViz) | `reports/figures/attn_*.html` |

## Hipótesis a atacar

- **H1**: la 1ra persona singular y vocabulario absolutista son los
  marcadores más salientes.
- **H2**: la polaridad negativa explica la mayor parte de la señal.
- **H3**: los modelos que mejor balancean interpretabilidad y métricas
  son los de tipo logístico + features manuales (vs. BETO fine-tuned).

## Convenciones

- Reportar siempre SHAP values globales (bar) + locales (waterfall) para
  el mismo ejemplo.
- LIME se usa solo para inspección cualitativa (no como métrica).
- Attention viz solo aplica a BETO (no al baseline).
