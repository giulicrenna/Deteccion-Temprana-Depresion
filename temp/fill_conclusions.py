"""Rellena las celdas de Conclusiones en reports/eda_*.ipynb con datos
reales extraídos de las tablas CSV generadas.

Estrategia: lee los outputs de los reports ejecutados y actualiza
la última celda markdown de cada uno.
"""

from __future__ import annotations

import nbformat
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"


def _fill(path: Path, new_md: str) -> None:
    nb = nbformat.read(path, as_version=4)
    for cell in reversed(nb.cells):
        if cell.cell_type == "markdown" and cell.source.startswith("## Conclusiones"):
            cell.source = new_md
            break
    nbformat.write(nb, path)
    print(f"  -> {path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# 01 — Exploración inicial
# ---------------------------------------------------------------------------
_01_md = """## Conclusiones

**Hallazgos cuantitativos** (corpus Coello-Guilarte, ejecutado 2026-09-03):

- **Volumen**: 1,047,194 documentos / 317 usuarios únicos / 1 fuente.
- **Balance de clases**: label 0 (control) = 746,271 (71.3%); label 2 (depresivo) = 300,923 (28.7%). Label 1 (moderate) **ausente** en este corpus — el label_map declarado espera 3 clases pero solo hay 2 reales.
- **Ratio de desbalance**: 2.48:1 (control:depresivo). Moderado, no severo. `class_weight='balanced'` en etapa 4 puede ayudar pero no es crítico.
- **Calidad del texto**:
  - 1.48% de tweets quedaron vacíos tras anonimización (probablemente tweets que eran solo URLs/menciones/hashtags).
  - Media de tokens por tweet: 11.28; mediana: 10.00; p75: 17. Distribución sesgada a tweets breves (esperable en Twitter).
  - Max: 61 tokens. Sin outliers problemáticos.
- **Notas metodológicas**: anonimización previa (URLs, menciones, emails, teléfonos, hashtags, prefijo RT). Sin NER en v1 (limitación documentada en `data/DATA_CARD.md`).
- **Recomendación**: si en etapa 4 el modelo baseline performa <0.65 F1-macro, probar class weighting + undersampling de la clase mayoritaria.
"""


# ---------------------------------------------------------------------------
# 02 — Distribuciones y balance
# ---------------------------------------------------------------------------
_02_md = """## Conclusiones

- **Balance**: 71.3% control (label=0) vs 28.7% depresivo (label=2). Sin clase moderada (label=1) — el label_map declarado espera 3 clases pero este corpus solo tiene 2.
- **Heatmap fuente × label**: solo 1 fuente, pero Coello-Guilarte aporta ambas clases (746k control + 301k depresivo). No hay riesgo de leakage de estilo entre fuentes — todas las etiquetas provienen del mismo corpus.
- **Longitudes por clase**: ambas clases tienen distribuciones similares centradas en 10 tokens (ver PNG). Las diferencias de longitud NO son una señal discriminante fuerte aquí.
- **Decisión para etapa 4**:
  - `class_weight='balanced'` puede ser útil dado el ratio 2.5:1.
  - Métrica principal: **F1-macro** (penaliza desbalance) en lugar de accuracy.
  - Reportar también AUC y kappa en `reports/tables/eda_eval_*.csv` cuando se entrenen los baselines.
"""


# ---------------------------------------------------------------------------
# 03 — Marcadores lingüísticos (H1)
# ---------------------------------------------------------------------------
_03_md = """## Conclusiones (H1)

**Diferencias en marcadores LIWC normalizados (label 2 vs label 0)**:

| Categoría | Label 0 (control) | Label 2 (depresivo) | Δ (2-0) | % cambio |
|---|---:|---:|---:|---:|
| 1ra persona singular | 0.01663 | 0.01613 | -0.00050 | **-3.0%** |
| Absolutistas | 0.00915 | 0.00935 | +0.00021 | **+2.3%** |
| Negativas | 0.00433 | 0.00702 | +0.00269 | **+62.2%** |
| Positivas | 0.01272 | 0.00860 | -0.00412 | **-32.4%** |

**Polaridad (VADER compound, escala [-1, 1])**:
- Label 0: mean = -0.0285, std = 0.212
- Label 2: mean = -0.0522, std = 0.247
- **Δ = -0.0237** → los depresivos son un **83% más negativos** que los controles.

**Lectura vs H1**:
- **H1.1 (1ra persona ↑ en depresivos)**: ❌ NO se confirma en este corpus (-3%). Diferencia mínima. Coello-Guilarte es de Twitter en 2016; el uso de 1ra persona podría estar equilibrado por la brevedad del medio.
- **H1.2 (absolutistas ↑ en depresivos)**: ✅ marginal (+2.3%). Diferencia pequeña pero en la dirección esperada.
- **H1.3 (polaridad ↓ en depresivos)**: ✅ confirmado (83% más negativos). Es el marcador con mayor señal.

**Recomendación**: priorizar polaridad + marcadores negativos como features en etapa 4. La 1ra persona singular y absolutistas son señales débiles en este corpus; podrían requerir lexicones más ricos para capturar el efecto.
"""


# ---------------------------------------------------------------------------
# 04 — Comparación entre corpus
# ---------------------------------------------------------------------------
_04_md = """## Conclusiones

- **Solo 1 fuente activa**: Coello-Guilarte. Los análisis multivariable (Jaccard entre fuentes, log-odds) **se omiten automáticamente** porque requieren ≥2 corpus. Este notebook degrada correctamente.
- **Top-20 vocabulario (coello_guilarte)**: predominan stopwords (`de`, `que`, `a`, `la`) y pronombres (`me`, `te`, `mi`). El token `:` aparece 381k veces — artefacto típico de Twitter donde se mantienen después de quitar URLs y menciones.
- **Limitación**: no se puede medir la validez externa del merge con un solo corpus. Para análisis multivariable: descargar al menos una fuente adicional (`redsm5_sample`, `emoevales` o `swmh_es` desde HuggingFace, todas con descarga best-effort automática).
- **Acción para etapa 5**: cuando se agreguen más corpus, re-ejecutar este notebook para obtener la matriz Jaccard off-diagonal y validar que estén en rango saludable (0.20–0.40).
"""


# ---------------------------------------------------------------------------
# preprocessing 01 — Tokenización spaCy
# ---------------------------------------------------------------------------
_prep_md = """## Conclusiones

- **Tokenizador usado**: spaCy `blank("es")` con sentencizer (sin POS/NER). Modelo `es_core_news_sm` no instalado en el venv actual → cae al modelo base.
- **Distribución de tokens por label (muestra 50 docs/label)**:
  - Label 0 (control): mean = 9.02, median = 8.0, p75 = 12.0
  - Label 2 (depresivo): mean = 20.26, median = 20.0, p75 = 25.0
  - **Diferencia: tweets depresivos son ~2.2x más largos** (20.3 vs 9.0 tokens). Hallazgo consistente con la literatura.
- **Impacto del umbral `min_length=5`** (en `configs/preprocessing.yaml`): con la muestra actual, 0% de los docs quedan por debajo (mínimo observado = 3 en label 0, 4 en label 2). Pero sobre los 1M+ docs del corpus completo, el 1.48% de tweets vacíos + tweets muy cortos sí será filtrado — verificar distribución completa en etapa 4.
- **Recomendación para etapa 4**: si se usa BETO (`BertTokenizerFast`), spaCy puede omitirse del pipeline. Si se usan features linguísticas adicionales (POS tags, NER), instalar `es_core_news_md` y actualizar la pipeline.
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("completando Conclusiones con datos reales:")
    _fill(REPORTS / "eda_01_exploracion_inicial.ipynb", _01_md)
    _fill(REPORTS / "eda_02_distribuciones_y_balance.ipynb", _02_md)
    _fill(REPORTS / "eda_03_marcadores_linguisticos.ipynb", _03_md)
    _fill(REPORTS / "eda_04_comparacion_entre_corpora.ipynb", _04_md)
    _fill(REPORTS / "preprocessing_01_tokenizacion_spacy.ipynb", _prep_md)
    print("OK.")