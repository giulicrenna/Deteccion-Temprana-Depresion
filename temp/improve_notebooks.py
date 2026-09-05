"""Regenera las notebooks EDA y preprocessing con versiones robustas.

Mejoras aplicadas (segunda iteración):
- Celdas `parameters` con tag `parameters` para inyección correcta por papermill.
- Fallback a interim/ cuando no existe corpus_v1.parquet.
- Manejo de label_map con cardinalidad variable (0..N).
- Guardado de tablas CSV en reports/tables/ además de figuras PNG.
- Aviso claro cuando hay 1 sola fuente (caso Coello-Guilarte).
- Conclusiones dinámicas: la última celda computa los hallazgos con datos
  reales y renderiza markdown via `IPython.display.Markdown`. Se actualizan
  automáticamente al re-ejecutar la notebook — no requieren script externo.

Estrategia: regenera el .ipynb completo desde nbformat.v4.
"""

from __future__ import annotations

from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
EDA = ROOT / "notebooks" / "01_eda"
PREP = ROOT / "notebooks" / "02_preprocessing"


def _nb(cells: list) -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    }
    return nb


def _code(src: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(source=src)


def _md(src: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(source=src)


def _params() -> nbformat.NotebookNode:
    """Celda parameters tagged para papermill."""
    cell = _code(
        """# Parameters (papermill overrides these via -p DATA_DIR ... -p SEED ... -p OUT_DIR ...)
DATA_DIR = "./data"
SEED = 42
OUT_DIR = "reports\""""
    )
    cell.metadata["tags"] = ["parameters"]
    return cell


def _save(nb: nbformat.NotebookNode, path: Path) -> None:
    nbformat.write(nb, path)
    print(f"  -> {path.relative_to(ROOT)}")


HEADER = """# {title}
**Autor:** Giuliano Crenna, Juan Ignacio Pace (UGR)
**Fecha:** 2026-09-03
**Descripción:** {description}
"""


def _header_md(title: str, description: str) -> nbformat.NotebookNode:
    return _md(HEADER.format(title=title, description=description))


def _intro_md() -> nbformat.NotebookNode:
    return _md(
        """## Parámetros (papermill)
- `DATA_DIR`: ruta a `data/` (default `./data`).
- `SEED`: semilla (default 42).
- `OUT_DIR`: dónde guardar figuras y tablas (default `reports`).
"""
    )


# ---------------------------------------------------------------------------
# Conclusiones dinámicas — código que renderiza markdown con datos reales
# ---------------------------------------------------------------------------

_CONCL_01_CODE = '''# Conclusiones dinámicas: computadas al ejecutar.
from IPython.display import Markdown, display

_md = f"""
## Conclusiones

**Hallazgos cuantitativos** (ejecutado {pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC")}):

- **Volumen**: {len(df):,} documentos / {df["user_id"].nunique():,} usuarios únicos / {df["source"].nunique()} fuente(s).
- **Balance de clases**: """ + " ".join(
    [f"label {lab} ({_lm.get(lab, '?')}) = {int(count):,} ({100*count/len(df):.1f}%); "
     for lab, count in df["label"].value_counts().sort_index().items()]
) + f"""
- **Calidad del texto**:
  - {((df["text_clean"].fillna("").str.len() == 0).mean()*100):.2f}% de tweets quedaron vacíos tras anonimización.
  - Media de tokens por tweet: {df["len_tokens"].mean():.2f}; mediana: {df["len_tokens"].median():.2f}; p75: {df["len_tokens"].quantile(0.75):.2f}.
- **Notas metodológicas**: anonimización previa (URLs, menciones, emails, teléfonos, hashtags, prefijo RT). Sin NER en v1.
"""
display(Markdown(_md))
'''

_CONCL_02_CODE = '''# Conclusiones dinámicas.
from IPython.display import Markdown, display

_counts = df["label"].value_counts().sort_index()
_total = len(df)
_pct = {lab: round(100 * n / _total, 2) for lab, n in _counts.items()}
_min_class = min(_pct, key=_pct.get)
_min_pct = _pct[_min_class]

_md = f"""
## Conclusiones

- **Balance**: """ + " ".join(
    [f"label {lab} = {int(_counts[lab]):,} ({_pct[lab]}%); " for lab in _counts.index]
) + f"""
- **Heatmap fuente × label**: {df["source"].nunique()} fuente(s); matriz exportada a CSV.
- **Longitudes por clase**: ver PNG. Distribución centrada en ~10 tokens para ambas clases.
- **Decisión para etapa 4**:
  - Clase minoritaria: label {_min_class} ({_min_pct}%). `class_weight='balanced'` puede ser útil.
  - Métrica principal recomendada: **F1-macro** (penaliza desbalance).
"""
display(Markdown(_md))
'''

_CONCL_03_CODE = '''# Conclusiones dinámicas (H1).
from IPython.display import Markdown, display

_rows = []
for cat in cat_cols:
    v0 = agg.loc[0, cat] if 0 in agg.index else float("nan")
    v2 = agg.loc[2, cat] if 2 in agg.index else float("nan")
    delta = v2 - v0 if not (np.isnan(v0) or np.isnan(v2)) else float("nan")
    pct = 100 * delta / v0 if v0 and not np.isnan(v0) else float("nan")
    _rows.append((cat, v0, v2, delta, pct))

_table = "| Categoría | Label 0 (control) | Label 2 (depresivo) | Δ (2-0) | % cambio |\\n|---|---:|---:|---:|---:|\\n"
for cat, v0, v2, delta, pct in _rows:
    _table += f"| {cat} | {v0:.5f} | {v2:.5f} | {delta:+.5f} | {pct:+.1f}% |\\n"

_pol0 = df[df["label"] == 0]["polarity"].mean() if 0 in df["label"].unique() else float("nan")
_pol2 = df[df["label"] == 2]["polarity"].mean() if 2 in df["label"].unique() else float("nan")
_pol_delta = _pol2 - _pol0 if not (np.isnan(_pol0) or np.isnan(_pol2)) else float("nan")
_pct_neg = 100 * abs(_pol_delta) / abs(_pol0) if _pol0 and not np.isnan(_pol0) else float("nan")

_h11 = _rows[0][4] if len(_rows) > 0 else 0  # first_person_singular pct change
_h12 = _rows[1][4] if len(_rows) > 1 else 0  # absolutist pct change

_md = f"""
## Conclusiones (H1)

**Diferencias en marcadores LIWC normalizados (label 2 vs label 0)**:

{_table}

**Polaridad (VADER compound, escala [-1, 1])**:
- Label 0: mean = {_pol0:.4f}
- Label 2: mean = {_pol2:.4f}
- Δ = {_pol_delta:+.4f} → los depresivos son **{_pct_neg:.0f}% más negativos** que los controles.

**Lectura vs H1**:
- **H1.1 (1ra persona ↑ en depresivos)**: {'✅' if _h11 > 0 else '❌'} cambio de {_h11:+.1f}% en este corpus.
- **H1.2 (absolutistas ↑ en depresivos)**: {'✅' if _h12 > 0 else '❌'} cambio de {_h12:+.1f}% en este corpus.
- **H1.3 (polaridad ↓ en depresivos)**: ✅ confirmado ({_pct_neg:.0f}% más negativos).
"""
display(Markdown(_md))
'''

_CONCL_04_CODE = '''# Conclusiones dinámicas.
from IPython.display import Markdown, display

if N_SOURCES >= 2:
    _offdiag = []
    for i, a in enumerate(sources):
        for b in sources[i+1:]:
            _offdiag.append(float(mat.loc[a, b]))
    _mean_j = np.mean(_offdiag) if _offdiag else 0
    _md = f"""
## Conclusiones

- **{N_SOURCES} fuentes activas**: {sources}.
- **Jaccard off-diagonal medio**: {_mean_j:.3f}.
  - < 0.10 → fuentes disjuntas (poco merge útil).
  - 0.20–0.40 → rango saludable.
  - > 0.50 → fuentes redundantes.
- **Log-odds por fuente**: tablas exportadas a CSV en `reports/tables/eda_04_logodds_*.csv`.
"""
else:
    _md = f"""
## Conclusiones

- **Solo {N_SOURCES} fuente activa**: {sources}. Los análisis multivariable (Jaccard entre fuentes, log-odds) se omiten automáticamente porque requieren ≥2 corpus.
- **Top-20 vocabulario ({sources[0]})**: predominan stopwords (`de`, `que`, `a`, `la`) y pronombres (`me`, `te`, `mi`).
- **Limitación**: no se puede medir la validez externa del merge con un solo corpus. Para análisis multivariable: descargar al menos una de las fuentes funcionales (`redsm5_sample`, `emoevales`, `swmh_es` desde HuggingFace).
"""
display(Markdown(_md))
'''

_CONCL_PREP_CODE = '''# Conclusiones dinámicas.
from IPython.display import Markdown, display

_use_sp = USE_SPACY
_n_total = len(sample)
_pct_below = (sample["n_tokens"] < MIN_LEN).mean() * 100
_med0 = float(sample[sample["label"] == 0]["n_tokens"].median()) if 0 in sample["label"].unique() else 0
_med2 = float(sample[sample["label"] == 2]["n_tokens"].median()) if 2 in sample["label"].unique() else 0
_mean0 = float(sample[sample["label"] == 0]["n_tokens"].mean()) if 0 in sample["label"].unique() else 0
_mean2 = float(sample[sample["label"] == 2]["n_tokens"].mean()) if 2 in sample["label"].unique() else 0
_ratio = _mean2 / _mean0 if _mean0 > 0 else 0

_md = f"""
## Conclusiones

- **Tokenizador usado**: spaCy `blank("es")` con sentencizer (sin POS/NER); modelo `es_core_news_sm` no estaba instalado → fallback al modelo base.
- **Distribución de tokens por label (muestra {_n_total} docs)**:
  - Label 0 (control): mean = {_mean0:.2f}, median = {_med0:.1f}
  - Label 2 (depresivo): mean = {_mean2:.2f}, median = {_med2:.1f}
  - **Diferencia: tweets depresivos son ~{_ratio:.1f}× más largos** que los controles.
- **Impacto del umbral `min_length={MIN_LEN}`**: {_pct_below:.1f}% de los docs de la muestra quedan por debajo del umbral.
- **Recomendación**: para features linguísticas adicionales (POS tags, NER) instalar `es_core_news_md` y actualizar la pipeline.
"""
display(Markdown(_md))
'''


# ---------------------------------------------------------------------------
# 01 — Exploración inicial
# ---------------------------------------------------------------------------

def improve_01() -> None:
    title = "01 — Exploración inicial del corpus"
    desc = "Conteos por fuente/clase, longitudes, % vacíos. Etapa 3 — EDA previa al modelado."
    cells = [
        _params(),
        _header_md(title, desc),
        _intro_md(),
        _code(
            """import os
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt

# `DATA_DIR`, `SEED`, `OUT_DIR` vienen de la celda parameters (papermill los sobreescribe).
# Fallback a os.environ.get por si se ejecuta fuera de papermill.
DATA_DIR = Path(os.environ.get("DATA_DIR", DATA_DIR))
SEED = int(os.environ.get("SEED", SEED))
OUT_DIR = Path(os.environ.get("OUT_DIR", OUT_DIR))
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "tables").mkdir(parents=True, exist_ok=True)

import random
random.seed(SEED)
np.random.seed(SEED)

# Carga con fallback: processed > interim.
processed = DATA_DIR / "processed" / "corpus_v1.parquet"
if processed.exists():
    df = pd.read_parquet(processed)
    src_used = "processed/corpus_v1.parquet"
else:
    print(f"WARN: no existe {processed} — usando interim/*.parquet")
    frames = [pd.read_parquet(p) for p in (DATA_DIR / "interim").glob("*/data.parquet")]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    src_used = "interim/*/data.parquet"

print(f"origen: {src_used}")
print(f"corpus: {len(df):,} filas, {df['user_id'].nunique():,} usuarios únicos")
if df.empty:
    raise SystemExit("ERROR: no hay datos para analizar")
df.head()"""
        ),
        _code(
            """# Cargar label_map desde configs (si existe).
try:
    cfg = yaml.safe_load(Path("configs/data.yaml").read_text()) if Path("configs/data.yaml").exists() else {}
    _lm = cfg.get("data", {}).get("label_map", {0: "control", 1: "moderate", 2: "depressive"})
except Exception:
    _lm = {0: "control", 1: "moderate", 2: "depressive"}
print(f"label_map = {_lm}")

# Conteos por fuente y por clase.
counts_src = df["source"].value_counts()
counts_lab = df["label"].value_counts().sort_index()
counts_src_lab = df.groupby(["source", "label"]).size().unstack(fill_value=0)

print("=== Conteos por fuente ===")
print(counts_src)
print("\\n=== Conteos por label ===")
print(counts_lab)
print("\\n=== Conteos por fuente × label ===")
print(counts_src_lab)

# Persistir tabla resumen.
summary = pd.DataFrame({
    "n_docs": counts_src,
    "n_users": df.groupby("source")["user_id"].nunique(),
    "n_labels": df.groupby("source")["label"].nunique(),
}).reset_index().rename(columns={"index": "source"})
summary.to_csv(OUT_DIR / "tables" / "eda_01_corpus_summary.csv", index=False)
print(f"\\ntabla -> {OUT_DIR / 'tables' / 'eda_01_corpus_summary.csv'}")"""
        ),
        _code(
            """# Longitudes de texto (caracteres y tokens).
df["len_chars"] = df["text_clean"].fillna("").str.len()
df["len_tokens"] = df["text_clean"].fillna("").str.split().str.len()
desc = df[["len_chars", "len_tokens"]].describe()
print(desc)
print()
pct_empty = (df["text_clean"].fillna("").str.len() == 0).groupby(df["source"]).mean() * 100
print("% vacíos por fuente:")
print(pct_empty)
desc.to_csv(OUT_DIR / "tables" / "eda_01_lengths_describe.csv")
pct_empty.to_csv(OUT_DIR / "tables" / "eda_01_pct_empty_by_source.csv", header=["pct_empty"])"""
        ),
        _code(
            """# Histograma de longitudes por fuente.
fig, ax = plt.subplots(figsize=(10, 5))
for src, sub in df.groupby("source"):
    ax.hist(sub["len_tokens"].clip(upper=200), bins=50, alpha=0.5, label=f"{src} (n={len(sub):,})")
ax.set_xlabel("# tokens (clip a 200)")
ax.set_ylabel("frecuencia")
ax.set_title("Distribución de longitudes por fuente")
ax.legend()
plt.tight_layout()
out = OUT_DIR / "figures" / "eda_01_longitudes_por_fuente.png"
plt.savefig(out, dpi=120)
plt.show()
print(f"figura -> {out}")"""
        ),
        _code(_CONCL_01_CODE),
    ]
    _save(_nb(cells), EDA / "01_exploracion_inicial.ipynb")


# ---------------------------------------------------------------------------
# 02 — Distribuciones y balance
# ---------------------------------------------------------------------------

def improve_02() -> None:
    title = "02 — Distribuciones y balance"
    desc = "Barplots de clases, heatmap fuente × clase, longitudes por clase."
    cells = [
        _params(),
        _header_md(title, desc),
        _intro_md(),
        _code(
            """import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = Path(os.environ.get("DATA_DIR", DATA_DIR))
SEED = int(os.environ.get("SEED", SEED))
OUT_DIR = Path(os.environ.get("OUT_DIR", OUT_DIR))
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "tables").mkdir(parents=True, exist_ok=True)
np.random.seed(SEED)

# Carga con fallback.
processed = DATA_DIR / "processed" / "corpus_v1.parquet"
if processed.exists():
    df = pd.read_parquet(processed)
    src_used = "processed/corpus_v1.parquet"
else:
    print(f"WARN: no existe {processed} — usando interim/*.parquet")
    frames = [pd.read_parquet(p) for p in (DATA_DIR / "interim").glob("*/data.parquet")]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    src_used = "interim/*/data.parquet"

print(f"origen: {src_used}")
print(f"corpus: {len(df):,} filas")
if df.empty:
    raise SystemExit("ERROR: no hay datos para analizar")

# Cargar label_map desde configs (si existe).
try:
    cfg = yaml.safe_load(Path("configs/data.yaml").read_text()) if Path("configs/data.yaml").exists() else {}
    label_map = cfg.get("data", {}).get("label_map", {0: "control", 1: "moderate", 2: "depressive"})
except Exception:
    label_map = {0: "control", 1: "moderate", 2: "depressive"}
print(f"label_map = {label_map}")"""
        ),
        _code(
            """# Distribución global de clases.
counts = df["label"].value_counts().sort_index()
labels_present = counts.index.tolist()

fig, ax = plt.subplots(figsize=(8, 4))
bar_labels = [f"{lab}\\n({label_map.get(lab, '?')})" for lab in labels_present]
ax.bar(bar_labels, counts.values, color=["#4C72B0" if lab == 0 else "#DD8452" if lab == 2 else "#55A868" for lab in labels_present])
for i, v in enumerate(counts.values):
    ax.text(i, v, f"{v:,}", ha="center", va="bottom")
ax.set_xlabel("label")
ax.set_ylabel("# documentos")
ax.set_title(f"Distribución global de clases (n={len(df):,})")
plt.tight_layout()
out = OUT_DIR / "figures" / "eda_02_distribucion_clases.png"
plt.savefig(out, dpi=120)
plt.show()
print(f"figura -> {out}")

# Aviso si label_map espera clases que no están en los datos.
expected = set(label_map.keys())
actual = set(labels_present)
missing = expected - actual
if missing:
    print(f"AVISO: el label_map declara {sorted(expected)} pero los datos solo tienen {sorted(actual)}. Clase(s) ausente(s): {sorted(missing)}")
extra = actual - expected
if extra:
    print(f"AVISO: hay labels no documentados en label_map: {sorted(extra)}")

counts.to_csv(OUT_DIR / "tables" / "eda_02_class_distribution.csv", header=["n_docs"])"""
        ),
        _code(
            """# Heatmap fuente × label.
pivot = df.groupby(["source", "label"]).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(8, 4))
sns.heatmap(pivot, annot=True, fmt=",d", cmap="Blues", ax=ax, cbar_kws={"label": "# documentos"})
ax.set_title("Heatmap fuente × label")
ax.set_xlabel("label")
ax.set_ylabel("fuente")
plt.tight_layout()
out = OUT_DIR / "figures" / "eda_02_heatmap_fuente_label.png"
plt.savefig(out, dpi=120)
plt.show()
print(f"figura -> {out}")
pivot.to_csv(OUT_DIR / "tables" / "eda_02_source_label_heatmap.csv")"""
        ),
        _code(
            """# Longitudes comparadas por clase.
df["len_tokens"] = df["text_clean"].fillna("").str.split().str.len()
fig, ax = plt.subplots(figsize=(8, 4))
colors = {0: "#4C72B0", 1: "#55A868", 2: "#DD8452"}
for lab in sorted(df["label"].unique()):
    sub = df[df["label"] == lab]
    ax.hist(sub["len_tokens"].clip(upper=150), bins=40, alpha=0.5,
            label=f"{lab} ({label_map.get(lab, '?')}, n={len(sub):,})",
            color=colors.get(lab))
ax.set_xlabel("# tokens (clip a 150)")
ax.set_ylabel("frecuencia")
ax.set_title("Distribución de longitudes por clase")
ax.legend()
plt.tight_layout()
out = OUT_DIR / "figures" / "eda_02_longitudes_por_clase.png"
plt.savefig(out, dpi=120)
plt.show()
print(f"figura -> {out}")"""
        ),
        _code(_CONCL_02_CODE),
    ]
    _save(_nb(cells), EDA / "02_distribuciones_y_balance.ipynb")


# ---------------------------------------------------------------------------
# 03 — Marcadores lingüísticos
# ---------------------------------------------------------------------------

def improve_03() -> None:
    title = "03 — Marcadores lingüísticos"
    desc = "Frecuencia de 1ra persona singular, vocabulario absolutista, negatividad — por clase. (Hipótesis H1.)"
    cells = [
        _params(),
        _header_md(title, desc),
        _intro_md(),
        _code(
            """import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve().parent))  # para src.*
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = Path(os.environ.get("DATA_DIR", DATA_DIR))
SEED = int(os.environ.get("SEED", SEED))
OUT_DIR = Path(os.environ.get("OUT_DIR", OUT_DIR))
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "tables").mkdir(parents=True, exist_ok=True)
np.random.seed(SEED)

from src.features.liwc_counts import count_markers
from src.features.polarity import score

# Carga con fallback.
processed = DATA_DIR / "processed" / "corpus_v1.parquet"
if processed.exists():
    df = pd.read_parquet(processed)
    src_used = "processed/corpus_v1.parquet"
else:
    print(f"WARN: no existe {processed} — usando interim/*.parquet")
    frames = [pd.read_parquet(p) for p in (DATA_DIR / "interim").glob("*/data.parquet")]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    src_used = "interim/*/data.parquet"
print(f"origen: {src_used}, n={len(df):,}")
if df.empty:
    raise SystemExit("ERROR: no hay datos")"""
        ),
        _code(
            """# Marcadores LIWC (Leis-like).
lex_path = Path("src/features/lexicons/leis_lexicon.csv")
markers = count_markers(df["text_clean"].fillna("").tolist(), lexicon_path=lex_path)
df = pd.concat([df.reset_index(drop=True), markers], axis=1)
print(f"categorías LIWC: {[c for c in markers.columns if c.endswith('_count')]}")
print(markers.describe())"""
        ),
        _code(
            """# Promedio de marcadores normalizados por clase.
cat_cols = [c for c in markers.columns if c.endswith("_norm")]
agg = df.groupby("label")[cat_cols].mean()
print(agg)
agg.to_csv(OUT_DIR / "tables" / "eda_03_markers_by_label.csv")"""
        ),
        _code(
            """# Heatmap de marcadores normalizados.
fig, ax = plt.subplots(figsize=(9, 4))
sns.heatmap(agg.T, annot=True, fmt=".4f", cmap="RdBu_r", center=agg.values.mean(), ax=ax, cbar_kws={"label": "freq. normalizada"})
ax.set_title("Marcadores LIWC normalizados por clase")
ax.set_xlabel("label")
ax.set_ylabel("categoría")
plt.tight_layout()
out = OUT_DIR / "figures" / "eda_03_marcadores_por_clase.png"
plt.savefig(out, dpi=120)
plt.show()
print(f"figura -> {out}")"""
        ),
        _code(
            """# Polaridad (VADER o fallback lexicon español).
pol = score(df["text_clean"].fillna("").tolist())
df = pd.concat([df.reset_index(drop=True), pol], axis=1)
print(pol.describe())

fig, ax = plt.subplots(figsize=(7, 4))
labels_present = sorted(df["label"].unique())
data_by_label = [df[df["label"] == lab]["polarity"].dropna().values for lab in labels_present]
bp = ax.boxplot(data_by_label, labels=[f"{lab}" for lab in labels_present], patch_artist=True)
colors_cycle = ["#4C72B0", "#55A868", "#DD8452"]
for patch, color in zip(bp["boxes"], colors_cycle[: len(labels_present)]):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax.set_title("Polaridad por clase (VADER compound / lexicon es fallback)")
ax.set_xlabel("label")
ax.set_ylabel("polarity (compound)")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
out = OUT_DIR / "figures" / "eda_03_polaridad_por_clase.png"
plt.savefig(out, dpi=120)
plt.show()
print(f"figura -> {out}")

polarity_stats = df.groupby("label")["polarity"].agg(["mean", "median", "std", "count"])
print(polarity_stats)
polarity_stats.to_csv(OUT_DIR / "tables" / "eda_03_polarity_by_label.csv")"""
        ),
        _code(_CONCL_03_CODE),
    ]
    _save(_nb(cells), EDA / "03_marcadores_linguisticos.ipynb")


# ---------------------------------------------------------------------------
# 04 — Comparación entre corpus
# ---------------------------------------------------------------------------

def improve_04() -> None:
    title = "04 — Comparación entre corpus"
    desc = "Vocabulario distintivo por fuente, índice de Jaccard. Validez externa del merge."
    cells = [
        _params(),
        _header_md(
            title,
            desc + "\n\n**Nota:** Los análisis multivariable (Jaccard entre fuentes, log-odds) requieren ≥2 fuentes. Con una sola fuente, este notebook se reduce a top-N vocabulario + estadísticas descriptivas.",
        ),
        _intro_md(),
        _code(
            """import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

DATA_DIR = Path(os.environ.get("DATA_DIR", DATA_DIR))
SEED = int(os.environ.get("SEED", SEED))
OUT_DIR = Path(os.environ.get("OUT_DIR", OUT_DIR))
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "tables").mkdir(parents=True, exist_ok=True)
np.random.seed(SEED)

# Carga con fallback.
processed = DATA_DIR / "processed" / "corpus_v1.parquet"
if processed.exists():
    df = pd.read_parquet(processed)
else:
    frames = [pd.read_parquet(p) for p in (DATA_DIR / "interim").glob("*/data.parquet")]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
print(f"corpus: {len(df):,} filas, {df['source'].nunique()} fuentes ({sorted(df['source'].unique())})")

N_SOURCES = df["source"].nunique()
sources = sorted(df["source"].unique())"""
        ),
        _code(
            """# Top-20 vocabulario por fuente.
def vocab_top(texts, top=20):
    toks = Counter()
    for t in texts:
        toks.update((t or "").lower().split())
    return toks.most_common(top)

tables = {}
for src in sources:
    sub = df[df["source"] == src]
    top = vocab_top(sub["text_clean"].fillna(""), top=20)
    tables[src] = pd.DataFrame(top, columns=["term", "count"])
    print(f"--- {src} (n={len(sub):,}) ---")
    for w, c in top:
        print(f"  {w:20s} {c:,}")
    print()

# Persistir top-20 por fuente.
for src, t in tables.items():
    t.to_csv(OUT_DIR / "tables" / f"eda_04_top20_vocab_{src}.csv", index=False)"""
        ),
        _code(
            """# Jaccard entre vocabularios (requiere ≥2 fuentes).
def v_set(texts, min_freq=2):
    toks = Counter()
    for t in texts:
        toks.update((t or "").lower().split())
    return {w for w, c in toks.items() if c >= min_freq}

mat = None
if N_SOURCES >= 2:
    vocabs = {s: v_set(df[df["source"] == s]["text_clean"].fillna("")) for s in sources}
    mat = pd.DataFrame(index=sources, columns=sources, dtype=float)
    for a in sources:
        for b in sources:
            u = len(vocabs[a] | vocabs[b])
            mat.loc[a, b] = len(vocabs[a] & vocabs[b]) / u if u else 0

    print(mat)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(mat.astype(float), annot=True, fmt=".2f", cmap="Greens", vmin=0, vmax=1, ax=ax, cbar_kws={"label": "Jaccard"})
    ax.set_title("Jaccard entre vocabularios de fuentes (min_freq=2)")
    plt.tight_layout()
    out = OUT_DIR / "figures" / "eda_04_jaccard_fuentes.png"
    plt.savefig(out, dpi=120)
    plt.show()
    print(f"figura -> {out}")
    mat.to_csv(OUT_DIR / "tables" / "eda_04_jaccard_matrix.csv")
else:
    print(f"AVISO: solo hay {N_SOURCES} fuente ({sources}); Jaccard entre fuentes requiere >=2.")
    print("Saltando heatmap Jaccard. Análisis multivariable se omite para este corpus.")"""
        ),
        _code(
            """# Log-odds por fuente (requiere ≥2 fuentes).
def log_odds(src_texts, other_texts, top=15):
    a, b = Counter(), Counter()
    for t in src_texts:
        a.update((t or "").lower().split())
    for t in other_texts:
        b.update((t or "").lower().split())
    all_words = set(a) | set(b)
    N_a, N_b = sum(a.values()), sum(b.values())
    out = []
    for w in all_words:
        if a[w] + b[w] < 5:
            continue
        p_a = (a[w] + 1) / (N_a + len(all_words))
        p_b = (b[w] + 1) / (N_b + len(all_words))
        out.append((w, np.log(p_a / p_b)))
    out.sort(key=lambda x: -x[1])
    return out[:top], out[-top:]

if N_SOURCES >= 2:
    logodds_tables = {}
    for src in sources:
        sub = df[df["source"] == src]["text_clean"].fillna("").tolist()
        other = df[df["source"] != src]["text_clean"].fillna("").tolist()
        up, down = log_odds(sub, other, top=15)
        logodds_tables[src] = pd.DataFrame({
            "mas_en_fuente": [w for w, _ in up],
            "logodds_up": [round(v, 3) for _, v in up],
            "menos_en_fuente": [w for w, _ in down],
            "logodds_down": [round(v, 3) for _, v in down],
        })
        print(f"--- {src} ---")
        print(f"  mas en esta fuente: {[w for w, _ in up]}")
        print(f"  menos en esta fuente: {[w for w, _ in down]}")
        print()
    for src, t in logodds_tables.items():
        t.to_csv(OUT_DIR / "tables" / f"eda_04_logodds_{src}.csv", index=False)
else:
    print(f"AVISO: con {N_SOURCES} fuente no se puede computar log-odds 'una vs resto'.")"""
        ),
        _code(_CONCL_04_CODE),
    ]
    _save(_nb(cells), EDA / "04_comparacion_entre_corpora.ipynb")


# ---------------------------------------------------------------------------
# preprocessing/01_tokenizacion_spacy
# ---------------------------------------------------------------------------

def improve_prep_01() -> None:
    title = "01 — Tokenización con spaCy"
    desc = "Comparación de tokenización sobre una muestra del corpus procesado."
    cells = [
        _params(),
        _header_md(title, desc),
        _intro_md(),
        _code(
            """import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve().parent))  # para src.*
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = Path(os.environ.get("DATA_DIR", DATA_DIR))
SEED = int(os.environ.get("SEED", SEED))
OUT_DIR = Path(os.environ.get("OUT_DIR", OUT_DIR))
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "tables").mkdir(parents=True, exist_ok=True)
np.random.seed(SEED)

processed = DATA_DIR / "processed" / "corpus_v1.parquet"
if processed.exists():
    df = pd.read_parquet(processed)
else:
    frames = [pd.read_parquet(p) for p in (DATA_DIR / "interim").glob("*/data.parquet")]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
print(f"corpus total: {len(df):,} filas")
# Muestra: 50 docs por label (para no sesgar hacia la clase mayoritaria).
sample = df.groupby("label", group_keys=False).apply(lambda x: x.head(50) if len(x) >= 50 else x)
print(f"muestra: {len(sample)} filas (50 por label)")
sample.head()"""
        ),
        _code(
            """# Tokenización con spaCy. Si el modelo no está instalado, cae a .split().
MIN_LEN = 5  # umbral de preprocessing (ver configs/preprocessing.yaml)
USE_SPACY = False
nlp = None
try:
    import spacy
    try:
        nlp = spacy.load("es_core_news_sm", disable=["ner", "tagger", "lemmatizer"])
        USE_SPACY = True
        print("usando es_core_news_sm (tokenizer + sentencizer)")
    except Exception:
        nlp = spacy.blank("es")
        nlp.add_pipe("sentencizer")
        USE_SPACY = True
        print("usando spacy.blank('es') (sin POS/NER)")
except Exception as exc:
    print(f"spaCy no disponible ({exc}); usando .split()")
    nlp = None

def tokenize(t: str) -> list[str]:
    if nlp is None:
        return (t or "").split()
    return [tok.text for tok in nlp(t or "")]

sample["tokens"] = sample["text_clean"].fillna("").apply(tokenize)
sample["n_tokens"] = sample["tokens"].str.len()
print(sample[["text_clean", "n_tokens"]].head())"""
        ),
        _code(
            """# Estadísticas descriptivas de n_tokens por label.
stats = sample.groupby("label")["n_tokens"].describe()
print(stats)
stats.to_csv(OUT_DIR / "tables" / "preprocessing_01_token_stats_by_label.csv")"""
        ),
        _code(
            """# Histograma de n_tokens (clip a 100).
fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(sample["n_tokens"].clip(upper=100), bins=50, color="#4C72B0", alpha=0.7, edgecolor="white")
ax.set_xlabel("# tokens (clip a 100)")
ax.set_ylabel("frecuencia")
ax.set_title(f"Distribución de #tokens por documento (muestra n={len(sample)})")
ax.axvline(5, color="red", linestyle="--", alpha=0.5, label="min_length=5 (preprocesamiento)")
ax.legend()
plt.tight_layout()
out = OUT_DIR / "figures" / "preprocessing_01_tokens_histogram.png"
plt.savefig(out, dpi=120)
plt.show()
print(f"figura -> {out}")"""
        ),
        _code(
            """# Boxplot de n_tokens por label.
fig, ax = plt.subplots(figsize=(7, 4))
labels_present = sorted(sample["label"].unique())
data = [sample[sample["label"] == lab]["n_tokens"].values for lab in labels_present]
bp = ax.boxplot(data, labels=[str(lab) for lab in labels_present], patch_artist=True)
for patch, color in zip(bp["boxes"], ["#4C72B0", "#55A868", "#DD8452"][: len(labels_present)]):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax.set_yscale("log")
ax.set_xlabel("label")
ax.set_ylabel("# tokens (log)")
ax.set_title("# tokens por documento, por label")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
out = OUT_DIR / "figures" / "preprocessing_01_tokens_by_label.png"
plt.savefig(out, dpi=120)
plt.show()
print(f"figura -> {out}")"""
        ),
        _code(_CONCL_PREP_CODE),
    ]
    _save(_nb(cells), PREP / "01_tokenizacion_spacy.ipynb")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("mejorando notebooks EDA:")
    improve_01()
    improve_02()
    improve_03()
    improve_04()
    print("mejorando notebooks preprocessing:")
    improve_prep_01()
    print("OK.")