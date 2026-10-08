"""Proyecta los 3 CSV que el usuario dejó en data/interim/ al esquema unificado.

El usuario descargó manualmente versiones distintas a las que esperan los loaders
de `make_dataset.py`:

  - Kaggle SDD v14 (`suicide`/`non-suicide`, 232k) en vez de v13
    (que sería `depression`/`SuicideWatch`/`teenagers`, 348k).
  - Reddit MH Posts como CSV (no JSONL) y FILTRADO a solo r/depression
    (sin controles ADHD/OCD; todos label=2).
  - Depression Reddit (mrjunos) como CSV con columna `labels` (no `label`).

Este script los lee, anonimiza con la misma función que usa `make_dataset.py`,
los proyecta al esquema unificado y los escribe como
`data/interim/<fuente>/data.parquet` para que `merge_corpora.py` los pueda
mergear igual que cualquier corpus procesado por la pipeline estándar.
"""

from __future__ import annotations

import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# ponytail: Windows cp1252 rompe con caracteres no-ASCII al imprimir
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

# Reutilizamos la función de anonimización del pipeline estándar
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.make_dataset import anonymize  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
INTERIM = REPO / "data" / "interim"

# Mapeos de label específicos para cada CSV del usuario
KAGGLE_V14_LABEL = {"suicide": 2, "non-suicide": 0}    # depression+suicideWatch fusionados
DEPRESSION_REDDIT_LABEL = {"1": 2, "0": 0}              # mrjunos usa {0,1}


def hash_id(*parts: str) -> str:
    """SHA-256 truncado a 16 chars, igual que en make_dataset.py."""
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def _utc_iso_from_csv(value) -> str:
    """Reddit created_utc puede venir como ISO 8601 o epoch segundos."""
    if pd.isna(value) or value == "":
        return ""
    s = str(value).strip()
    # ISO 8601 directo (caso típico en el CSV que descargó el usuario)
    try:
        return pd.Timestamp(s).tz_convert("UTC").isoformat()
    except (ValueError, TypeError):
        pass
    # Fallback: epoch segundos
    try:
        return datetime.fromtimestamp(float(s), tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OverflowError):
        return ""


def project_kaggle_sdd() -> pd.DataFrame:
    """Suicide_Detection.csv v14 → label binario."""
    src = INTERIM / "Suicide_Detection.csv"
    df = pd.read_csv(src)
    rows = []
    for i, (text, cls) in enumerate(zip(df["text"].astype(str), df["class"].astype(str))):
        if cls not in KAGGLE_V14_LABEL:
            continue
        text_clean = anonymize(text)
        if not text_clean.strip():
            continue
        rows.append({
            "doc_id": hash_id("kaggle_sdd", str(i)),
            # Agrupamos user_id por clase: cada subreddit es un "usuario" sintético
            # para que build_splits los distribuya y no se vaya todo a train.
            "user_id": hash_id("kaggle_sdd_user", cls),
            "source": "kaggle_sdd",
            "timestamp": "",  # Kaggle SDD v14 no preserva timestamp
            "text_raw": text,
            "text_clean": text_clean,
            "label": KAGGLE_V14_LABEL[cls],
            # v14 fusiona r/depression + r/SuicideWatch bajo "suicide"
            "label_source": "subreddit_membership:suicide_or_suicidewatch",
            "lang": "en",
            "orig_split": "",
        })
    return pd.DataFrame(rows)


def project_reddit_mh_posts() -> pd.DataFrame:
    """depression.csv → solo r/depression, todos label=2."""
    src = INTERIM / "depression.csv"
    df = pd.read_csv(src)
    rows = []
    for _, r in df.iterrows():
        body = str(r.get("body") or "").strip()
        title = str(r.get("title") or "").strip()
        if body in {"[removed]", "[deleted]", "nan"}:
            body = ""
        text = "\n\n".join(t for t in [title, body] if t)
        text_clean = anonymize(text)
        if not text_clean.strip():
            continue
        author = str(r.get("author") or "")
        if author in {"", "[deleted]", "None", "nan"}:
            author = f"anon_{r.get('id', '')}"
        rows.append({
            "doc_id": hash_id("reddit_mh_posts", str(r["id"])),
            "user_id": hash_id("reddit_mh_posts_user", str(author)),
            "source": "reddit_mh_posts",
            "timestamp": _utc_iso_from_csv(r.get("created_utc")),
            "text_raw": text,
            "text_clean": text_clean,
            "label": 2,  # todos r/depression
            "label_source": "subreddit_membership:r/depression",
            "lang": "en",
            "orig_split": "",
        })
    return pd.DataFrame(rows)


def project_depression_reddit() -> pd.DataFrame:
    """depression_reddit_cleaned_ds.csv → label=1 → 2, label=0 → 0."""
    src = INTERIM / "depression_reddit_cleaned_ds.csv"
    df = pd.read_csv(src)
    rows = []
    for i, r in df.iterrows():
        text = str(r.get("text") or "")
        text_clean = anonymize(text)
        if not text_clean.strip():
            continue
        raw_label = str(int(r["labels"]))
        if raw_label not in DEPRESSION_REDDIT_LABEL:
            continue
        rows.append({
            "doc_id": hash_id("depression_reddit", str(i)),
            "user_id": hash_id("depression_reddit_user", str(i)),
            "source": "depression_reddit",
            "timestamp": "",
            "text_raw": text,
            "text_clean": text_clean,
            "label": DEPRESSION_REDDIT_LABEL[raw_label],
            "label_source": "subreddit_membership",
            "lang": "en",
            "orig_split": "",
        })
    return pd.DataFrame(rows)


def main() -> None:
    sources = {
        "kaggle_sdd": project_kaggle_sdd,
        "reddit_mh_posts": project_reddit_mh_posts,
        "depression_reddit": project_depression_reddit,
    }
    for name, fn in sources.items():
        print(f"\n=== {name} ===")
        df = fn()
        out_dir = INTERIM / name
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "data.parquet"
        df.to_parquet(out, index=False)
        print(f"  filas: {len(df):,}")
        if len(df):
            print(f"  label counts: {df['label'].value_counts().to_dict()}")
            print(f"  cols: {df.columns.tolist()}")
            print(f"  → {out}")


if __name__ == "__main__":
    main()