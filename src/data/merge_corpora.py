"""Une todos los `interim/<fuente>/data.parquet` en un corpus único."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

from src.utils.logging import get_logger
from src.utils.seeds import set_seed

log = get_logger(__name__)


SCHEMA = [
    "doc_id",
    "user_id",
    "source",
    "timestamp",
    "text_raw",
    "text_clean",
    "label",
    "label_source",
    "lang",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="interim", type=Path, default=Path("./data/interim"))
    parser.add_argument("--out", dest="processed", type=Path, default=Path("./data/processed"))
    parser.add_argument("--config", type=Path, default=Path("./configs/data.yaml"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = yaml.safe_load(args.config.read_text())

    args.processed.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    for src, meta in cfg["data"]["sources"].items():
        if not meta.get("enabled", False):
            continue
        p = args.interim / src / "data.parquet"
        if not p.exists():
            log.warning("no existe %s — saltando", src)
            continue
        df = pd.read_parquet(p)
        # Asegurar el esquema.
        for col in SCHEMA:
            if col not in df.columns:
                df[col] = "" if col != "label" else 0
        df = df[SCHEMA].copy()
        df["corpus_source"] = src
        df["is_translated"] = src in {"swmh_es", "redsm5_sample"}
        frames.append(df)
        log.info("cargado %s : %d filas", src, len(df))

    if not frames:
        raise SystemExit("No hay interim/<fuente>/data.parquet para mergear.")

    corpus = pd.concat(frames, ignore_index=True)
    # Deduplicar por doc_id.
    before = len(corpus)
    corpus = corpus.drop_duplicates(subset=["doc_id"]).reset_index(drop=True)
    log.info("dedup: %d → %d", before, len(corpus))

    out_path = args.processed / "corpus_v1.parquet"
    corpus.to_parquet(out_path, index=False)
    log.info("→ %s : %d filas", out_path, len(corpus))

    # Estadísticas.
    stats: dict = {
        "n_rows": int(len(corpus)),
        "n_users": int(corpus["user_id"].nunique()),
        "by_source": corpus["source"].value_counts().to_dict(),
        "by_label": corpus["label"].value_counts().to_dict(),
        "by_source_label": (
            corpus.groupby(["source", "label"]).size().unstack(fill_value=0).to_dict()
        ),
    }
    (args.processed / "corpus_statistics.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False)
    )
    log.info("→ corpus_statistics.json : %s", stats)


if __name__ == "__main__":
    main()
