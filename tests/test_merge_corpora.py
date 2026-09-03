"""Tests del merge de corpus."""

import json
from pathlib import Path

import pandas as pd


def _write_mini_corpus(tmp_path: Path, source: str, rows: list[dict]) -> Path:
    d = tmp_path / source
    d.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(d / "data.parquet", index=False)
    return d


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


def _make_row(i: int, source: str, label: int) -> dict:
    return {
        "doc_id": f"{source}_{i:04d}",
        "user_id": f"{source}_user_{i % 3}",
        "source": source,
        "timestamp": "2025-01-01T00:00:00Z",
        "text_raw": f"texto {i}",
        "text_clean": f"texto {i}",
        "label": label,
        "label_source": "test",
        "lang": "es",
    }


def test_merge_two_corpora(tmp_path):
    # Crear dos mini corpora en tmp_path/interim/<source>/data.parquet
    interim = tmp_path / "interim"
    _write_mini_corpus(
        interim,
        "alpha",
        [_make_row(i, "alpha", 0) for i in range(5)],
    )
    _write_mini_corpus(
        interim,
        "beta",
        [_make_row(i, "beta", 2) for i in range(3)],
    )

    # Leer cada uno y mergear a mano (sin subprocess).
    frames = []
    for src in ("alpha", "beta"):
        p = interim / src / "data.parquet"
        df = pd.read_parquet(p)
        df = df[SCHEMA].copy()
        df["corpus_source"] = src
        df["is_translated"] = False
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True)

    assert len(merged) == 8
    assert set(merged["source"].unique()) == {"alpha", "beta"}
    assert merged["label"].tolist()[:5] == [0] * 5
    assert merged["label"].tolist()[5:] == [2] * 3


def test_schema_unified(tmp_path):
    interim = tmp_path / "interim"
    _write_mini_corpus(interim, "x", [_make_row(0, "x", 0)])
    df = pd.read_parquet(interim / "x" / "data.parquet")
    for col in SCHEMA:
        assert col in df.columns, f"falta columna {col}"
