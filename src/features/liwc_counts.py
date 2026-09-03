"""Conteo de marcadores LIWC en español (basado en Leis et al.)."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)


DEFAULT_LEXICON: dict[str, list[str]] = {
    "first_person_singular": [
        "yo", "mi", "mí", "mismo", "misma", "conmigo", "mis",
    ],
    "absolutist": [
        "siempre", "nunca", "nada", "todo", "jamás", "absolutamente",
        "completamente", "totalmente", "nadie",
    ],
    "negative_emotion": [
        "triste", "tristeza", "deprimido", "deprimida", "ansioso", "ansiosa",
        "miedo", "sola", "solo", "vacío", "vacía", "dolor", "angustia",
        "lloro", "llorando", "cansado", "cansada", "agotado", "agotada",
        "horrible", "terrible", "mal", "peor", "odio",
    ],
    "positive_emotion": [
        "feliz", "felicidad", "contento", "contenta", "alegre", "alegría",
        "amor", "amo", "encanta", "genial", "increíble", "hermoso", "hermosa",
        "disfruto", "disfruté", "mejor", "bien", "bueno", "buena",
    ],
}


_word_re = re.compile(r"\b\w+\b", re.UNICODE)


def _load_lexicon(path: Path | None) -> dict[str, list[str]]:
    if path is None or not path.exists():
        log.warning("no hay lexicon en %s — usando DEFAULT_LEXICON", path)
        return DEFAULT_LEXICON
    import csv

    out: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            cat = row["category"]
            out.setdefault(cat, []).append(row["term"].strip().lower())
    return out


def count_markers(texts: list[str], lexicon_path: Path | None = None) -> pd.DataFrame:
    """Devuelve un DataFrame con counts normalizados por #tokens."""
    lex = _load_lexicon(lexicon_path)
    rows = []
    for t in texts:
        tokens = [w.lower() for w in _word_re.findall(t or "")]
        n = max(1, len(tokens))
        row = {"n_tokens": n}
        for cat, words in lex.items():
            count = sum(1 for tok in tokens if tok in words)
            row[f"{cat}_count"] = count
            row[f"{cat}_norm"] = count / n
        rows.append(row)
    return pd.DataFrame(rows).fillna(0.0)


__all__ = ["count_markers", "DEFAULT_LEXICON"]
