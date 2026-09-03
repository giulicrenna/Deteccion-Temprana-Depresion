"""Polaridad / sentimiento en español.

Backend principal: NLTK VADER (SentimentIntensityAnalyzer). Como VADER
está entrenado sobre inglés, se usa como primera aproximación — la
calidad baja en español es una limitación documentada.

Fallback opcional: lexicon-based español (src/features/lexicons/polarity_es.csv).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)


def _lexicon_polarity(texts: list[str], lexicon_path: Path) -> pd.DataFrame:
    """Fallback: suma de polaridades de un lexicon en español."""
    if not lexicon_path.exists():
        log.warning("no hay lexicon %s — devolviendo ceros", lexicon_path)
        return pd.DataFrame(
            {"polarity": [0.0] * len(texts), "positives": [0] * len(texts), "negatives": [0] * len(texts)}
        )
    import csv

    pos, neg = set(), set()
    with open(lexicon_path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            (pos if row["polarity"] == "positive" else neg).add(row["term"].lower())

    rows = []
    for t in texts:
        toks = (t or "").lower().split()
        p = sum(1 for w in toks if w in pos)
        n = sum(1 for w in toks if w in neg)
        rows.append(
            {"polarity": (p - n) / max(1, p + n), "positives": p, "negatives": n}
        )
    return pd.DataFrame(rows)


def score(texts: list[str], lexicon_path: Path | None = None) -> pd.DataFrame:
    """Devuelve un DataFrame con columnas `compound`, `pos`, `neu`, `neg`."""
    try:
        import nltk
        from nltk.sentiment import SentimentIntensityAnalyzer

        try:
            sia = SentimentIntensityAnalyzer()
        except LookupError:
            log.info("descargando vader_lexicon...")
            nltk.download("vader_lexicon", quiet=True)
            sia = SentimentIntensityAnalyzer()
        rows = [sia.polarity_scores(t or "") for t in texts]
        df = pd.DataFrame(rows).rename(columns={"compound": "polarity"})
        return df
    except Exception as exc:
        log.warning("VADER no disponible (%s) — usando lexicon fallback", exc)
        return _lexicon_polarity(texts, lexicon_path or Path("src/features/lexicons/polarity_es.csv"))


__all__ = ["score"]
