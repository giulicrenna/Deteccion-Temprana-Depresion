"""Enmascarado de marcadores diatópicos (dialectismos, topónimos y figuras políticas).

Reemplaza cada coincidencia por `[UNK]`: TF-IDF lo descarta porque `unk` no está en su
vocabulario, y el tokenizador de BETO lo reconoce como su token desconocido. Así el texto
enmascarado no introduce una señal nueva en ninguno de los dos tipos de modelo.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

MASK_TOKEN = "[UNK]"
DEFAULT_MARKERS = Path(__file__).parent / "lexicons" / "dialect_markers.yaml"


def load_markers(path: Path | str = DEFAULT_MARKERS) -> dict[str, dict[str, list[str]]]:
    """Lee el YAML de marcadores: categoría → {'terms': [...], 'patterns': [...]}."""
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return {
        cat: {"terms": list(v.get("terms") or []), "patterns": list(v.get("patterns") or [])}
        for cat, v in raw.items()
    }


def build_regex(markers: dict[str, dict[str, list[str]]], categories: list[str]) -> re.Pattern:
    """Regex de palabra completa que une los marcadores de las categorías pedidas."""
    terms, patterns = set(), []
    for cat in categories:
        terms.update(t.lower() for t in markers[cat]["terms"])
        patterns.extend(markers[cat]["patterns"])
    # Los términos más largos primero, para que "real madrid" gane sobre "madrid".
    alternatives = [re.escape(t) for t in sorted(terms, key=len, reverse=True)] + patterns
    return re.compile(r"(?<!\w)(?:" + "|".join(alternatives) + r")(?!\w)", re.IGNORECASE)


def mask_texts(texts: list[str], regex: re.Pattern) -> tuple[list[str], list[int]]:
    """Enmascara cada texto y devuelve también la cantidad de reemplazos por texto."""
    masked, counts = [], []
    for text in texts:
        new_text, n = regex.subn(MASK_TOKEN, text or "")
        masked.append(new_text)
        counts.append(n)
    return masked, counts


__all__ = ["MASK_TOKEN", "DEFAULT_MARKERS", "build_regex", "load_markers", "mask_texts"]
