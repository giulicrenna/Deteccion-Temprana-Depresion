"""SWMH (Ji et al., 2021 — autores de MentalBERT) — inglés, a traducir.

HF: AIMH/SWMH (54k posts; r/depression, r/SuicideWatch, r/Anxiety,
r/bipolar, r/offmychest). CC BY-NC 4.0.
GATED (aprobación automática): aceptar condiciones en la ficha HF y hacer
`huggingface-cli login`. El token nunca se lee desde este código.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.data.download._common import download_hf_dataset, make_cli

SOURCE = "swmh"
LICENSE = "CC BY-NC 4.0 (gated, aprobación automática)"
HF_ID = "AIMH/SWMH"


def download(target_dir: Path) -> dict[str, Any]:
    return download_hf_dataset(
        target_dir, SOURCE, LICENSE, HF_ID, gated=True, extra={"lang": "en"}
    )


if __name__ == "__main__":
    make_cli(__name__, download, "SWMH (HF gated, EN)")
