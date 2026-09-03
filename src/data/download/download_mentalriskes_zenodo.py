"""MentalRiskES vía Zenodo — STUB.

ADVERTENCIA: el record Zenodo 8055604 corresponde a PRECOM-SM (no a
MentalRiskES). Este script queda como stub para cuando los autores
publiquen la versión abierta del corpus.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.data.download._common import make_cli

SOURCE = "mentalriskes_zenodo"


def download(target_dir: Path) -> dict[str, Any]:
    """Stub — no implementa descarga real."""
    raise NotImplementedError(
        "El record Zenodo 8055604 corresponde a PRECOM-SM, no a MentalRiskES.\n"
        "Para el corpus MentalRiskES usar `download_mentalriskes_github` o "
        "`download_mentalriskes_full` (ambos gated)."
    )


if __name__ == "__main__":
    make_cli(__name__, download, "MentalRiskES Zenodo (stub)")
