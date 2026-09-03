"""DAIC-WOZ — STUB gated.

Dataset de entrevistas clínicas con PHQ-8.
Acceso: https://dcapswoz.ict.usc.edu — requiere DUA firmado.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.data.download._common import make_cli

SOURCE = "daicwoz"


def download(target_dir: Path) -> dict[str, Any]:
    """Stub — DAIC-WOZ requiere DUA."""
    raise NotImplementedError(
        "DAIC-WOZ requiere Data Usage Agreement firmado.\n"
        "Pasos:\n"
        "  1. Solicitar acceso en: https://dcapswoz.ict.usc.edu\n"
        "  2. Firmar el DUA.\n"
        "  3. Bajar los archives a: ./data/raw/daicwoz/\n"
        "  4. Re-correr este script."
    )


if __name__ == "__main__":
    make_cli(__name__, download, "DAIC-WOZ (stub gated)")
