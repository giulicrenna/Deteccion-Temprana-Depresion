"""Mental health en español desde Figshare — STUB.

URL candidata: https://figshare.com/articles/dataset/28498766
Estado: Figshare aplica un WAF que challenge-a wget/curl directos.
Para bajar hay que usar un browser (copiar el link de descarga) o
automatizar con Playwright (no soportado acá por restricción de tokens).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.data.download._common import make_cli

SOURCE = "figshare_mh_es"


def download(target_dir: Path) -> dict[str, Any]:
    """Stub — Figshare WAF bloquea wget/curl directos."""
    raise NotImplementedError(
        "Figshare aplica un WAF (challenge JS) que bloquea wget/curl directos.\n"
        "Pasos manuales:\n"
        "  1. Abrir en el browser: https://figshare.com/articles/dataset/28498766\n"
        "  2. Click en 'Download' y guardar el zip en:\n"
        "     ./data/raw/figshare_mh_es/figshare_mh_es.zip\n"
        "  3. Re-correr este script — generará el manifest."
    )


if __name__ == "__main__":
    make_cli(__name__, download, "Figshare mental health ES (stub por WAF)")
