"""Leis et al. 2019 corpus completo — STUB gated.

El corpus está en Kaggle (requiere API key) o se puede pedir por mail a:
  - Francesco Ronzano <francesco.ronzano@upf.edu>
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.data.download._common import make_cli

SOURCE = "leis"


def download(target_dir: Path) -> dict[str, Any]:
    """Stub — el corpus requiere gestión manual."""
    raise NotImplementedError(
        "El corpus Leis et al. (2019) está solo en Kaggle (requiere API key) o por mail.\n"
        "Recomendación: usar Coello-Guilarte 2019 como sustituto (ya disponible, wget).\n"
        "Si querés el Leis original:\n"
        "  1. Email a Francesco Ronzano <francesco.ronzano@upf.edu>\n"
        "     Asunto: 'Solicitud corpus depression tweets — Tesis UGR'\n"
        "  2. O bajar de Kaggle: https://www.kaggle.com/datasets/francescoronzano/\n"
        "  3. Guardar el .zip en: ./data/raw/leis/\n"
        "  4. Re-correr este script."
    )


if __name__ == "__main__":
    make_cli(__name__, download, "Leis et al. 2019 (stub gated)")
