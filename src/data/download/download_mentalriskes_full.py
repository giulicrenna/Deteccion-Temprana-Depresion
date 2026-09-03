"""MentalRiskES corpus completo (45k mensajes) — STUB gated.

El corpus completo requiere acuerdo con los autores:
  - amarmol@ujaen.es  (Ana Martín-Maldonado)
  - amontejo@ujaen.es  (Ángel Montejo-Ráez)
El zip de GitHub está cifrado con contraseña.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.data.download._common import make_cli

SOURCE = "mentalriskes_full"


def download(target_dir: Path) -> dict[str, Any]:
    """Stub — el corpus completo requiere acceso formal."""
    raise NotImplementedError(
        "El corpus MentalRiskES completo (45k mensajes) requiere acuerdo formal.\n"
        "Pasos:\n"
        "  1. Email a amarmol@ujaen.es y amontejo@ujaen.es con asunto:\n"
        "     'Solicitud acceso corpus MentalRiskES — Tesis UGR'\n"
        "  2. Adjuntar: afiliación, director de tesis, resumen del proyecto.\n"
        "  3. Firmar DUA si lo piden.\n"
        "  4. Bajar el .zip recibido a: ./data/raw/mentalriskes_full/\n"
        "  5. Re-correr este script — generará el manifest."
    )


if __name__ == "__main__":
    make_cli(__name__, download, "MentalRiskES corpus completo (stub gated)")
