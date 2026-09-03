"""MentalRiskES (muestra GitHub) — gated con contraseña.

El zip en GitHub (corpusMentalRiskES.zip) está CIFRADO con contraseña.
Los autores la entregan tras contactarlos a:
  - amarmol@ujaen.es  (Ana Martín-Maldonado)
  - amontejo@ujaen.es  (Ángel Montejo-Ráez)
Formulario de solicitud: https://docs.google.com/forms/d/e/1FAIpQLSfASdCzvR6DCWpFXb4eDpF6gh7CjhJKT1bH2l-SdCcxTP7l7Q/viewform

Este script:
  1. Si ya hay un archivo `mentalriskes_github.zip` en target_dir, lo detecta
     y genera el manifest sin re-descargar.
  2. Si no, intenta bajar con HTTP plano. Si el zip viene cifrado (caso
     esperado), no se puede procesar automáticamente → raise NotImplementedError
     con instrucciones claras.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.data.download._common import (
    is_already_downloaded,
    make_cli,
    write_manifest,
)

URL = "https://github.com/sinai-uja/corpusMentalRiskES/raw/main/corpusMentalRiskES.zip"
SOURCE = "mentalriskes_github"
LICENSE = "Gated — contactar autores (amarmol@ujaen.es / amontejo@ujaen.es)"


def _instructions() -> str:
    return (
        "El corpus MentalRiskES (muestra de GitHub) está CIFRADO con contraseña.\n"
        "Pasos para obtenerlo:\n"
        "  1. Llenar el formulario: "
        "https://docs.google.com/forms/d/e/1FAIpQLSfASdCzvR6DCWpFXb4eDpF6gh7CjhJKT1bH2l-SdCcxTP7l7Q/viewform\n"
        "  2. Esperar mail de los autores con la contraseña.\n"
        "  3. Bajar manualmente el .zip a: ./data/raw/mentalriskes_github/mentalriskes_github.zip\n"
        "  4. Re-correr este script — detectará el archivo y generará el manifest."
    )


def download(target_dir: Path) -> dict[str, Any]:
    """Baja o detecta el zip cifrado. Genera manifest si ya está local."""
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir / f"{SOURCE}.zip"

    # Si el archivo ya está bajado manualmente, manifest y fuera.
    if zip_path.exists() and zip_path.stat().st_size > 0:
        from src.data.download._common import read_manifest, sha256_file

        if not is_already_downloaded(target_dir):
            sha = sha256_file(zip_path)
            return write_manifest(
                target_dir=target_dir,
                source=SOURCE,
                license=LICENSE,
                sha256=sha,
                path=str(zip_path),
                n_files=1,
                extra={"url": URL, "status": "local_copy_encrypted"},
            )
        return read_manifest(target_dir) or {}

    # No está local. No intentamos descarga porque el zip está cifrado y no
    # podemos procesarlo sin la contraseña.
    raise NotImplementedError(_instructions())


if __name__ == "__main__":
    # Permitir que se importe sin ejecutar; gateamos solo en CLI.
    if os.environ.get("MENTALRISKES_PASSPHRASE"):
        print(
            "WARNING: variable MENTALRISKES_PASSPHRASE seteada pero el script no la usa. "
            "El zip está cifrado y requiere flujo manual."
        )
    make_cli(__name__, download, "MentalRiskES muestra GitHub (cifrado — ver instrucciones)")
