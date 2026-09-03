"""Descarga el corpus Coello-Guilarte et al. (2019) desde INAOE.

Fuente: https://ccc.inaoep.mx/~mmontesg/resources/CrossLingualDepression.zip
Tipo: tweets en español etiquetados como depresivos vs no-depresivos.
Método: HTTP plano con urllib (sin auth).
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

from src.data.download._common import (
    download_file,
    is_already_downloaded,
    make_cli,
    sha256_file,
    write_manifest,
)

URL = "https://ccc.inaoep.mx/~mmontesg/resources/CrossLingualDepression.zip"
SOURCE = "coello_guilarte"
LICENSE = "Research use; cite Coello-Guilarte et al. (2019)"


def download(target_dir: Path) -> dict[str, Any]:
    """Baja el zip a target_dir/coello_guilarte.zip, valida y genera manifest."""
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir / f"{SOURCE}.zip"

    if is_already_downloaded(target_dir) and zip_path.exists():
        from src.data.download._common import read_manifest

        m = read_manifest(target_dir) or {}
        return m

    sha = download_file(URL, zip_path)
    n_files = 0
    with zipfile.ZipFile(zip_path) as zf:
        n_files = len(zf.namelist())

    return write_manifest(
        target_dir=target_dir,
        source=SOURCE,
        license=LICENSE,
        sha256=sha,
        path=str(zip_path),
        n_files=n_files,
        extra={"url": URL},
    )


if __name__ == "__main__":
    make_cli(__name__, download, "Coello-Guilarte 2019 — CrossLingualDepression")
