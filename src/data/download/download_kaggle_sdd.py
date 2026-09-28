"""Suicide and Depression Detection (Komati, Kaggle) — inglés, a traducir.

Kaggle: nikhileswarkomati/suicide-watch, VERSIÓN 13 (348k posts de Reddit,
balanceados en 3 clases: `depression`, `SuicideWatch`, `teenagers`).
La versión actual (v14) fusiona depression+SuicideWatch en `suicide`; para
una tesis de depresión conviene la v13, que conserva el subreddit.
Es el dataset que el plan de tesis menciona explícitamente ("Suicide and
Depression Detection y derivados, previa traducción").
Cita: Komati et al., "Suicide Ideation Detection in Social Media Forums"
(IEEE, 2021). Método: endpoint público de Kaggle, sin credenciales.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

from src.data.download._common import (
    download_kaggle_dataset,
    is_already_downloaded,
    make_cli,
    read_manifest,
    write_manifest,
)

SLUG = "nikhileswarkomati/suicide-watch"
VERSION = 13
SOURCE = "kaggle_sdd"
LICENSE = "CC BY-SA 4.0"


def download(target_dir: Path) -> dict[str, Any]:
    """Baja el zip de Kaggle a target_dir/kaggle_sdd.zip y genera manifest."""
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir / f"{SOURCE}.zip"
    m = read_manifest(target_dir) or {}
    if is_already_downloaded(target_dir) and zip_path.exists() and m.get("kaggle_version") == VERSION:
        return m

    sha = download_kaggle_dataset(SLUG, zip_path, version=VERSION)
    with zipfile.ZipFile(zip_path) as zf:
        n_files = len(zf.namelist())

    return write_manifest(
        target_dir=target_dir,
        source=SOURCE,
        license=LICENSE,
        sha256=sha,
        path=str(zip_path),
        n_files=n_files,
        extra={"kaggle_slug": SLUG, "kaggle_version": VERSION, "lang": "en"},
    )


if __name__ == "__main__":
    make_cli(__name__, download, "Suicide and Depression Detection v13 (Kaggle, EN)")
