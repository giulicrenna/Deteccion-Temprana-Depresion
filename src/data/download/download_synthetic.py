"""Copiador del mini-corpus sintético al directorio de raw.

El JSONL vive en `src/data/synthetic/sample.jsonl` (versionado en git).
Este script lo copia a `data/raw/synthetic/data.jsonl` y genera el
manifest. Es un NO-OP de red — siempre funciona, sin auth, sin HF.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from src.data.download._common import (
    is_already_downloaded,
    make_cli,
    sha256_file,
    write_manifest,
)
from src.utils.seeds import set_seed

SOURCE = "synthetic"
LICENSE = "Generated for pipeline testing only — no scientific use"
SYNTH_PATH = Path(__file__).resolve().parent.parent / "synthetic" / "sample.jsonl"


def download(target_dir: Path) -> dict[str, Any]:
    """Copia el JSONL sintético a target_dir y genera el manifest."""
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / "data.jsonl"

    if is_already_downloaded(target_dir) and out_path.exists():
        from src.data.download._common import read_manifest

        m = read_manifest(target_dir) or {}
        if m:
            return m

    if not SYNTH_PATH.exists():
        raise FileNotFoundError(
            f"No se encuentra el corpus sintético en {SYNTH_PATH}. "
            "Restaurar el archivo desde git."
        )

    set_seed(42)
    shutil.copy2(SYNTH_PATH, out_path)
    sha = sha256_file(out_path)
    n_rows = sum(1 for _ in open(out_path, encoding="utf-8"))
    return write_manifest(
        target_dir=target_dir,
        source=SOURCE,
        license=LICENSE,
        sha256=sha,
        path=str(out_path),
        n_files=1,
        extra={
            "n_rows": n_rows,
            "note": "mini-corpus de desarrollo — NO usar para entrenar",
        },
    )


if __name__ == "__main__":
    make_cli(__name__, download, "Mini-corpus sintético (desarrollo)")
