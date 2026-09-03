"""SWMH-ES (SuicideWatch + MentalHealth, traducidos al español) — HF Hub.

HF: jorgeortizfuentes/swmh_es (público, sin token). Si no está disponible,
se cae a un fallback documentado.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.data.download._common import make_cli, write_manifest

SOURCE = "swmh_es"
LICENSE = "Mixed (see HF dataset card)"
CANDIDATE_HF_IDS = [
    "jorgeortizfuentes/swmh_es",
    "ajyy/SWMH-ES",
]


def download(target_dir: Path) -> dict[str, Any]:
    """Baja SWMH-ES desde HF (público, sin token)."""
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / "data.jsonl"
    last_err: Exception | None = None

    try:
        from datasets import load_dataset
    except Exception as exc:
        last_err = exc

    if last_err is None:
        for hf_id in CANDIDATE_HF_IDS:
            try:
                ds = load_dataset(hf_id, split="train", trust_remote_code=True)
                with open(out_path, "w", encoding="utf-8") as fh:
                    for row in ds:
                        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                from src.data.download._common import sha256_file

                sha = sha256_file(out_path)
                return write_manifest(
                    target_dir=target_dir,
                    source=SOURCE,
                    license=LICENSE,
                    sha256=sha,
                    path=str(out_path),
                    n_files=1,
                    extra={"hf_id": hf_id, "n_rows": len(ds)},
                )
            except Exception as exc:
                last_err = exc
                continue

    raise NotImplementedError(
        "No se pudo bajar SWMH-ES automáticamente.\n"
        f"Último error: {last_err}\n"
        "Pasos manuales:\n"
        "  1. Buscar la versión abierta en https://huggingface.co/datasets\n"
        "  2. Guardar el JSONL en: ./data/raw/swmh_es/data.jsonl\n"
        "  3. Re-correr este script."
    )


if __name__ == "__main__":
    make_cli(__name__, download, "SWMH-ES (HF, sin token)")
