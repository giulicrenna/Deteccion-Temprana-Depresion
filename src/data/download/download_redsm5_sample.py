"""ReDSM5 (Bao et al., CIKM 2025) — inglés, a traducir.

HF: irlab-udc/redsm5 (1.484 posts de Reddit anotados por psicólogos con
los 9 síntomas del DSM-5). Apache-2.0.
GATED con aprobación MANUAL: pedir acceso en la ficha HF (indicar uso
académico/tesis), esperar aprobación y hacer `huggingface-cli login`.
El token nunca se lee desde este código.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.data.download._common import make_cli, write_manifest

SOURCE = "redsm5_sample"
LICENSE = "Apache-2.0 (gated, aprobación manual)"
HF_ID = "irlab-udc/redsm5"


def download(target_dir: Path) -> dict[str, Any]:
    """Baja el snapshot completo del repo (CSV de posts + anotaciones)."""
    target_dir.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download  # import lazy

    try:
        local = snapshot_download(HF_ID, repo_type="dataset", local_dir=target_dir / "repo")
    except Exception as exc:
        raise NotImplementedError(
            f"{HF_ID} es gated con aprobación manual. Pasos:\n"
            f"  1. Pedir acceso en https://huggingface.co/datasets/{HF_ID}\n"
            "  2. Esperar el mail de aprobación.\n"
            "  3. `huggingface-cli login` y re-correr este script.\n"
            f"Error original: {exc}"
        ) from exc

    files = [p for p in Path(local).rglob("*") if p.is_file() and ".cache" not in p.parts]
    return write_manifest(
        target_dir=target_dir,
        source=SOURCE,
        license=LICENSE,
        sha256="(snapshot multi-archivo, ver repo/)",
        path=str(local),
        n_files=len(files),
        extra={"hf_id": HF_ID, "lang": "en"},
    )


if __name__ == "__main__":
    make_cli(__name__, download, "ReDSM5 (HF gated manual, EN)")
