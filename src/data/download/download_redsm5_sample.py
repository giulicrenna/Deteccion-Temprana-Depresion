"""ReDSM5 paraphrase sample — dataset público en HF Hub.

HF: irlab-udc/redsm5, config redsm5-sample (25 entries, MIT).
No requiere token. Usa `datasets.load_dataset` (público).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.data.download._common import make_cli, write_manifest

SOURCE = "redsm5_sample"
LICENSE = "MIT"
HF_ID = "irlab-udc/redsm5"
HF_CONFIG = "redsm5-sample"


def download(target_dir: Path) -> dict[str, Any]:
    """Baja la muestra paraphrase de ReDSM5 desde Hugging Face (público, sin token)."""
    target_dir.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset  # import lazy

    ds = load_dataset(HF_ID, HF_CONFIG, split="train", trust_remote_code=True)
    out_path = target_dir / "data.jsonl"
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
        extra={"hf_id": HF_ID, "hf_config": HF_CONFIG, "n_rows": len(ds)},
    )


if __name__ == "__main__":
    make_cli(__name__, download, "ReDSM5 paraphrase sample (HF, sin token)")
