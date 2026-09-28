"""Depression: Reddit Dataset (Cleaned) — inglés, a traducir.

HF: mrjunos/depression-reddit-cleaned (7.7k posts, CC BY 4.0, público).
Ojo: el texto viene en minúsculas y sin puntuación → la traducción pierde
calidad; usar como complemento, no como fuente principal.
Se baja el CSV directo: el repo trae un loading script legacy que
`datasets` no logra ejecutar.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.download._common import (
    download_file,
    is_already_downloaded,
    make_cli,
    read_manifest,
    sha256_file,
    write_manifest,
)

SOURCE = "depression_reddit"
LICENSE = "CC BY 4.0"
HF_ID = "mrjunos/depression-reddit-cleaned"
URL = f"https://huggingface.co/datasets/{HF_ID}/resolve/main/depression_reddit_cleaned_ds.csv"


def download(target_dir: Path) -> dict[str, Any]:
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / "data.jsonl"
    if is_already_downloaded(target_dir) and out_path.exists():
        return read_manifest(target_dir) or {}

    csv_path = target_dir / "depression_reddit_cleaned_ds.csv"
    download_file(URL, csv_path)
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"labels": "label"})
    with open(out_path, "w", encoding="utf-8") as fh:
        for row in df.to_dict(orient="records"):
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    return write_manifest(
        target_dir=target_dir,
        source=SOURCE,
        license=LICENSE,
        sha256=sha256_file(out_path),
        path=str(out_path),
        n_files=1,
        extra={"hf_id": HF_ID, "url": URL, "n_rows": len(df), "lang": "en"},
    )


if __name__ == "__main__":
    make_cli(__name__, download, "Depression Reddit Cleaned (HF, EN)")
