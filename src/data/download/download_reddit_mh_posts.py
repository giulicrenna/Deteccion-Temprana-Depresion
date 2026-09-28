"""Reddit Mental Health Posts — inglés, a traducir.

HF: solomonk/reddit_mental_health_posts (151k posts crudos, público).
Subreddits: depression, ADHD, OCD, aspergers, ptsd. Trae `author` y
`created_utc` → permite split user-level y features temporales.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.data.download._common import download_hf_dataset, make_cli

SOURCE = "reddit_mh_posts"
LICENSE = "Sin licencia declarada en HF — datos públicos de Reddit, uso académico"
HF_ID = "solomonk/reddit_mental_health_posts"


def download(target_dir: Path) -> dict[str, Any]:
    return download_hf_dataset(target_dir, SOURCE, LICENSE, HF_ID, extra={"lang": "en"})


if __name__ == "__main__":
    make_cli(__name__, download, "Reddit Mental Health Posts (HF, EN)")
