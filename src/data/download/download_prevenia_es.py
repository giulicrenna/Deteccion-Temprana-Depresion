"""PrevenIA spanish-suicide-intent — YA EN ESPAÑOL (no requiere traducción).

HF: PrevenIA/spanish-suicide-intent (189k textos, CC BY 4.0, público).
Combina tweets/posts en español y traducciones de corpus en inglés
(incluye ~81k posts de r/SuicideWatch traducidos). Columna `dataset`
indica la fuente original de cada fila.
Constructo: intención suicida (no depresión) → no se mergea por default.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.data.download._common import download_hf_dataset, make_cli

SOURCE = "prevenia_es"
LICENSE = "CC BY 4.0"
HF_ID = "PrevenIA/spanish-suicide-intent"


def download(target_dir: Path) -> dict[str, Any]:
    return download_hf_dataset(target_dir, SOURCE, LICENSE, HF_ID, extra={"lang": "es"})


if __name__ == "__main__":
    make_cli(__name__, download, "PrevenIA spanish-suicide-intent (HF, ES)")
