"""RSDD (Risk Suicide Depression Dataset) — N/AV según Bucuram 2025.

El corpus ya no se distribuye. Queda como stub documentado.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.data.download._common import make_cli

SOURCE = "rsdd"


def download(target_dir: Path) -> dict[str, Any]:
    """Stub — RSDD ya no se distribuye."""
    raise NotImplementedError(
        "RSDD (Bucuram 2025) ya no se distribuye — N/AV.\n"
        "Ver tesis: 'no longer available'.\n"
        "Si aparece una nueva versión, actualizar este script."
    )


if __name__ == "__main__":
    make_cli(__name__, download, "RSDD (N/AV — no longer available)")
