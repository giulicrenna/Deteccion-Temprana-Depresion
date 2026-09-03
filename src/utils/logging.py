"""Logger con formato uniforme para todos los scripts.

Uso:
    from src.utils.logging import get_logger
    log = get_logger(__name__)
    log.info("Hola")
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Devuelve un logger configurado con formato uniforme."""
    logger = logging.getLogger(name if name else "root")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


__all__ = ["get_logger"]
