"""Seteo centralizado de seeds para reproducibilidad.

Funciones:
    set_seed(seed=42): fija random, numpy, torch (CPU+CUDA), transformers.
"""

from __future__ import annotations

import os
import random


def set_seed(seed: int = 42) -> int:
    """Fija el seed global para random, numpy, torch y transformers.

    Idempotente: llamar más de una vez no rompe.
    Devuelve el seed efectivamente aplicado.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
    except Exception:
        pass

    try:
        from transformers import set_seed as _hf_set_seed

        _hf_set_seed(seed)
    except Exception:
        pass

    return seed


__all__ = ["set_seed"]
