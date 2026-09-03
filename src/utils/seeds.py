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
    # 1) Variables de entorno que afectan a las estructuras hash de Python
    #    y a las semillas de torch/transformers antes de cualquier import lazy.
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    # 2) random + numpy (disponibles casi siempre).
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    # 3) torch — opcional. Importamos lazy para no romper entornos sin GPU.
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Algunas ops de cuDNN no son deterministas; las apagamos si podemos.
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
    except Exception:
        pass

    # 4) transformers — setea el seed de su generador interno.
    try:
        from transformers import set_seed as _hf_set_seed

        _hf_set_seed(seed)
    except Exception:
        pass

    return seed


__all__ = ["set_seed"]
