"""Tests de reproducibilidad para set_seed."""

import random

import numpy as np


def test_set_seed_random_reproducible():
    from src.utils.seeds import set_seed

    set_seed(42)
    a = [random.random() for _ in range(10)]
    set_seed(42)
    b = [random.random() for _ in range(10)]
    assert a == b, "set_seed(42) no es reproducible para random"


def test_set_seed_numpy_reproducible():
    from src.utils.seeds import set_seed

    set_seed(42)
    a = np.random.rand(10).tolist()
    set_seed(42)
    b = np.random.rand(10).tolist()
    assert a == b, "set_seed(42) no es reproducible para numpy"


def test_set_seed_returns_seed():
    from src.utils.seeds import set_seed

    assert set_seed(42) == 42
    assert set_seed(123) == 123


def test_set_seed_idempotent():
    from src.utils.seeds import set_seed

    set_seed(42)
    set_seed(42)
    set_seed(42)
    # Si rompió en algún re-set, fallaría arriba; si llegó acá, OK.
    assert True
