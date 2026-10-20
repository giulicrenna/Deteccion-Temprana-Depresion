"""Tests del enmascarado de marcadores diatópicos y del control aleatorio."""

import numpy as np
import pytest

from src.evaluation.dialect_control import random_mask_texts
from src.features.dialect_masking import MASK_TOKEN, build_regex, load_markers, mask_texts


@pytest.fixture(scope="module")
def markers():
    return load_markers()


def test_masks_whole_words_case_insensitive(markers):
    regex = build_regex(markers, ["dialectal"])
    masked, counts = mask_texts(["Joder, WEY, vos sabés", "vosotros"], regex)
    assert masked[0] == f"{MASK_TOKEN}, {MASK_TOKEN}, {MASK_TOKEN} {MASK_TOKEN}"
    assert masked[1] == MASK_TOKEN
    assert counts == [4, 1]


def test_does_not_mask_inside_words(markers):
    regex = build_regex(markers, list(markers))
    text = "Los pobres nosotros reos tenemos postales"
    masked, counts = mask_texts([text], regex)
    assert masked == [text]
    assert counts == [0]


def test_multiword_terms_win_over_parts(markers):
    regex = build_regex(markers, ["toponimos"])
    masked, counts = mask_texts(["Fui al Real Madrid"], regex)
    assert masked == [f"Fui al {MASK_TOKEN}"]
    assert counts == [1]


def test_vosotros_verb_pattern(markers):
    regex = build_regex(markers, ["dialectal"])
    masked, _ = mask_texts(["¿Tenéis hambre? Sabéis que sí"], regex)
    assert masked == [f"¿{MASK_TOKEN} hambre? {MASK_TOKEN} que sí"]


def test_categories_are_independent(markers):
    text = ["Piñera en Puebla, wey"]
    assert mask_texts(text, build_regex(markers, ["politicos"]))[1] == [1]
    assert mask_texts(text, build_regex(markers, ["toponimos"]))[1] == [1]
    assert mask_texts(text, build_regex(markers, ["dialectal"]))[1] == [1]


def test_random_control_matches_mask_count_and_spares_markers(markers):
    regex = build_regex(markers, list(markers))
    texts = ["wey hoy me siento muy triste y solo", "nada que ver", ""]
    _, counts = mask_texts(texts, regex)
    out = random_mask_texts(texts, regex, counts, seed=0)
    assert out[0].count(MASK_TOKEN) == counts[0] == 1
    assert out[0].startswith("wey ")
    assert out[1:] == texts[1:]
    assert random_mask_texts(texts, regex, counts, seed=0) == out
    assert np.sum([t.count(MASK_TOKEN) for t in out]) == sum(counts)
