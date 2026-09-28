"""Tests del módulo de traducción y del join de traducciones en el merge."""

import pandas as pd

from src.data.merge_corpora import attach_translation
from src.translation.translate import qc_flags, split_segments, stratified_sample


def test_stratified_sample_keeps_proportions():
    df = pd.DataFrame({"doc_id": [str(i) for i in range(1000)], "label": [0] * 800 + [2] * 200})
    s = stratified_sample(df, 100, seed=42)
    assert len(s) == 100
    assert s["label"].value_counts().to_dict() == {0: 80, 2: 20}
    assert s.equals(stratified_sample(df, 100, seed=42))  # determinístico


def test_split_segments_caps_length():
    segs = split_segments("One. Two!\n" + " ".join(["w"] * 250), max_words=100)
    assert segs[:2] == ["One.", "Two!"]
    assert all(len(s.split()) <= 100 for s in segs)


def test_qc_flags():
    assert qc_flags("I feel sad", "Me siento triste") == []
    assert "empty" in qc_flags("I feel sad", "")
    assert "possible_refusal_or_added_note" in qc_flags(
        "I want to die", "Lo siento, pero no puedo ayudar con eso."
    )
    assert "english_residue" in qc_flags("I feel sad and alone", "I feel sad and you are alone")


def test_attach_translation_keeps_original():
    df = pd.DataFrame({"doc_id": ["a", "b"], "text_clean": ["hello", "bye"], "lang": ["en", "en"]})
    tr = pd.DataFrame({"doc_id": ["a"], "text_es": ["hola"], "mt_system": ["x:y"]})
    out = attach_translation(df, tr)
    assert len(out) == 1  # sólo filas traducidas
    row = out.iloc[0]
    assert row["text_clean"] == "hola" and row["lang"] == "es"
    assert row["text_orig"] == "hello" and row["lang_orig"] == "en"
    assert bool(row["is_translated"]) and row["mt_system"] == "x:y"
