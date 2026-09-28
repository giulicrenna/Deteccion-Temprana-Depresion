"""Tests de anonimización."""

from src.data.make_dataset import anonymize


def test_removes_url():
    assert "http" not in anonymize("mirá esto https://ejemplo.com/foo y eso")
    assert "www" not in anonymize("visit www.google.com hoy")


def test_removes_mention():
    assert "@pepe" not in anonymize("hola @pepe como andás")


def test_removes_email():
    out = anonymize("contactame a juan.perez@gmail.com porfa")
    assert "@" not in out
    assert "gmail" not in out
    assert "juan" not in out  # antes quedaba "juan.perez.com"


def test_removes_reddit_user():
    out = anonymize("thanks u/some_user and /u/Other-One, see r/depression")
    assert "some_user" not in out
    assert "Other-One" not in out
    assert "r/depression" in out  # los subreddits no son PII


def test_removes_phone():
    out = anonymize("llamame al +54 11 4321 5678 o al 115-123-4567")
    assert "+54" not in out
    assert "4321" not in out


def test_normalizes_whitespace():
    out = anonymize("hola    mundo\n\ncruel")
    assert "  " not in out
    assert "\n" not in out


def test_handles_empty():
    assert anonymize("") == ""
    assert anonymize(None) == ""


def test_keeps_words():
    out = anonymize("estoy muy triste @juan https://x.com")
    assert "estoy" in out
    assert "muy" in out
    assert "triste" in out
