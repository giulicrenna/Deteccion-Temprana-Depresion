"""Tests del bootstrap por usuario y de las métricas binarias."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    recall_score,
    roc_auc_score,
)

from src.evaluation.bootstrap import (
    METRICS,
    UserBootstrap,
    paired_difference,
    point_metrics,
    summarize,
)


@pytest.fixture
def toy_predictions():
    """Predicciones sintéticas con usuarios de varios documentos y empates en el score."""
    rng = np.random.default_rng(0)
    n_users = 30
    user_label = np.where(np.arange(n_users) < 10, 2, 0)
    sizes = rng.integers(3, 20, size=n_users)
    user_id = np.repeat([f"u{i}" for i in range(n_users)], sizes)
    label = np.repeat(user_label, sizes)
    score = np.round(np.clip(0.3 * (label == 2) + rng.normal(0.4, 0.2, len(label)), 0, 1), 1)
    pred = np.where(score >= 0.5, 2, 0)
    return user_id, label, pred, score


def test_point_metrics_match_sklearn(toy_predictions):
    _, label, pred, score = toy_predictions
    pm = point_metrics(label, pred, score)
    assert pm["accuracy"] == pytest.approx(accuracy_score(label, pred))
    assert pm["f1_macro"] == pytest.approx(f1_score(label, pred, average="macro"))
    assert pm["f1_dep"] == pytest.approx(f1_score(label, pred, pos_label=2))
    assert pm["recall_dep"] == pytest.approx(recall_score(label, pred, pos_label=2))
    assert pm["specificity"] == pytest.approx(recall_score(label, pred, pos_label=0))
    assert pm["kappa"] == pytest.approx(cohen_kappa_score(label, pred))
    assert pm["auc"] == pytest.approx(roc_auc_score(label == 2, score))


def test_unit_multiplicity_reproduces_point(toy_predictions):
    user_id, label, pred, score = toy_predictions
    boot = UserBootstrap(user_id, label, n_boot=3, seed=0)
    boot.multiplicity[:] = 1
    reps = boot.replicates(pred, score)
    pm = point_metrics(label, pred, score)
    for m in METRICS:
        assert reps[m].to_numpy() == pytest.approx(np.full(3, pm[m]))


def test_weighted_replicate_matches_row_expansion(toy_predictions):
    user_id, label, pred, score = toy_predictions
    boot = UserBootstrap(user_id, label, n_boot=5, seed=1)
    reps = boot.replicates(pred, score)
    rows = np.repeat(np.arange(len(label)), boot.multiplicity[2][boot.user_codes].astype(int))
    expected = point_metrics(label[rows], pred[rows], score[rows])
    for m in METRICS:
        assert reps.loc[2, m] == pytest.approx(expected[m])


def test_bootstrap_is_deterministic_and_stratified(toy_predictions):
    user_id, label, _, _ = toy_predictions
    a = UserBootstrap(user_id, label, n_boot=20, seed=7)
    b = UserBootstrap(user_id, label, n_boot=20, seed=7)
    np.testing.assert_array_equal(a.multiplicity, b.multiplicity)
    dep_users = pd.Series(label).groupby(a.user_codes).first().to_numpy() == 2
    assert (a.multiplicity[:, dep_users].sum(axis=1) == dep_users.sum()).all()
    assert (a.multiplicity.sum(axis=1) == a.n_users).all()


def test_summary_and_paired_difference(toy_predictions):
    user_id, label, pred, score = toy_predictions
    boot = UserBootstrap(user_id, label, n_boot=200, seed=0)
    pm = point_metrics(label, pred, score)
    reps = boot.replicates(pred, score)
    for row in summarize(pm, reps):
        assert row["ci_low"] <= row["ci_high"]
    worse_pred = np.where(np.arange(len(pred)) % 3 == 0, 0, pred)
    pw = point_metrics(label, worse_pred, score)
    diffs = paired_difference(pm, reps, pw, boot.replicates(worse_pred, score))
    recall = next(d for d in diffs if d["metric"] == "recall_dep")
    assert recall["diff"] > 0 and recall["p_a_gt_b"] > 0.5
    auc = next(d for d in diffs if d["metric"] == "auc")
    assert auc["diff"] == pytest.approx(0.0) and auc["ci_low"] == pytest.approx(0.0)


def test_svm_without_score_has_nan_auc(toy_predictions):
    user_id, label, pred, _ = toy_predictions
    boot = UserBootstrap(user_id, label, n_boot=4, seed=0)
    assert boot.replicates(pred, None)["auc"].isna().all()
    assert np.isnan(point_metrics(label, pred, None)["auc"])
