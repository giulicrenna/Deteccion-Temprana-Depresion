"""Tests de la agregación de predicciones a nivel de usuario."""

import pandas as pd
import pytest

from src.evaluation.user_aggregation import aggregate_users, user_level_metrics


@pytest.fixture
def pred_df():
    rows = []
    for u, label, scores in [
        ("a", 2, [0.9, 0.8, 0.2]),
        ("b", 2, [0.4, 0.6, 0.3, 0.2]),
        ("c", 0, [0.1, 0.7]),
        ("d", 0, [0.2, 0.3, 0.1]),
    ]:
        for i, s in enumerate(scores):
            rows.append(
                {"doc_id": f"{u}{i}", "user_id": u, "label": label, "score": s, "pred": 2 if s >= 0.5 else 0}
            )
    return pd.DataFrame(rows)


def test_aggregate_users(pred_df):
    users = aggregate_users(pred_df).set_index("user_id")
    assert users.loc["a", "n_docs"] == 3
    assert users.loc["a", "mean_score"] == pytest.approx((0.9 + 0.8 + 0.2) / 3)
    assert users.loc["b", "positive_rate"] == pytest.approx(0.25)
    assert users.loc["c", "label"] == 0


def test_user_level_metrics_levels(pred_df):
    out = pd.DataFrame(user_level_metrics(pred_df, "lr_mono", "native_test", n_boot=50, seed=0))
    assert set(out["level"]) == {"document", "user_mean_score", "user_positive_rate"}
    assert (out["n_users"] == 4).all()
    auc = out.query("level == 'user_mean_score' and metric == 'auc'")["value"].iloc[0]
    # Medias: depresivos 0.633 y 0.375; controles 0.4 y 0.2 → 3 de 4 pares bien ordenados.
    assert auc == pytest.approx(0.75)
    rate_recall = out.query("level == 'user_positive_rate' and metric == 'recall_dep'")
    assert rate_recall["value"].iloc[0] == pytest.approx(0.5)
