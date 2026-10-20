"""Métricas a nivel de usuario a partir de las predicciones por documento.

Las publicaciones heredan la etiqueta de su autor, por lo que la clasificación por
documento tiene un techo bajo de precisión. Agregar los scores de cada usuario acerca la
evaluación a la unidad de interés clínico (el perfil). Solo es informativo donde hay
muchos documentos por usuario (Coello-Guilarte); en las fuentes de Reddit y Kaggle casi
cada publicación tiene su propio `user_id`, y así se reporta en `mean_docs_per_user`.

Agregaciones:
- `mean_score`: media del score del usuario, umbral del modelo (0.5 o 0 para la SVM).
- `positive_rate`: proporción de documentos predichos como depresivos, umbral 0.5
  (voto mayoritario).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.bootstrap import DEP, UserBootstrap, point_metrics, summarize

SCORE_THRESHOLD = {"svm_mono": 0.0}
DEFAULT_THRESHOLD = 0.5


def aggregate_users(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Una fila por usuario con su etiqueta, score medio y tasa de positivos."""
    grouped = pred_df.assign(is_pos_pred=pred_df["pred"] == DEP).groupby("user_id")
    users = grouped.agg(
        label=("label", lambda s: s.mode().iloc[0]),
        n_docs=("label", "size"),
        mean_score=("score", "mean"),
        positive_rate=("is_pos_pred", "mean"),
    )
    return users.reset_index()


def user_level_metrics(
    pred_df: pd.DataFrame, model: str, scope: str, n_boot: int, seed: int
) -> list[dict]:
    """Métricas por documento y por usuario (dos agregaciones) con IC por bootstrap."""
    threshold = SCORE_THRESHOLD.get(model, DEFAULT_THRESHOLD)
    users = aggregate_users(pred_df)
    base = {
        "model": model,
        "scope": scope,
        "n_users": len(users),
        "n_users_dep": int((users["label"] == DEP).sum()),
        "mean_docs_per_user": float(users["n_docs"].mean()),
    }
    rows = []

    doc_boot = UserBootstrap(pred_df["user_id"].to_numpy(), pred_df["label"].to_numpy(), n_boot, seed)
    doc_args = (pred_df["pred"].to_numpy(), pred_df["score"].to_numpy())
    doc_point = point_metrics(pred_df["label"].to_numpy(), *doc_args)
    rows += [
        {**base, "level": "document", **r}
        for r in summarize(doc_point, doc_boot.replicates(*doc_args))
    ]

    user_boot = UserBootstrap(users["user_id"].to_numpy(), users["label"].to_numpy(), n_boot, seed)
    label = users["label"].to_numpy()
    for level, score, cut in (
        ("user_mean_score", users["mean_score"].to_numpy(), threshold),
        ("user_positive_rate", users["positive_rate"].to_numpy(), 0.5),
    ):
        pred = np.where(score >= cut, DEP, 0)
        point = point_metrics(label, pred, score)
        rows += [
            {**base, "level": level, **r}
            for r in summarize(point, user_boot.replicates(pred, score))
        ]
    return rows


__all__ = ["aggregate_users", "user_level_metrics"]
