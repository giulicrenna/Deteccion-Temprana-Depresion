"""Métricas binarias con intervalos de confianza por bootstrap de usuarios.

Las publicaciones heredan la etiqueta de su autor, por lo que no son observaciones
independientes: el remuestreo se hace sobre `user_id` (bootstrap por conglomerados),
estratificado por la etiqueta del usuario. Cada réplica se resuelve con sumas
ponderadas por la multiplicidad de cada usuario, sin materializar filas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEP = 2
METRICS = (
    "accuracy",
    "recall_dep",
    "specificity",
    "f1_dep",
    "f1_control",
    "f1_macro",
    "kappa",
    "auc",
)


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    out = np.full(np.broadcast(num, den).shape, np.nan)
    np.divide(num, den, out=out, where=den > 0)
    return out


def metrics_from_confusion(
    tp: np.ndarray, fp: np.ndarray, fn: np.ndarray, tn: np.ndarray
) -> dict[str, np.ndarray]:
    """Métricas de umbral a partir de conteos de la matriz de confusión (clase 2 positiva)."""
    n = tp + fp + fn + tn
    f1_dep = _safe_div(2 * tp, 2 * tp + fp + fn)
    f1_control = _safe_div(2 * tn, 2 * tn + fn + fp)
    p_obs = _safe_div(tp + tn, n)
    p_exp = _safe_div((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn), n**2)
    return {
        "accuracy": p_obs,
        "recall_dep": _safe_div(tp, tp + fn),
        "specificity": _safe_div(tn, tn + fp),
        "f1_dep": f1_dep,
        "f1_control": f1_control,
        "f1_macro": (f1_dep + f1_control) / 2,
        "kappa": _safe_div(p_obs - p_exp, 1 - p_exp),
    }


@dataclass
class _RankedScores:
    """Scores ordenados y agrupados por empates, para calcular AUC ponderada en O(n)."""

    order: np.ndarray
    group_starts: np.ndarray
    is_pos_sorted: np.ndarray

    @classmethod
    def build(cls, score: np.ndarray, is_pos: np.ndarray) -> _RankedScores:
        order = np.argsort(score, kind="mergesort")
        sorted_score = score[order]
        starts = np.flatnonzero(np.r_[True, sorted_score[1:] != sorted_score[:-1]])
        return cls(order=order, group_starts=starts, is_pos_sorted=is_pos[order])

    def auc(self, weights: np.ndarray) -> float:
        """AUC-ROC con pesos por fila; los empates cuentan 1/2 (equivale a Mann-Whitney)."""
        w = weights[self.order]
        pos_w = np.add.reduceat(np.where(self.is_pos_sorted, w, 0.0), self.group_starts)
        neg_w = np.add.reduceat(np.where(self.is_pos_sorted, 0.0, w), self.group_starts)
        total_pos, total_neg = pos_w.sum(), neg_w.sum()
        if total_pos == 0 or total_neg == 0:
            return np.nan
        neg_below = np.cumsum(neg_w) - neg_w
        return float(np.sum(pos_w * (neg_below + 0.5 * neg_w)) / (total_pos * total_neg))


class UserBootstrap:
    """Bootstrap por usuario sobre un conjunto fijo de filas.

    Todos los modelos evaluados con la misma instancia comparten las mismas réplicas,
    lo que permite calcular diferencias pareadas.
    """

    def __init__(
        self,
        user_id: np.ndarray,
        label: np.ndarray,
        n_boot: int = 2000,
        seed: int = 42,
    ) -> None:
        self.user_codes, uniques = pd.factorize(pd.Series(user_id))
        self.n_users = len(uniques)
        self.is_pos = np.asarray(label) == DEP
        self.n_boot = n_boot

        user_label = (
            pd.Series(self.is_pos).groupby(self.user_codes).agg(lambda s: s.mode().iloc[0])
        )
        rng = np.random.default_rng(seed)
        # float32 representa exactamente conteos < 2**24 y evita copias float64 en el matmul.
        self.multiplicity = np.zeros((n_boot, self.n_users), dtype=np.float32)
        for stratum in (True, False):
            members = user_label.index[user_label.to_numpy() == stratum].to_numpy()
            if len(members) == 0:
                continue
            draws = rng.integers(0, len(members), size=(n_boot, len(members)))
            for b in range(n_boot):
                np.add.at(self.multiplicity[b], members[draws[b]], 1)

    def _user_confusion(self, pred_pos: np.ndarray) -> np.ndarray:
        """Conteos (tp, fp, fn, tn) por usuario, forma (4, n_users)."""
        cells = (
            pred_pos & self.is_pos,
            pred_pos & ~self.is_pos,
            ~pred_pos & self.is_pos,
            ~pred_pos & ~self.is_pos,
        )
        return np.stack(
            [np.bincount(self.user_codes, weights=c, minlength=self.n_users) for c in cells]
        ).astype(np.float32)

    def replicates(self, pred: np.ndarray, score: np.ndarray | None) -> pd.DataFrame:
        """Matriz (n_boot × métricas) para un modelo."""
        pred_pos = np.asarray(pred) == DEP
        counts = (self.multiplicity @ self._user_confusion(pred_pos).T).astype(np.float64)
        tp, fp, fn, tn = counts.T
        out = metrics_from_confusion(tp, fp, fn, tn)
        if score is not None:
            ranked = _RankedScores.build(np.asarray(score, dtype=float), self.is_pos)
            out["auc"] = np.array(
                [ranked.auc(self.multiplicity[b][self.user_codes]) for b in range(self.n_boot)]
            )
        else:
            out["auc"] = np.full(self.n_boot, np.nan)
        return pd.DataFrame(out, columns=list(METRICS))


def point_metrics(label: np.ndarray, pred: np.ndarray, score: np.ndarray | None) -> dict:
    """Métricas sobre la muestra observada (mismas fórmulas que las réplicas)."""
    is_pos = np.asarray(label) == DEP
    pred_pos = np.asarray(pred) == DEP
    counts = [
        np.sum(pred_pos & is_pos),
        np.sum(pred_pos & ~is_pos),
        np.sum(~pred_pos & is_pos),
        np.sum(~pred_pos & ~is_pos),
    ]
    out = {k: float(v) for k, v in metrics_from_confusion(*map(np.float64, counts)).items()}
    if score is None:
        out["auc"] = np.nan
    else:
        ranked = _RankedScores.build(np.asarray(score, dtype=float), is_pos)
        out["auc"] = ranked.auc(np.ones(len(is_pos)))
    return out


def summarize(
    point: dict, reps: pd.DataFrame, alpha: float = 0.05
) -> list[dict[str, float | str]]:
    """Valor puntual e IC percentil por métrica."""
    rows = []
    for m in METRICS:
        values = reps[m].dropna()
        lo, hi = (
            np.quantile(values, [alpha / 2, 1 - alpha / 2]) if len(values) else (np.nan, np.nan)
        )
        rows.append({"metric": m, "value": point[m], "ci_low": lo, "ci_high": hi})
    return rows


def paired_difference(
    point_a: dict,
    reps_a: pd.DataFrame,
    point_b: dict,
    reps_b: pd.DataFrame,
    alpha: float = 0.05,
) -> list[dict[str, float | str]]:
    """Diferencia A − B con IC pareado y proporción de réplicas en que A supera a B."""
    rows = []
    for m in METRICS:
        diff = (reps_a[m] - reps_b[m]).dropna()
        if diff.empty:
            continue
        lo, hi = np.quantile(diff, [alpha / 2, 1 - alpha / 2])
        rows.append(
            {
                "metric": m,
                "diff": point_a[m] - point_b[m],
                "ci_low": lo,
                "ci_high": hi,
                "p_a_gt_b": float(np.mean(diff > 0)),
            }
        )
    return rows


__all__ = [
    "DEP",
    "METRICS",
    "UserBootstrap",
    "metrics_from_confusion",
    "paired_difference",
    "point_metrics",
    "summarize",
]
