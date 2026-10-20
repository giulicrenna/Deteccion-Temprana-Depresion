"""Validación de hipótesis sobre modelos ya entrenados (sin reentrenar).

Pasos:
1. Unifica predicciones por documento (`collect_predictions`).
2. Leaderboard con IC 95 % por bootstrap de usuarios, en el test nativo, el test multi
   completo y cada fuente (`leaderboard_ci.csv`).
3. Diferencias pareadas para la hipótesis principal (BETO frente a líneas base) y para la
   transferencia translingüística (multi frente a mono) (`paired_diffs.csv`).
4. Hs4: coeficientes de la regresión logística frente al léxico psicolingüístico
   (`hs4_lexicon_enrichment_*.csv`, `lr_coefficients_*.csv`, `lr_lexicon_terms_*.csv`).
5. Métricas a nivel de usuario en el test nativo (`user_level_metrics.csv`).

El control dialectal corre aparte porque requiere GPU para BETO:
`python -m src.evaluation.dialect_control --beto`.

Uso:
    python -m src.evaluation.run_validation
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from src.evaluation.bootstrap import UserBootstrap, paired_difference, point_metrics, summarize
from src.evaluation.collect_predictions import NATIVE_SOURCE, collect, load_predictions
from src.evaluation.user_aggregation import user_level_metrics
from src.utils.logging import get_logger
from src.utils.seeds import set_seed
from src.xai import linear_coefficients

logger = get_logger("run_validation")

NATIVE_SCOPE = "native_test"
MULTI_SCOPE = "multi_test"

# (hipótesis, modelo A, modelo B, alcances); se reporta A − B.
COMPARISONS = [
    ("H_principal", "beto_mono", "lr_mono", [NATIVE_SCOPE]),
    ("H_principal", "beto_mono", "svm_mono", [NATIVE_SCOPE]),
    ("H_principal", "beto_mono", "lr_mono_downsampled", [NATIVE_SCOPE]),
    ("H_principal", "beto_multi", "lr_multi", [NATIVE_SCOPE, MULTI_SCOPE]),
    ("H_transferencia", "beto_multi", "beto_mono", None),
    ("H_transferencia", "lr_multi", "lr_coello", None),
]

HS4_MODELS = {
    "lr_mono": "models/baseline_logreg.joblib",
    "lr_mono_downsampled": "models/logreg_downsampled.joblib",
}

NEW_TABLES = [
    "leaderboard_ci.csv",
    "paired_diffs.csv",
    "user_level_metrics.csv",
    "hs4_lexicon_enrichment_lr_mono.csv",
    "hs4_lexicon_enrichment_lr_mono_downsampled.csv",
    "lr_coefficients_lr_mono.csv",
    "lr_lexicon_terms_lr_mono.csv",
    "dialect_control.csv",
    "dialect_control_diffs.csv",
    "dialect_marker_prevalence.csv",
    "dialect_masking_sample.csv",
]


def scope_masks(reference: pd.DataFrame) -> dict[str, pd.Series]:
    """Subconjuntos del test multi sobre los que se evalúa."""
    masks = {
        NATIVE_SCOPE: reference["corpus_source"] == NATIVE_SOURCE,
        MULTI_SCOPE: pd.Series(True, index=reference.index),
    }
    for source in sorted(reference["corpus_source"].unique()):
        if source != NATIVE_SOURCE:
            masks[source] = reference["corpus_source"] == source
    return masks


def evaluate_scopes(
    preds: dict[str, pd.DataFrame], n_boot: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leaderboard con IC y diferencias pareadas, compartiendo réplicas dentro de cada alcance."""
    reference = next(iter(preds.values()))
    for name, df in preds.items():
        if not (df["doc_id"].to_numpy() == reference["doc_id"].to_numpy()).all():
            raise ValueError(f"[{name}] predicciones desalineadas con el resto")

    board_rows, diff_rows = [], []
    for scope, mask in scope_masks(reference).items():
        sub_ref = reference[mask.to_numpy()]
        label = sub_ref["label"].to_numpy()
        boot = UserBootstrap(sub_ref["user_id"].to_numpy(), label, n_boot, seed)
        logger.info(f"[{scope}] filas={len(sub_ref):,} usuarios={boot.n_users:,}")

        results = {}
        for name, df in preds.items():
            sub = df[mask.to_numpy()]
            args = (sub["pred"].to_numpy(), sub["score"].to_numpy())
            results[name] = (point_metrics(label, *args), boot.replicates(*args))
            board_rows += [
                {
                    "model": name,
                    "scope": scope,
                    "n_rows": len(sub),
                    "n_users": boot.n_users,
                    **r,
                }
                for r in summarize(*results[name])
            ]

        for hypothesis, model_a, model_b, scopes in COMPARISONS:
            if scopes is not None and scope not in scopes:
                continue
            if model_a not in results or model_b not in results:
                continue
            diff_rows += [
                {
                    "hypothesis": hypothesis,
                    "scope": scope,
                    "model_a": model_a,
                    "model_b": model_b,
                    **r,
                }
                for r in paired_difference(*results[model_a], *results[model_b])
            ]
    return pd.DataFrame(board_rows), pd.DataFrame(diff_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validación de hipótesis sin reentrenar.")
    parser.add_argument("--split-dir", default="data/processed/splits_multi")
    parser.add_argument("--pred-dir", default="reports/predictions")
    parser.add_argument("--out-dir", default="reports/tables")
    parser.add_argument("--informe-dir", default="informe/tables")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    repo = Path.cwd()
    pred_dir, out_dir = repo / args.pred_dir, repo / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    models = collect(repo, repo / args.split_dir, pred_dir)
    preds = {name: load_predictions(pred_dir, name) for name in models}

    board, diffs = evaluate_scopes(preds, args.n_boot, args.seed)
    board.to_csv(out_dir / "leaderboard_ci.csv", index=False)
    diffs.to_csv(out_dir / "paired_diffs.csv", index=False)

    for tag, model_path in HS4_MODELS.items():
        linear_coefficients.run(
            repo / model_path,
            repo / linear_coefficients.DEFAULT_LEXICON,
            out_dir,
            top_n=100,
            top_k=500,
            tag=tag,
        )

    user_rows = []
    for name in models:
        native = load_predictions(pred_dir, name, native_only=True)
        user_rows += user_level_metrics(native, name, NATIVE_SCOPE, args.n_boot, args.seed)
    pd.DataFrame(user_rows).to_csv(out_dir / "user_level_metrics.csv", index=False)

    informe_dir = repo / args.informe_dir
    for table in NEW_TABLES:
        if (out_dir / table).exists():
            shutil.copy2(out_dir / table, informe_dir / table)
    logger.info(f"Tablas en {out_dir} (copiadas a {informe_dir})")


if __name__ == "__main__":
    main()
