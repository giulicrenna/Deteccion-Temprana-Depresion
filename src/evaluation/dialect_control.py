"""Control de atajos dialectales sin reentrenar (prueba de sensibilidad).

Los modelos ya entrenados se vuelven a puntuar sobre el test nativo con los marcadores
diatópicos enmascarados. La caída de rendimiento estima cuánto dependen de esos términos.
Para separar ese efecto de la simple pérdida de palabras se agrega un control aleatorio:
cada texto recibe la misma cantidad de máscaras, pero sobre palabras no marcadoras
elegidas al azar.

Limitación: un modelo entrenado sin esos términos podría compensar con otras señales;
esta prueba mide la dependencia de los modelos actuales, no el rendimiento de un modelo
libre de atajos.

Uso:
    python -m src.evaluation.dialect_control            # LR y SVM (CPU)
    python -m src.evaluation.dialect_control --beto     # agrega BETO mono y multi (GPU)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.evaluation.bootstrap import (
    UserBootstrap,
    paired_difference,
    point_metrics,
    summarize,
)
from src.evaluation.collect_predictions import (
    NATIVE_SOURCE,
    load_predictions,
    load_test,
    score_pipeline,
)
from src.features.dialect_masking import MASK_TOKEN, build_regex, load_markers, mask_texts
from src.utils.logging import get_logger
from src.utils.seeds import set_seed

logger = get_logger("dialect_control")

SKLEARN_MODELS = {
    "lr_mono": "models/baseline_logreg.joblib",
    "lr_mono_downsampled": "models/logreg_downsampled.joblib",
    "svm_mono": "models/baseline_svm.joblib",
}
# fp16 replica la precisión con que se generaron las predicciones originales guardadas.
BETO_MODELS = {
    "beto_mono": ("models/beto/best_model", True),
    "beto_multi": ("models/beto_multi/best_model", False),
}
_WORD = re.compile(r"\w+")


def random_mask_texts(
    texts: list[str], regex: re.Pattern, counts: list[int], seed: int
) -> list[str]:
    """Enmascara en cada texto tantas palabras no marcadoras como marcadores tenía."""
    rng = np.random.default_rng(seed)
    out = []
    for text, n in zip(texts, counts):
        text = text or ""
        if n == 0:
            out.append(text)
            continue
        marker_spans = [m.span() for m in regex.finditer(text)]
        candidates = [
            m.span()
            for m in _WORD.finditer(text)
            if not any(s < m.end() and m.start() < e for s, e in marker_spans)
        ]
        chosen = sorted(
            (candidates[i] for i in rng.choice(len(candidates), min(n, len(candidates)), False)),
            reverse=True,
        )
        for s, e in chosen:
            text = text[:s] + MASK_TOKEN + text[e:]
        out.append(text)
    return out


def score_beto(model_dir: Path, texts: list[str], fp16: bool, tmp_dir: Path) -> tuple:
    """Inferencia por lotes con BETO; devuelve (pred, proba de la clase depresiva)."""
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    from src.models.evaluate_beto_multi import _TextDS

    meta = json.loads((model_dir.parent / "model_meta.json").read_text())
    id2label = {int(k): int(v) for k, v in meta["id2label"].items()}
    pos_idx = next(k for k, v in id2label.items() if v == 2)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to("cuda").eval()
    args = TrainingArguments(
        output_dir=str(tmp_dir),
        per_device_eval_batch_size=32,
        report_to="none",
        fp16=fp16,
    )
    ds = _TextDS(texts, [0] * len(texts), tokenizer, int(meta["max_length"]))
    logits = Trainer(model=model, args=args).predict(ds).predictions
    proba = np.exp(logits - logits.max(1, keepdims=True))
    proba /= proba.sum(1, keepdims=True)
    pred = np.array([id2label[int(i)] for i in logits.argmax(1)])
    return pred, proba[:, pos_idx]


def marker_prevalence(test: pd.DataFrame, markers: dict) -> pd.DataFrame:
    """Proporción de textos y de usuarios con al menos un marcador, por categoría y clase."""
    rows = []
    for cat in markers:
        _, counts = mask_texts(test["text_clean"].tolist(), build_regex(markers, [cat]))
        has = pd.Series(np.asarray(counts) > 0, index=test.index)
        by_user = has.groupby(test["user_id"]).mean()
        user_label = test.groupby("user_id")["label"].first()
        for label in sorted(test["label"].unique()):
            in_label = (test["label"] == label).to_numpy()
            users = user_label.index[user_label == label]
            rows.append(
                {
                    "category": cat,
                    "label": int(label),
                    "share_docs_with_marker": float(has[in_label].mean()),
                    "mean_user_share_docs_with_marker": float(by_user[users].mean()),
                    "n_users": len(users),
                }
            )
    return pd.DataFrame(rows)


def build_conditions(
    texts: list[str], markers: dict, seed: int, per_category: bool
) -> dict[str, list[str]]:
    """Textos de cada condición: todas las categorías, cada una por separado y el control."""
    all_regex = build_regex(markers, list(markers))
    masked_all, counts = mask_texts(texts, all_regex)
    conditions = {"masked_all": masked_all}
    if per_category:
        for cat in markers:
            conditions[f"masked_{cat}"] = mask_texts(texts, build_regex(markers, [cat]))[0]
    conditions["random_control"] = random_mask_texts(texts, all_regex, counts, seed)
    share = float(np.mean(np.asarray(counts) > 0))
    n_random = sum(t.count(MASK_TOKEN) for t in conditions["random_control"])
    logger.info(
        f"Textos con al menos un marcador: {share:.1%} | máscaras marcadores: {sum(counts):,}"
        f" | máscaras control aleatorio: {n_random:,}"
    )
    return conditions


def run(
    repo: Path,
    split_dir: Path,
    pred_dir: Path,
    out_dir: Path,
    include_beto: bool,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    """Puntúa cada modelo en cada condición y compara con el texto original."""
    set_seed(seed)
    cache_dir = pred_dir / "dialect"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    test = load_test(split_dir)
    test = test[test["corpus_source"] == NATIVE_SOURCE].reset_index(drop=True)
    markers = load_markers()
    marker_prevalence(test, markers).to_csv(out_dir / "dialect_marker_prevalence.csv", index=False)

    texts = test["text_clean"].tolist()
    conditions = build_conditions(texts, markers, seed, per_category=True)
    sample = test.assign(masked=conditions["masked_all"])
    sample = sample[sample["masked"] != sample["text_clean"]]
    sample.sample(n=min(200, len(sample)), random_state=seed)[
        ["doc_id", "label", "text_clean", "masked"]
    ].to_csv(out_dir / "dialect_masking_sample.csv", index=False)

    models: dict[str, tuple] = {k: ("sklearn", repo / v) for k, v in SKLEARN_MODELS.items()}
    if include_beto:
        models.update({k: ("beto", repo / p, fp16) for k, (p, fp16) in BETO_MODELS.items()})

    boot = UserBootstrap(test["user_id"].to_numpy(), test["label"].to_numpy(), n_boot, seed)
    label = test["label"].to_numpy()
    summary_rows, diff_rows = [], []
    for name, spec in models.items():
        original = load_predictions(pred_dir, name, native_only=True)
        if not (original["doc_id"].to_numpy() == test["doc_id"].to_numpy()).all():
            raise ValueError(f"[{name}] predicciones originales desalineadas con el test")
        base_point = point_metrics(label, original["pred"].to_numpy(), original["score"].to_numpy())
        base_reps = boot.replicates(original["pred"].to_numpy(), original["score"].to_numpy())
        summary_rows += [
            {"model": name, "condition": "original", **r} for r in summarize(base_point, base_reps)
        ]

        model_conditions = conditions if spec[0] == "sklearn" else {
            k: conditions[k] for k in ("masked_all", "random_control")
        }
        pipeline = joblib.load(spec[1]) if spec[0] == "sklearn" else None
        for cond, cond_texts in model_conditions.items():
            cache = cache_dir / f"{name}__{cond}.parquet"
            if cache.exists():
                cached = pd.read_parquet(cache)
                pred, score = cached["pred"].to_numpy(), cached["score"].to_numpy()
            else:
                logger.info(f"[{name}] puntuando condición {cond}...")
                if pipeline is not None:
                    pred, score = score_pipeline(pipeline, cond_texts)
                else:
                    pred, score = score_beto(spec[1], cond_texts, spec[2], repo / "temp" / "_beto_eval")
                pd.DataFrame({"doc_id": test["doc_id"], "pred": pred, "score": score}).to_parquet(
                    cache, index=False
                )
            point = point_metrics(label, pred, score)
            reps = boot.replicates(pred, score)
            summary_rows += [{"model": name, "condition": cond, **r} for r in summarize(point, reps)]
            diff_rows += [
                {"model": name, "condition": cond, "comparison": f"{cond} - original", **r}
                for r in paired_difference(point, reps, base_point, base_reps)
            ]
            logger.info(
                f"[{name}] {cond}: F1-macro {base_point['f1_macro']:.4f} -> {point['f1_macro']:.4f}"
                f" | AUC {base_point['auc']:.4f} -> {point['auc']:.4f}"
            )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "dialect_control.csv", index=False)
    pd.DataFrame(diff_rows).to_csv(out_dir / "dialect_control_diffs.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Control de atajos dialectales (inferencia).")
    parser.add_argument("--split-dir", default="data/processed/splits_multi")
    parser.add_argument("--pred-dir", default="reports/predictions")
    parser.add_argument("--out-dir", default="reports/tables")
    parser.add_argument("--beto", action="store_true", help="Incluye BETO (requiere GPU).")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    repo = Path.cwd()
    run(
        repo,
        repo / args.split_dir,
        repo / args.pred_dir,
        repo / args.out_dir,
        args.beto,
        args.n_boot,
        args.seed,
    )


if __name__ == "__main__":
    main()
