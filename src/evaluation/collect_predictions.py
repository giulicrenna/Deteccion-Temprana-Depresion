"""Predicciones por documento de todos los modelos sobre el test multi, en un formato único.

No reentrena nada: reutiliza las predicciones ya guardadas (BETO y LR de la ablación) y
solo hace inferencia con los pipelines `.joblib` que no tenían predicciones registradas.
El test nativo (Coello-Guilarte, 64 usuarios) es el subconjunto
`corpus_source == "coello_guilarte"` del test multi y coincide fila a fila con
`data/processed/splits/test.parquet`.

Salida: `reports/predictions/{modelo}.parquet` con
`doc_id, user_id, corpus_source, label, pred, score`. Para la SVM `score` es
`decision_function`; para el resto, la probabilidad de la clase depresiva.

Uso:
    python -m src.evaluation.collect_predictions
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.evaluation.bootstrap import DEP
from src.utils.logging import get_logger

logger = get_logger("collect_predictions")

NATIVE_SOURCE = "coello_guilarte"
TEST_COLUMNS = ["doc_id", "user_id", "corpus_source", "label", "text_clean"]
PRED_COLUMNS = ["doc_id", "user_id", "corpus_source", "label", "pred", "score"]

# Pipelines entrenados sobre el train mono (Coello-Guilarte); sin predicciones guardadas.
JOBLIB_MODELS = {
    "lr_mono": "models/baseline_logreg.joblib",
    "lr_mono_downsampled": "models/logreg_downsampled.joblib",
    "svm_mono": "models/baseline_svm.joblib",
}

# Predicciones ya existentes: nombre → lista de parquets con doc_id, pred, proba_dep.
SAVED_PREDICTIONS = {
    "beto_mono": ["temp/scores_beto_mono.parquet", "temp/scores_beto_mono_reddit.parquet"],
    "beto_multi": ["models/beto_multi/predictions_test.parquet"],
    "lr_multi": ["temp/scores_lr_multi.parquet"],
    "lr_coello": ["temp/scores_lr_coello.parquet"],
}


def load_test(split_dir: Path) -> pd.DataFrame:
    """Test multi con texto limpio y metadatos de usuario y fuente."""
    test = pd.read_parquet(split_dir / "test.parquet", columns=TEST_COLUMNS)
    test["text_clean"] = test["text_clean"].fillna("").astype(str)
    return test


def score_pipeline(pipeline, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Predicción y score de la clase depresiva para un pipeline scikit-learn."""
    classes = list(pipeline.classes_)
    pred = pipeline.predict(texts)
    if hasattr(pipeline, "predict_proba"):
        score = pipeline.predict_proba(texts)[:, classes.index(DEP)]
    else:
        # En binario decision_function > 0 favorece a classes_[1].
        sign = 1.0 if classes[1] == DEP else -1.0
        score = sign * pipeline.decision_function(texts)
    return np.asarray(pred), np.asarray(score, dtype=float)


def from_saved(test: pd.DataFrame, paths: list[Path]) -> pd.DataFrame:
    """Une predicciones guardadas con los metadatos del test por `doc_id`."""
    saved = pd.concat([pd.read_parquet(p, columns=["doc_id", "pred", "proba_dep"]) for p in paths])
    saved = saved.rename(columns={"proba_dep": "score"})
    out = test.drop(columns="text_clean").merge(saved, on="doc_id", how="inner", validate="1:1")
    return out[PRED_COLUMNS]


def from_joblib(test: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Inferencia de un pipeline `.joblib` sobre todo el test multi."""
    pred, score = score_pipeline(joblib.load(path), test["text_clean"].tolist())
    out = test.drop(columns="text_clean").assign(pred=pred, score=score)
    return out[PRED_COLUMNS]


def load_predictions(pred_dir: Path, name: str, native_only: bool = False) -> pd.DataFrame:
    """Lee las predicciones unificadas de un modelo."""
    df = pd.read_parquet(pred_dir / f"{name}.parquet")
    if native_only:
        df = df[df["corpus_source"] == NATIVE_SOURCE].reset_index(drop=True)
    return df


def collect(repo: Path, split_dir: Path, out_dir: Path, overwrite: bool = False) -> list[str]:
    """Genera los parquets unificados y devuelve los nombres de modelos disponibles."""
    out_dir.mkdir(parents=True, exist_ok=True)
    test = load_test(split_dir)
    sources = {name: [repo / p for p in paths] for name, paths in SAVED_PREDICTIONS.items()}
    sources.update({name: repo / p for name, p in JOBLIB_MODELS.items()})

    available = []
    for name, src in sources.items():
        out_path = out_dir / f"{name}.parquet"
        if out_path.exists() and not overwrite:
            available.append(name)
            continue
        missing = [p for p in (src if isinstance(src, list) else [src]) if not p.exists()]
        if missing:
            logger.warning(f"[{name}] se omite, faltan: {missing}")
            continue
        logger.info(f"[{name}] generando predicciones...")
        df = from_saved(test, src) if isinstance(src, list) else from_joblib(test, src)
        df.to_parquet(out_path, index=False)
        logger.info(f"[{name}] {len(df):,} filas -> {out_path}")
        available.append(name)
    return available


def main() -> None:
    parser = argparse.ArgumentParser(description="Unifica predicciones por documento.")
    parser.add_argument("--split-dir", default="data/processed/splits_multi")
    parser.add_argument("--out-dir", default="reports/predictions")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    repo = Path.cwd()
    collect(repo, repo / args.split_dir, repo / args.out_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
