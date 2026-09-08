"""Entrenamiento y evaluación del baseline clásico (TF-IDF + LogReg / SVM).

Etapa 4 de la tesina. Diseñado para ejecutarse rápidamente en CPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.evaluation.metrics import compute
from src.utils.logging import get_logger
from src.utils.seeds import set_seed

logger = get_logger("train_baseline")


def load_config(config_path: Path | str = "configs/models.yaml") -> dict[str, Any]:
    """Carga configuración de modelos YAML."""
    p = Path(config_path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_split_data(
    data_dir: Path | str = "data/processed/splits",
    max_samples: int | None = None,
    synthetic_dev: bool = False,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga train, val y test. Si no existen o se activa synthetic_dev, carga datos sintéticos."""
    data_path = Path(data_dir)
    train_file = data_path / "train.parquet"
    val_file = data_path / "val.parquet"
    test_file = data_path / "test.parquet"

    if not synthetic_dev and train_file.exists() and val_file.exists() and test_file.exists():
        logger.info(f"Cargando splits desde {data_path}...")
        df_train = pd.read_parquet(train_file)
        df_val = pd.read_parquet(val_file)
        df_test = pd.read_parquet(test_file)
    else:
        logger.warning(
            "Splits no encontrados o modo --synthetic-dev activo. Usando datos sintéticos."
        )
        synth_file = Path("src/data/synthetic/sample.jsonl")
        if not synth_file.exists():
            raise FileNotFoundError(f"No se encontró {synth_file}")
        records = []
        with open(synth_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        df = pd.DataFrame(records)
        df["text_clean"] = df.get("text_clean", df.get("text", ""))

        # Partición sintética rápida 70/15/15
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n = len(df)
        n_tr = max(2, int(n * 0.7))
        n_v = max(1, int(n * 0.15))
        df_train = df.iloc[:n_tr].copy()
        df_val = df.iloc[n_tr : n_tr + n_v].copy()
        df_test = df.iloc[n_tr + n_v :].copy()

    # Submuestreo para computadoras con poca RAM si se solicita
    if max_samples is not None and len(df_train) > max_samples:
        logger.info(f"Submuestreando df_train a {max_samples} filas para optimizar memoria.")
        df_train = df_train.sample(n=max_samples, random_state=seed).reset_index(drop=True)

    # Validar columnas requeridas
    for name, d in [("train", df_train), ("val", df_val), ("test", df_test)]:
        if "text_clean" not in d.columns:
            d["text_clean"] = d.get("text_raw", d.get("text", "")).fillna("").astype(str)
        else:
            d["text_clean"] = d["text_clean"].fillna("").astype(str)
        if "label" not in d.columns:
            raise ValueError(f"Falta columna 'label' en split {name}")

    return df_train, df_val, df_test


def build_baseline_pipeline(
    model_type: str = "logreg",
    config: dict[str, Any] | None = None,
    seed: int = 42,
) -> Pipeline:
    """Crea un Pipeline scikit-learn con TfidfVectorizer + clasificador."""
    config = config or {}
    base_cfg = config.get("baseline", {})
    tfidf_cfg = base_cfg.get("tfidf", {})

    ngram_range = tuple(tfidf_cfg.get("ngram_range", [1, 2]))
    max_features = tfidf_cfg.get("max_features", 15000)
    min_df = tfidf_cfg.get("min_df", 2)
    sublinear_tf = tfidf_cfg.get("sublinear_tf", True)

    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
    )

    if model_type == "logreg":
        logreg_cfg = base_cfg.get("logreg", {})
        clf = LogisticRegression(
            C=float(logreg_cfg.get("C", 1.0)),
            class_weight=logreg_cfg.get("class_weight", "balanced"),
            max_iter=int(logreg_cfg.get("max_iter", 1000)),
            solver=logreg_cfg.get("solver", "lbfgs"),
            random_state=seed,
        )
    elif model_type == "svm":
        svm_cfg = base_cfg.get("svm", {})
        clf = LinearSVC(
            C=float(svm_cfg.get("C", 1.0)),
            class_weight=svm_cfg.get("class_weight", "balanced"),
            max_iter=int(svm_cfg.get("max_iter", 2000)),
            random_state=seed,
        )
    else:
        raise ValueError(f"Modelo desconocido: {model_type}. Opciones: 'logreg', 'svm'.")

    return Pipeline([("tfidf", vectorizer), ("clf", clf)])


def train_and_eval_model(
    model_type: str,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    config: dict[str, Any] | None = None,
    seed: int = 42,
) -> tuple[Pipeline, dict[str, Any]]:
    """Entrena el pipeline y calcula métricas sobre train, val y test."""
    set_seed(seed)
    pipeline = build_baseline_pipeline(model_type, config=config, seed=seed)

    X_train = df_train["text_clean"].tolist()
    y_train = df_train["label"].astype(int).tolist()

    logger.info(f"Entrenando {model_type.upper()} sobre {len(X_train)} ejemplos...")
    pipeline.fit(X_train, y_train)

    results: dict[str, Any] = {"model": model_type}

    for split_name, df_split in [("val", df_val), ("test", df_test)]:
        X_eval = df_split["text_clean"].tolist()
        y_eval = df_split["label"].astype(int).tolist()

        y_pred = pipeline.predict(X_eval)
        y_proba = None
        if hasattr(pipeline.named_steps["clf"], "predict_proba"):
            y_proba = pipeline.predict_proba(X_eval).tolist()

        metrics = compute(y_eval, y_pred.tolist(), y_proba=y_proba)
        results[split_name] = metrics
        logger.info(
            f"[{model_type.upper()}] {split_name.upper()} -> "
            f"Acc: {metrics['accuracy']:.4f} | "
            f"F1-Macro: {metrics['f1_macro']:.4f} | "
            f"F1-Weighted: {metrics['f1_weighted']:.4f}"
        )

    return pipeline, results


def save_baseline_artifacts(
    pipeline: Pipeline,
    results: dict[str, Any],
    model_type: str,
    out_dir: Path | str = "models",
    reports_dir: Path | str = "reports/tables",
) -> None:
    """Guarda el modelo entrenado y persiste métricas en tabla CSV."""
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    model_path = out_p / f"baseline_{model_type}.joblib"
    joblib.dump(pipeline, model_path)
    logger.info(f"Modelo guardado en {model_path}")

    # Guardar métricas en CSV
    rep_p = Path(reports_dir)
    rep_p.mkdir(parents=True, exist_ok=True)
    metrics_csv = rep_p / "baseline_metrics.csv"

    rows = []
    for split in ["val", "test"]:
        m = results.get(split, {})
        rows.append(
            {
                "model": model_type,
                "split": split,
                "accuracy": m.get("accuracy"),
                "f1_macro": m.get("f1_macro"),
                "f1_weighted": m.get("f1_weighted"),
                "kappa": m.get("kappa"),
                "auc_ovr_macro": m.get("auc_ovr_macro"),
            }
        )
    df_new = pd.DataFrame(rows)

    if metrics_csv.exists():
        df_old = pd.read_csv(metrics_csv)
        # Reemplazar métricas previas del mismo modelo y split
        df_comb = pd.concat([df_old, df_new]).drop_duplicates(
            subset=["model", "split"], keep="last"
        )
        df_comb.to_csv(metrics_csv, index=False)
    else:
        df_new.to_csv(metrics_csv, index=False)
    logger.info(f"Métricas guardadas en {metrics_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenar baseline clásico (Etapa 4).")
    parser.add_argument(
        "--model",
        choices=["logreg", "svm", "both"],
        default="both",
        help="Tipo de modelo a entrenar.",
    )
    parser.add_argument(
        "--config",
        default="configs/models.yaml",
        help="Ruta al archivo YAML de configuración.",
    )
    parser.add_argument(
        "--data-dir",
        default="data/processed/splits",
        help="Directorio con splits train/val/test.",
    )
    parser.add_argument(
        "--out-dir",
        default="models",
        help="Directorio donde guardar artefactos .joblib.",
    )
    parser.add_argument(
        "--reports-dir",
        default="reports/tables",
        help="Directorio donde guardar métricas CSV.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Límite de filas de train para máquinas con poca memoria.",
    )
    parser.add_argument(
        "--synthetic-dev",
        action="store_true",
        help="Fuerza el uso del dataset sintético para pruebas rápidas de desarrollo.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Semilla de reproducibilidad.")

    args = parser.parse_args()

    cfg = load_config(args.config)
    df_train, df_val, df_test = load_split_data(
        data_dir=args.data_dir,
        max_samples=args.max_samples,
        synthetic_dev=args.synthetic_dev,
        seed=args.seed,
    )

    models_to_run = ["logreg", "svm"] if args.model == "both" else [args.model]

    for m in models_to_run:
        pipe, res = train_and_eval_model(
            model_type=m,
            df_train=df_train,
            df_val=df_val,
            df_test=df_test,
            config=cfg,
            seed=args.seed,
        )
        save_baseline_artifacts(
            pipeline=pipe,
            results=res,
            model_type=m,
            out_dir=args.out_dir,
            reports_dir=args.reports_dir,
        )


if __name__ == "__main__":
    main()
