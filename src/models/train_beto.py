"""Script de fine-tuning para BETO (Spanish BERT).

Etapa 4 de la tesina.
Diseñado con soporte de GPU (CUDA) para Google Colab / Kaggle y modo --dry-run
para verificar el funcionamiento sin exigir recursos en CPUs locales.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.evaluation.metrics import compute
from src.utils.logging import get_logger
from src.utils.seeds import set_seed

logger = get_logger("train_beto")


class TextClassificationDataset(Dataset):
    """Dataset para tokenización bajo demanda con Hugging Face."""

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: Any,
        max_length: int = 128,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def load_config(config_path: Path | str = "configs/models.yaml") -> dict[str, Any]:
    """Carga configuración de modelos YAML."""
    p = Path(config_path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_label_mapping(labels: list[int]) -> tuple[dict[int, int], dict[int, int]]:
    """Crea mapeo bidireccional de etiquetas a índices contiguos 0..N-1."""
    unique = sorted(set(labels))
    label2id = {orig: idx for idx, orig in enumerate(unique)}
    id2label = {idx: orig for orig, idx in label2id.items()}
    return label2id, id2label


def compute_metrics_wrapper(id2label: dict[int, int]):
    """Wrapper para pasar función compute_metrics a transformers.Trainer."""

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        # Revertir mapeo a etiquetas originales
        y_true = [id2label.get(int(y), int(y)) for y in labels]
        y_pred = [id2label.get(int(p), int(p)) for p in preds]

        # Calcular probabilidades con softmax
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probas = (exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)).tolist()

        m = compute(y_true, y_pred, y_proba=probas)
        return {
            "accuracy": m["accuracy"],
            "f1_macro": m["f1_macro"],
            "f1_weighted": m["f1_weighted"],
            "kappa": m["kappa"],
        }

    return compute_metrics


def run_training(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    model_name: str = "dccuchile/bert-base-spanish-wwm-cased",
    max_length: int = 128,
    batch_size: int = 16,
    eval_batch_size: int = 32,
    learning_rate: float = 2e-5,
    epochs: int = 3,
    output_dir: str = "models/beto",
    seed: int = 42,
    dry_run: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Ejecuta el pipeline de fine-tuning de BETO."""
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Dispositivo detectado: {device.upper()}")
    if device == "cpu" and not dry_run:
        logger.warning(
            "Entrenamiento completo en CPU local puede ser extremadamente lento. "
            "Se recomienda ejecutar en Google Colab con aceleración GPU (T4)."
        )

    if dry_run:
        logger.info("Modo --dry-run activo: reduciendo dataset a 10 ejemplos y 1 época corta.")
        df_train = df_train.head(10).copy()
        df_val = df_val.head(6).copy()
        df_test = df_test.head(6).copy()
        epochs = 1
        batch_size = 2
        eval_batch_size = 2

    # Mapeo de etiquetas
    all_labels = df_train["label"].tolist() + df_val["label"].tolist() + df_test["label"].tolist()
    label2id, id2label = get_label_mapping(all_labels)
    num_labels = len(label2id)
    logger.info(f"Etiquetas detectadas: {label2id} (num_labels={num_labels})")

    y_train = [label2id[y] for y in df_train["label"].astype(int)]
    y_val = [label2id[y] for y in df_val["label"].astype(int)]
    y_test = [label2id[y] for y in df_test["label"].astype(int)]

    logger.info(f"Cargando tokenizer para '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds = TextClassificationDataset(
        df_train["text_clean"].tolist(), y_train, tokenizer, max_length
    )
    val_ds = TextClassificationDataset(df_val["text_clean"].tolist(), y_val, tokenizer, max_length)
    test_ds = TextClassificationDataset(
        df_test["text_clean"].tolist(), y_test, tokenizer, max_length
    )

    logger.info(f"Cargando modelo '{model_name}'...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label={str(k): str(v) for k, v in id2label.items()},
        label2id={str(k): str(v) for k, v in label2id.items()},
    )

    use_fp16 = torch.cuda.is_available() and not dry_run

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=not dry_run,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        seed=seed,
        fp16=use_fp16,
        logging_steps=5 if dry_run else 50,
        save_total_limit=1,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_wrapper(id2label),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] if not dry_run else [],
    )

    logger.info("Iniciando entrenamiento con Trainer...")
    trainer.train()

    logger.info("Evaluando sobre test split...")
    test_metrics = trainer.evaluate(eval_dataset=test_ds)
    logger.info(f"Métricas en TEST: {test_metrics}")

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_p / "best_model"))
    tokenizer.save_pretrained(str(out_p / "best_model"))

    # Guardar metadata del modelo
    meta = {
        "model_name": model_name,
        "label2id": label2id,
        "id2label": id2label,
        "max_length": max_length,
        "test_metrics": test_metrics,
    }
    with open(out_p / "model_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Modelo final y artefactos guardados en {output_dir}/best_model")
    return trainer, test_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tuning de BETO (Etapa 4).")
    parser.add_argument("--config", default="configs/models.yaml", help="Ruta a models.yaml.")
    parser.add_argument(
        "--data-dir", default="data/processed/splits", help="Directorio con splits."
    )
    parser.add_argument("--output-dir", default="models/beto", help="Directorio de salida.")
    parser.add_argument("--epochs", type=int, default=None, help="Número de épocas.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size de train.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Ejecuta 1 paso de prueba sobre datos mínimos para validar el pipeline sin GPU.",
    )
    parser.add_argument("--synthetic-dev", action="store_true", help="Usar dataset sintético.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla.")

    args = parser.parse_args()

    cfg = load_config(args.config)
    beto_cfg = cfg.get("beto", {})

    # Cargar datos
    from src.models.train_baseline import load_split_data

    df_train, df_val, df_test = load_split_data(
        data_dir=args.data_dir,
        synthetic_dev=args.synthetic_dev or args.dry_run,
        seed=args.seed,
    )

    epochs = args.epochs or beto_cfg.get("epochs", 3)
    batch_size = args.batch_size or beto_cfg.get("batch_size", 16)
    eval_batch_size = beto_cfg.get("eval_batch_size", 32)
    lr = args.lr or float(beto_cfg.get("learning_rate", 2e-5))
    model_name = beto_cfg.get("model_name", "dccuchile/bert-base-spanish-wwm-cased")
    max_length = beto_cfg.get("max_length", 128)

    run_training(
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        model_name=model_name,
        max_length=max_length,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        learning_rate=lr,
        epochs=epochs,
        output_dir=args.output_dir,
        seed=args.seed,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
