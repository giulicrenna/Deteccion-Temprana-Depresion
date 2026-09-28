"""Módulo unificado de inferencia y predicción para modelos entrenados.

Soporta tanto pipelines baseline (.joblib) como modelos BETO (directorio Transformers).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from src.utils.logging import get_logger

logger = get_logger("predict")


def is_transformer_model(model_path: Path) -> bool:
    """Verifica si el path corresponde a un directorio de Transformers."""
    return model_path.is_dir() and (
        (model_path / "config.json").exists()
        or (model_path / "best_model" / "config.json").exists()
    )


def load_model(model_path: Path | str) -> tuple[str, Any, Any]:
    """Carga el modelo y devuelve (tipo, modelo, extra_meta)."""
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"No existe el modelo en {p}")

    if is_transformer_model(p):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_dir = p / "best_model" if (p / "best_model" / "config.json").exists() else p
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        model.eval()
        return "beto", (model, tokenizer), {}

    # Si es .joblib
    pipeline = joblib.load(p)
    return "baseline", pipeline, {}


def predict_texts(
    model_path: Path | str,
    texts: list[str],
) -> list[dict[str, Any]]:
    """Genera predicciones sobre una lista de textos."""
    mtype, model_obj, _ = load_model(model_path)
    results = []

    if mtype == "baseline":
        pipeline = model_obj
        preds = pipeline.predict(texts)
        has_proba = hasattr(pipeline.named_steps.get("clf"), "predict_proba")
        probas = pipeline.predict_proba(texts) if has_proba else None

        for idx, text in enumerate(texts):
            entry: dict[str, Any] = {
                "text": text,
                "label_pred": int(preds[idx]),
            }
            if probas is not None:
                entry["probabilities"] = [float(p) for p in probas[idx]]
                entry["confidence"] = float(np.max(probas[idx]))
            results.append(entry)

    elif mtype == "beto":
        model, tokenizer = model_obj
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        inputs = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

        id2label = model.config.id2label or {}

        for idx, text in enumerate(texts):
            raw_pred = int(preds[idx])
            mapped_label = int(id2label.get(str(raw_pred), id2label.get(raw_pred, raw_pred)))
            results.append(
                {
                    "text": text,
                    "label_pred": mapped_label,
                    "probabilities": [float(p) for p in probs[idx]],
                    "confidence": float(np.max(probs[idx])),
                }
            )

    return results


def predict_parquet(
    model_path: Path | str,
    input_parquet: Path | str,
    output_parquet: Path | str,
    text_column: str = "text_clean",
) -> None:
    """Aplica inferencia batch sobre un archivo Parquet."""
    df = pd.read_parquet(input_parquet)
    if text_column not in df.columns:
        if "text" in df.columns:
            text_column = "text"
        else:
            raise KeyError(f"No se encontró columna '{text_column}' en {input_parquet}")

    texts = df[text_column].fillna("").astype(str).tolist()
    preds = predict_texts(model_path, texts)

    df["label_pred"] = [p["label_pred"] for p in preds]
    if "confidence" in preds[0]:
        df["confidence"] = [p["confidence"] for p in preds]

    out_p = Path(output_parquet)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_p, index=False)
    logger.info(f"Predicciones guardadas en {output_parquet} ({len(df)} filas)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inferencia con modelos de la etapa 4.")
    parser.add_argument("--model", required=True, help="Ruta al modelo (.joblib o carpeta BETO).")
    parser.add_argument("--text", nargs="+", help="Texto(s) a clasificar.")
    parser.add_argument("--input-parquet", help="Archivo Parquet de entrada para inferencia batch.")
    parser.add_argument("--output-parquet", help="Archivo Parquet de salida con predicciones.")

    args = parser.parse_args()

    if args.text:
        preds = predict_texts(args.model, args.text)
        print(json.dumps(preds, indent=2, ensure_ascii=False))
    elif args.input_parquet and args.output_parquet:
        predict_parquet(args.model, args.input_parquet, args.output_parquet)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
