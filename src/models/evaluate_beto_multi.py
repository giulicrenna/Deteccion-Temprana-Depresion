"""Eval post-hoc de BETO multi-corpus en batches (evita OOM).

Versión multi-fuente de `evaluate_beto_batched.py`: apunta a
`models/beto_multi/best_model` y `data/processed/splits_multi`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from src.evaluation.metrics import compute
from src.models.train_baseline import load_split_data
from src.utils.seeds import set_seed

SEED = 42
MODEL_DIR = Path("models/beto_multi/best_model")
DATA_PROC = Path("data/processed/splits_multi")
TAB_DIR = Path("reports/tables")
FIG_DIR = Path("reports/figures")
BATCH_SIZE = 32


class _TextDS(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item


def main() -> None:
    set_seed(SEED)
    assert MODEL_DIR.exists(), f"No existe {MODEL_DIR}. Entrená primero con 03_finetune_beto_multi.ipynb"
    assert torch.cuda.is_available(), "Requiere CUDA"

    print(f"Cargando modelo desde {MODEL_DIR}...")
    meta = json.loads((MODEL_DIR.parent / "model_meta.json").read_text())
    label2id = {int(k): int(v) for k, v in meta["label2id"].items()}
    id2label = {int(k): int(v) for k, v in meta["id2label"].items()}
    max_length = int(meta.get("max_length", 128))
    print(f"  label2id={label2id} id2label={id2label} max_length={max_length}")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.to("cuda")
    model.eval()

    args = TrainingArguments(
        output_dir="models/beto_multi/_tmp_eval",
        per_device_eval_batch_size=BATCH_SIZE,
        report_to="none",
        seed=SEED,
    )
    trainer = Trainer(model=model, args=args, tokenizer=tokenizer)

    print("Cargando splits multi...")
    df_train, df_val, df_test = load_split_data(data_dir=DATA_PROC, max_samples=None, seed=SEED)

    df_pos = df_train[df_train["label"] == 2]
    n_train_down = 2 * len(df_pos)

    rows = []
    for split_name, df_split in [("val", df_val), ("test", df_test)]:
        texts = df_split["text_clean"].astype(str).tolist()
        labels_orig = df_split["label"].astype(int).tolist()
        labels = [label2id[l] for l in labels_orig]

        ds = _TextDS(texts, labels, tokenizer, max_length)
        t0 = time.time()
        out = trainer.predict(ds)
        elapsed = time.time() - t0
        logits = out.predictions
        y_true = labels_orig
        y_pred_idx = np.argmax(logits, axis=1)
        y_pred = [id2label[int(p)] for p in y_pred_idx]
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        proba = exp / exp.sum(axis=1, keepdims=True)
        pos_idx = 1 if id2label.get(1) == 2 else 0
        auc_bin = float(roc_auc_score(y_true, proba[:, pos_idx]))

        m = compute(y_true, y_pred, y_proba=proba.tolist())
        rows.append({
            "model": "beto_multi",
            "split": split_name,
            "n_train_rows": n_train_down,
            "class_balance_strategy": "downsample_majority",
            "feature_set": "beto_transformer_multi",
            "accuracy": m["accuracy"],
            "f1_macro": m["f1_macro"],
            "f1_weighted": m["f1_weighted"],
            "kappa": m["kappa"],
            "auc_binary_depressive": auc_bin,
            "f1_class_0_control": m["report"].get("0", {}).get("f1-score", np.nan),
            "f1_class_2_depressive": m["report"].get("2", {}).get("f1-score", np.nan),
            "created_at": pd.Timestamp.utcnow().isoformat(timespec="seconds"),
        })
        print(f"  {split_name}: {len(texts):,} ejemplos en {elapsed:.1f}s | F1={m['f1_macro']:.4f} | AUC_dep={auc_bin:.4f}")

    df_metrics = pd.DataFrame(rows)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    unified = TAB_DIR / "all_models_metrics.csv"
    df_existing = pd.read_csv(unified)
    df_new = pd.concat([df_existing, df_metrics], ignore_index=True)
    df_new.to_csv(unified, index=False)
    print(f"\nMétricas unificadas actualizadas: {unified} ({len(df_new)} filas)")
    print(df_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
