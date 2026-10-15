"""Puntajes de BETO mono sobre el test nativo (splits/test.parquet)."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.models.evaluate_beto_multi import _TextDS  # noqa: E402

MODEL_DIR = REPO / "models" / "beto" / "best_model"
meta = json.loads((MODEL_DIR.parent / "model_meta.json").read_text())
label2id = {int(k): int(v) for k, v in meta["label2id"].items()}
id2label = {int(k): int(v) for k, v in meta["id2label"].items()}
pos_idx = [k for k, v in id2label.items() if v == 2][0]

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "splits"
OUT = sys.argv[2] if len(sys.argv) > 2 else "scores_beto_mono.parquet"
test = pd.read_parquet(REPO / "data" / "processed" / SPLIT / "test.parquet", columns=["doc_id", "text_clean", "label", "corpus_source"])
if len(sys.argv) > 3:
    test = test[test["corpus_source"] != sys.argv[3]].reset_index(drop=True)
tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR)).to("cuda").eval()
args = TrainingArguments(output_dir=str(REPO / "temp" / "_beto_eval"), per_device_eval_batch_size=32, report_to="none", fp16=True)
ds = _TextDS(test["text_clean"].fillna("").astype(str).tolist(), [label2id[int(l)] for l in test["label"]], tok, int(meta["max_length"]))
logits = Trainer(model=model, args=args).predict(ds).predictions
proba = np.exp(logits - logits.max(1, keepdims=True))
proba /= proba.sum(1, keepdims=True)
out = pd.DataFrame({"doc_id": test["doc_id"], "corpus_source": test["corpus_source"], "label": test["label"],
                    "pred": [id2label[int(i)] for i in logits.argmax(1)], "proba_dep": proba[:, pos_idx]})
out.to_parquet(REPO / "temp" / OUT, index=False)
from sklearn.metrics import f1_score, roc_auc_score
print("F1", f1_score(out.label, out.pred, average="macro"), "AUC", roc_auc_score(out.label == 2, out.proba_dep))
