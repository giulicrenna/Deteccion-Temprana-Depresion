"""Ablación por fuente sobre splits_multi con la línea base TF-IDF + LR.

A) lr_multi:  entrenada sobre todo el train multi (downsample 50/50).
B) lr_coello: entrenada solo sobre filas Coello-Guilarte del train multi (downsample 50/50).
Ambas se evalúan sobre el mismo test multi, global y desagregado por fuente.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.models.train_baseline import build_baseline_pipeline, load_config  # noqa: E402
from src.utils.seeds import set_seed  # noqa: E402

SEED = 42
DEP = 2
SPLITS = REPO / "data" / "processed" / "splits_multi"
OUT_TAB = REPO / "informe" / "tables"
OUT_FIG = REPO / "informe" / "figures"


def downsample(df: pd.DataFrame) -> pd.DataFrame:
    pos = df[df["label"] == DEP]
    neg = df[df["label"] != DEP]
    n = min(len(pos), len(neg))
    return (
        pd.concat([pos.sample(n=n, random_state=SEED), neg.sample(n=n, random_state=SEED)])
        .sample(frac=1.0, random_state=SEED)
        .reset_index(drop=True)
    )


def metrics(y, pred, score) -> dict:
    both = len(np.unique(y)) == 2
    return {
        "n": len(y),
        "prev_dep": float(np.mean(y == DEP)),
        "accuracy": accuracy_score(y, pred),
        "recall_dep": recall_score(y, pred, pos_label=DEP, zero_division=0) if (y == DEP).any() else np.nan,
        "specificity": recall_score(y, pred, pos_label=0, zero_division=0) if (y == 0).any() else np.nan,
        "f1_macro": f1_score(y, pred, average="macro") if both else np.nan,
        "f1_dep": f1_score(y, pred, pos_label=DEP, zero_division=0) if both else np.nan,
        "kappa": cohen_kappa_score(y, pred) if both else np.nan,
        "auc": roc_auc_score(y == DEP, score) if both else np.nan,
    }


def main() -> None:
    set_seed(SEED)
    cfg = load_config(REPO / "configs" / "models.yaml")
    cols = ["doc_id", "text_clean", "label", "corpus_source"]
    train = pd.read_parquet(SPLITS / "train.parquet", columns=cols)
    test = pd.read_parquet(SPLITS / "test.parquet", columns=cols)
    for d in (train, test):
        d["text_clean"] = d["text_clean"].fillna("").astype(str)

    variants = {
        "lr_multi": downsample(train),
        "lr_coello": downsample(train[train["corpus_source"] == "coello_guilarte"]),
    }
    y_test = test["label"].to_numpy()
    test_ids = test["doc_id"].to_numpy()
    rows, roc = [], {}
    for name, df_tr in variants.items():
        print(f"[{name}] train={len(df_tr):,}", flush=True)
        pipe = build_baseline_pipeline("logreg", config=cfg, seed=SEED)
        pipe.fit(df_tr["text_clean"].tolist(), df_tr["label"].to_numpy())
        classes = list(pipe.classes_)
        score = pipe.predict_proba(test["text_clean"].tolist())[:, classes.index(DEP)]
        pred = pipe.predict(test["text_clean"].tolist())
        pd.DataFrame({"doc_id": test_ids, "label": y_test, "pred": pred, "proba_dep": score}).to_parquet(REPO / "temp" / f"scores_{name}.parquet", index=False)
        rows.append({"model": name, "subset": "test_total", **metrics(y_test, pred, score)})
        for src in sorted(test["corpus_source"].unique()):
            m = (test["corpus_source"] == src).to_numpy()
            rows.append({"model": name, "subset": src, **metrics(y_test[m], pred[m], score[m])})
        coello = (test["corpus_source"] == "coello_guilarte").to_numpy()
        roc[name] = roc_curve(y_test[coello] == DEP, score[coello])
        print(pd.DataFrame(rows[-5:]).round(4).to_string(), flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_TAB / "ablation_per_source.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    labels = {"lr_multi": "LR multi-corpus", "lr_coello": "LR solo Coello-Guilarte"}
    for name, (fpr, tpr, _) in roc.items():
        auc = out.query("model == @name and subset == 'coello_guilarte'")["auc"].iloc[0]
        ax.plot(fpr, tpr, label=f"{labels[name]} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1)
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title("ROC sobre el subconjunto Coello-Guilarte del test multi")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "ablation_roc_coello.png", dpi=200)
    print("OK", flush=True)


if __name__ == "__main__":
    main()
