"""Tabla por dominio, comparación de modelos y ROC sobre el test nativo común."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
from sklearn.metrics import roc_curve

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "temp"))
from ablation_per_source import metrics  # noqa: E402

TMP = REPO / "temp"
FIG = REPO / "informe" / "figures"
TAB = REPO / "informe" / "tables"

src = pd.read_parquet(REPO / "data/processed/splits_multi/test.parquet", columns=["doc_id", "corpus_source"])
models = {
    "LR solo Coello": pd.read_parquet(TMP / "scores_lr_coello.parquet"),
    "LR multi": pd.read_parquet(TMP / "scores_lr_multi.parquet"),
    "BETO mono": pd.concat([pd.read_parquet(TMP / "scores_beto_mono.parquet"),
                            pd.read_parquet(TMP / "scores_beto_mono_reddit.parquet")]),
    "BETO multi": pd.read_parquet(REPO / "models/beto_multi/predictions_test.parquet"),
}
SUBSETS = {"coello_guilarte": "Nativo (Coello-Guilarte)", "kaggle_sdd": "Kaggle SDD",
           "depression_reddit": "Depression Reddit", "reddit_mh_posts": "Reddit MH Posts"}

rows, scored = [], {}
for name, d in models.items():
    d = src.merge(d[["doc_id", "label", "pred", "proba_dep"]], on="doc_id", how="inner")
    assert len(d) == len(src), name
    scored[name] = d
    rows.append({"model": name, "subset": "test_total",
                 **metrics(d.label.to_numpy(), d.pred.to_numpy(), d.proba_dep.to_numpy())})
    for s in SUBSETS:
        q = d[d.corpus_source == s]
        rows.append({"model": name, "subset": s,
                     **metrics(q.label.to_numpy(), q.pred.to_numpy(), q.proba_dep.to_numpy())})
table = pd.DataFrame(rows)
table.to_csv(TAB / "models_by_domain.csv", index=False)
print(table.round(4).to_string())

COLOR = {"LR solo Coello": "#86b6ef", "LR multi": "#f4a77f", "BETO mono": "#2a78d6", "BETO multi": "#eb6834"}
order = list(models)
fig, axes = plt.subplots(2, 2, figsize=(10, 5.6), sharex=True)
for r, (subset, title) in enumerate([("coello_guilarte", "Test nativo: Coello-Guilarte (250.847 textos)"),
                                     ("kaggle_sdd", "Test traducido: Kaggle SDD (46.414 textos)")]):
    for c, (col, mname) in enumerate([("f1_macro", "F1-macro"), ("auc", "AUC-ROC")]):
        ax = axes[r, c]
        sub = table[table.subset == subset].set_index("model").loc[order[::-1]]
        ax.barh(sub.index, sub[col], color=[COLOR[m] for m in sub.index], height=0.62, edgecolor="white", linewidth=2)
        for y, v in enumerate(sub[col]):
            ax.text(v + 0.01, y, f"{v:.3f}".replace(".", ","), va="center", fontsize=8, color="#333333")
        ax.set_xlim(0, 1.08)
        ax.set_title(f"{title}\n{mname}", fontsize=9)
        ax.grid(axis="x", color="#eeeeee", lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
fig.legend(handles=[Patch(color=COLOR[m], label=m) for m in order], loc="lower center", ncol=4, frameon=False, fontsize=9)
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(FIG / "comparacion_modelos_test.png", dpi=200)

fig, ax = plt.subplots(figsize=(5.6, 5))
for name in order:
    d = scored[name]
    d = d[d.corpus_source == "coello_guilarte"]
    fpr, tpr, _ = roc_curve(d.label == 2, d.proba_dep)
    auc = table.query("model == @name and subset == 'coello_guilarte'")["auc"].iloc[0]
    ax.plot(fpr, tpr, color=COLOR[name], lw=2, label=f"{name} (AUC = {auc:.3f})".replace(".", ","))
ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1)
ax.set_xlabel("Tasa de falsos positivos")
ax.set_ylabel("Tasa de verdaderos positivos")
ax.set_title("ROC sobre el test nativo común", fontsize=10)
ax.legend(loc="lower right", frameon=False, fontsize=8)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "roc_test_nativo.png", dpi=200)
print("OK")
