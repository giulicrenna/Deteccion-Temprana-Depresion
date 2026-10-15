import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("informe/tables/all_models_metrics.csv")
df = df[df.split == "test"].drop_duplicates("model", keep="last")
order = ["logreg_combined", "logreg_downsampled", "logreg_full_balanced", "beto",
         "logreg_multi_downsampled", "logreg_multi_full", "beto_multi"]
names = {"logreg_combined": "LR TF-IDF+léxicos", "logreg_downsampled": "LR submuestreo",
         "logreg_full_balanced": "LR ponderada", "beto": "BETO",
         "logreg_multi_downsampled": "LR submuestreo", "logreg_multi_full": "LR ponderada",
         "beto_multi": "BETO"}
df = df.set_index("model").loc[order[::-1]]
colors = ["#eb6834" if "multi" in m else "#2a78d6" for m in df.index]
labels = [names[m] + (" (multi)" if "multi" in m else " (mono)") for m in df.index]
metrics = [("f1_macro", "F1-macro"), ("f1_class_2_depressive", "F1 clase depresiva"),
           ("auc_binary_depressive", "AUC-ROC")]
fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
for ax, (col, title) in zip(axes, metrics):
    ax.barh(labels, df[col], color=colors, height=0.6, edgecolor="white", linewidth=2)
    for y, v in enumerate(df[col]):
        ax.text(v + 0.005, y, f"{v:.3f}".replace(".", ","), va="center", fontsize=8, color="#333333")
    ax.set_xlim(0, 1.0)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="x", color="#e5e5e5", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color="#2a78d6", label="Mono-corpus (test mono)"),
                    Patch(color="#eb6834", label="Multi-corpus (test multi)")],
           loc="lower center", ncol=2, frameon=False, fontsize=9)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig("informe/figures/comparacion_modelos_test.png", dpi=200)
