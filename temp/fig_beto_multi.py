import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
FIG = "informe/figures/"

hist = json.load(open("models/beto_multi/checkpoint-117351/trainer_state.json"))["log_history"]
train = [(h["step"], h["loss"]) for h in hist if "loss" in h]
ev = [h for h in hist if "eval_loss" in h]
steps_per_epoch = ev[0]["step"]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
s, l = zip(*train)
a1.plot(s, pd.Series(l).rolling(20, min_periods=1).mean(), color=BLUE, lw=1.5, label="Pérdida de entrenamiento (media móvil)")
a1.plot([h["step"] for h in ev], [h["eval_loss"] for h in ev], color=ORANGE, lw=2, marker="o", ms=8, label="Pérdida de validación")
for e in range(1, 3):
    a1.axvline(e * steps_per_epoch, color="#cccccc", lw=0.8, ls="--")
a1.set_xlabel("Paso de entrenamiento")
a1.set_ylabel("Entropía cruzada")
a1.legend(frameon=False, fontsize=8)
a1.set_title("Pérdida", fontsize=10)
ep = [int(h["epoch"]) for h in ev]
for key, name, c in [("eval_f1_macro", "F1-macro", BLUE), ("eval_accuracy", "Exactitud", ORANGE), ("eval_kappa", "Kappa", AQUA)]:
    v = [h[key] for h in ev]
    a2.plot(ep, v, color=c, lw=2, marker="o", ms=8, label=name)
    a2.annotate(f"{v[-1]:.3f}".replace(".", ","), (ep[-1], v[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8)
a2.set_xticks(ep)
a2.set_xlabel("Época")
a2.set_ylim(0.4, 0.85)
a2.legend(frameon=False, fontsize=8, loc="lower right")
a2.set_title("Métricas de validación", fontsize=10)
for a in (a1, a2):
    a.grid(color="#eeeeee", lw=0.8)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(FIG + "beto_multi_training_curves.png", dpi=200)

pred = pd.read_parquet("models/beto_multi/predictions_test.parquet")
panels = [("Prueba completa", pred), ("Prueba nativa (Coello-Guilarte)", pred[pred.corpus_source == "coello_guilarte"]),
          ("Kaggle SDD (traducido)", pred[pred.corpus_source == "kaggle_sdd"])]
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
for ax, (title, d) in zip(axes, panels):
    cm = confusion_matrix(d.label, d.pred, labels=[0, 2])
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}".replace(",", "."), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "#222222", fontsize=10)
    ax.set_xticks([0, 1], ["0", "2"])
    ax.set_yticks([0, 1], ["0", "2"])
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(f"{title}\n(n = {len(d):,})".replace(",", "."), fontsize=10)
fig.tight_layout()
fig.savefig(FIG + "beto_multi_confusion_matrix.png", dpi=200)
print("ok")
