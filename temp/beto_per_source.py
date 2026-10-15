"""Métricas de BETO multi por fuente a partir de las predicciones guardadas."""

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "temp"))

from ablation_per_source import metrics  # noqa: E402


def main() -> None:
    pred = pd.read_parquet(REPO / "models" / "beto_multi" / "predictions_test.parquet")
    y, p, s = pred["label"].to_numpy(), pred["pred"].to_numpy(), pred["proba_dep"].to_numpy()
    rows = [{"model": "beto_multi", "subset": "test_total", **metrics(y, p, s)}]
    for src in sorted(pred["corpus_source"].unique()):
        m = (pred["corpus_source"] == src).to_numpy()
        rows.append({"model": "beto_multi", "subset": src, **metrics(y[m], p[m], s[m])})
    out = pd.DataFrame(rows)
    out.to_csv(REPO / "informe" / "tables" / "beto_multi_per_source.csv", index=False)
    print(out.round(4).to_string())


if __name__ == "__main__":
    main()
