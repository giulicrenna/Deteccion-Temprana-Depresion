"""Split user-level estratificado 70/10/20 (train/val/test).

Crítico: el split es por user_id, no por documento, para evitar que el
mismo usuario aparezca en train y test (leakage).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.utils.logging import get_logger
from src.utils.seeds import set_seed

log = get_logger(__name__)


def assign_users(
    user_df: pd.DataFrame,
    train: float,
    val: float,
    test: float,
    seed: int,
) -> dict[str, str]:
    """Asigna cada user_id a {train, val, test} estratificando por label dominante.

    La estratificación usa la etiqueta "modal" del usuario (la clase más
    frecuente de sus mensajes). Si el usuario es 50/50, queda como `train`
    por default (caso degenerado, raro en este corpus).
    """
    # Etiqueta modal por usuario.
    modal = (
        user_df.groupby("user_id")["label"]
        .agg(lambda s: int(s.mode().iloc[0]) if not s.mode().empty else int(s.iloc[0]))
        .reset_index()
        .rename(columns={"label": "modal_label"})
    )

    # Split 1: train+val vs test
    train_val, test = train_test_split(
        modal,
        test_size=test,
        stratify=modal["modal_label"],
        random_state=seed,
    )
    # Split 2: train vs val
    val_ratio = val / (train + val)
    train, val = train_test_split(
        train_val,
        test_size=val_ratio,
        stratify=train_val["modal_label"],
        random_state=seed,
    )
    out = {}
    for uid in train["user_id"]:
        out[uid] = "train"
    for uid in val["user_id"]:
        out[uid] = "val"
    for uid in test["user_id"]:
        out[uid] = "test"
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", type=Path, default=Path("./data/processed"))
    parser.add_argument("--out", dest="out", type=Path, default=Path("./data/processed/splits"))
    parser.add_argument("--config", type=Path, default=Path("./configs/data.yaml"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = yaml.safe_load(args.config.read_text())
    sp = cfg.get("splits", {})
    train_p = float(sp.get("train", 0.7))
    val_p = float(sp.get("val", 0.1))
    test_p = float(sp.get("test", 0.2))

    corpus_path = args.inp / "corpus_v1.parquet"
    if not corpus_path.exists():
        raise SystemExit(f"No existe {corpus_path} — corré `make data` primero.")
    df = pd.read_parquet(corpus_path)
    log.info("corpus: %d filas, %d usuarios", len(df), df["user_id"].nunique())

    user_assignment = assign_users(df, train_p, val_p, test_p, args.seed)
    df["split"] = df["user_id"].map(user_assignment)
    assert df["split"].notna().all(), "quedaron usuarios sin split"

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": args.seed, "fractions": {"train": train_p, "val": val_p, "test": test_p}}
    for split in ("train", "val", "test"):
        sub = df[df["split"] == split]
        out_path = args.out / f"{split}.parquet"
        sub.drop(columns=["split"]).to_parquet(out_path, index=False)
        manifest[split] = {
            "n_rows": int(len(sub)),
            "n_users": int(sub["user_id"].nunique()),
            "by_label": sub["label"].value_counts().to_dict(),
        }
        log.info("→ %s : %d filas, %d usuarios", out_path, len(sub), sub["user_id"].nunique())

    (args.out / "split_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    log.info("OK.")


if __name__ == "__main__":
    main()
