import sys, json
from pathlib import Path
import pandas as pd

for name in ["splits", "splits_multi"]:
    d = Path("data/processed") / name
    man = json.load(open(d / "split_manifest.json"))
    sp = {s: pd.read_parquet(d / f"{s}.parquet") for s in ["train", "val", "test"]}
    print(f"\n######## {name}  cols={list(sp['train'].columns)}")
    for s, df in sp.items():
        ok = len(df) == man[s]["n_rows"] and df["user_id"].nunique() == man[s]["n_users"]
        empty = (df["text_clean"].fillna("").str.strip() == "").sum()
        print(f"{s}: rows={len(df):,} users={df['user_id'].nunique():,} manifest_ok={ok} "
              f"empty={empty:,} ({empty/len(df):.2%}) labels={df['label'].value_counts().to_dict()} "
              f"dup_internal={df['text_clean'].duplicated().sum():,}")
    u = {s: set(df["user_id"]) for s, df in sp.items()}
    print("user overlap tr-va/tr-te/va-te:", len(u["train"] & u["val"]), len(u["train"] & u["test"]), len(u["val"] & u["test"]))
    t = {s: set(df["text_clean"].fillna("").str.strip()) - {""} for s, df in sp.items()}
    print("text overlap tr-va/tr-te/va-te:", len(t["train"] & t["val"]), len(t["train"] & t["test"]), len(t["val"] & t["test"]))
    te = sp["test"]; te_in_tr = te["text_clean"].fillna("").str.strip().isin(t["train"]).sum()
    print(f"test rows whose text appears in train: {te_in_tr:,} ({te_in_tr/len(te):.2%})")
    if "corpus_source" in sp["train"].columns:
        tab = pd.concat([df.assign(split=s) for s, df in sp.items()], ignore_index=True)
        print(pd.crosstab([tab["corpus_source"], tab["label"]], tab["split"]).to_string())
        print("users per source/split:")
        print(tab.groupby(["corpus_source", "split"])["user_id"].nunique().unstack().to_string())
    # usuarios con etiqueta mixta
    allu = pd.concat(sp.values(), ignore_index=True).groupby("user_id")["label"].nunique()
    print("users with >1 label:", int((allu > 1).sum()))

# mono vs multi: usuarios Coello del test multi que estaban en train mono
mono_tr = set(pd.read_parquet("data/processed/splits/train.parquet", columns=["user_id"])["user_id"])
mt = pd.read_parquet("data/processed/splits_multi/test.parquet", columns=["user_id", "corpus_source"])
mt_c = set(mt.loc[mt.corpus_source == "coello_guilarte", "user_id"])
print("\nCoello users in multi-test also in mono-train:", len(mt_c & mono_tr), "/", len(mt_c))
