"""Coeficientes de la regresión logística TF-IDF y contraste con el léxico psicolingüístico (Hs4).

Con `classes_ = [0, 2]`, un coeficiente positivo empuja hacia la clase depresiva. Para cada
categoría del léxico de Leis et al. se mide si sus términos aparecen en el top-K de
coeficientes depresivos más de lo esperado por azar (test exacto de Fisher sobre los
unigramas del vocabulario).

Uso:
    python -m src.xai.linear_coefficients --model models/baseline_logreg.joblib
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu

from src.evaluation.bootstrap import DEP
from src.features.liwc_counts import _load_lexicon
from src.utils.logging import get_logger

logger = get_logger("linear_coefficients")

DEFAULT_LEXICON = Path("src/features/lexicons/leis_lexicon.csv")


def coefficient_table(pipeline) -> pd.DataFrame:
    """Términos del vocabulario TF-IDF con su coeficiente orientado a la clase depresiva."""
    clf = pipeline.named_steps["clf"]
    terms = pipeline.named_steps["tfidf"].get_feature_names_out()
    sign = 1.0 if list(clf.classes_)[1] == DEP else -1.0
    coef = sign * clf.coef_[0]
    df = pd.DataFrame({"term": terms, "coef": coef})
    df["is_unigram"] = ~df["term"].str.contains(" ")
    df["rank_dep"] = df["coef"].rank(ascending=False, method="first").astype(int)
    return df.sort_values("rank_dep").reset_index(drop=True)


def lexicon_enrichment(
    coefs: pd.DataFrame, lexicon: dict[str, list[str]], top_k: int
) -> pd.DataFrame:
    """Enriquecimiento de cada categoría léxica en los extremos de la distribución de coeficientes."""
    uni = coefs[coefs["is_unigram"]].reset_index(drop=True)
    order = uni["coef"].rank(ascending=False, method="first")
    in_top_dep = (order <= top_k).to_numpy()
    in_top_ctrl = (order > len(uni) - top_k).to_numpy()

    rows = []
    for category, words in lexicon.items():
        member = uni["term"].isin(set(words)).to_numpy()
        n_member = int(member.sum())
        row = {
            "category": category,
            "n_terms_lexicon": len(set(words)),
            "n_terms_in_vocab": n_member,
            "top_k": top_k,
        }
        if n_member == 0:
            rows.append(row)
            continue
        member_coef = uni.loc[member, "coef"]
        table = [
            [int((member & in_top_dep).sum()), int((member & ~in_top_dep).sum())],
            [int((~member & in_top_dep).sum()), int((~member & ~in_top_dep).sum())],
        ]
        odds, p_dep = fisher_exact(table, alternative="greater")
        _, p_mw = mannwhitneyu(member_coef, uni.loc[~member, "coef"], alternative="greater")
        row.update(
            {
                "mean_coef": float(member_coef.mean()),
                "median_coef": float(member_coef.median()),
                "share_positive_coef": float((member_coef > 0).mean()),
                "n_in_top_dep": table[0][0],
                "n_in_top_control": int((member & in_top_ctrl).sum()),
                "odds_ratio_top_dep": float(odds),
                "p_fisher_top_dep": float(p_dep),
                "p_mannwhitney_greater": float(p_mw),
                "terms_in_top_dep": ", ".join(uni.loc[member & in_top_dep, "term"]),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def lexicon_term_detail(coefs: pd.DataFrame, lexicon: dict[str, list[str]]) -> pd.DataFrame:
    """Coeficiente y posición de cada término del léxico presente en el vocabulario."""
    term_to_cat = {w: cat for cat, words in lexicon.items() for w in words}
    uni = coefs[coefs["is_unigram"]].copy()
    uni["percentile_dep"] = 1 - (uni["coef"].rank(ascending=False) - 1) / len(uni)
    detail = uni[uni["term"].isin(term_to_cat.keys())].copy()
    detail.insert(0, "category", detail["term"].map(term_to_cat))
    return detail.sort_values(["category", "coef"], ascending=[True, False])


def run(
    model_path: Path, lexicon_path: Path, out_dir: Path, top_n: int, top_k: int, tag: str
) -> pd.DataFrame:
    """Exporta coeficientes extremos, detalle léxico y enriquecimiento por categoría."""
    out_dir.mkdir(parents=True, exist_ok=True)
    coefs = coefficient_table(joblib.load(model_path))
    lexicon = _load_lexicon(lexicon_path)

    extremes = pd.concat(
        [coefs.head(top_n).assign(side="depresivo"), coefs.tail(top_n).assign(side="control")]
    )
    extremes.to_csv(out_dir / f"lr_coefficients_{tag}.csv", index=False)
    lexicon_term_detail(coefs, lexicon).to_csv(
        out_dir / f"lr_lexicon_terms_{tag}.csv", index=False
    )
    enrichment = lexicon_enrichment(coefs, lexicon, top_k)
    enrichment.insert(0, "model", tag)
    enrichment.to_csv(out_dir / f"hs4_lexicon_enrichment_{tag}.csv", index=False)
    logger.info(f"[{tag}] vocabulario={len(coefs):,} | tablas en {out_dir}")
    return enrichment


def main() -> None:
    parser = argparse.ArgumentParser(description="Coeficientes LR y contraste léxico (Hs4).")
    parser.add_argument("--model", default="models/baseline_logreg.joblib")
    parser.add_argument("--tag", default="lr_mono")
    parser.add_argument("--lexicon", default=str(DEFAULT_LEXICON))
    parser.add_argument("--out-dir", default="reports/tables")
    parser.add_argument("--top-n", type=int, default=100, help="Términos exportados por lado.")
    parser.add_argument("--top-k", type=int, default=500, help="Corte para el test de Fisher.")
    args = parser.parse_args()
    result = run(
        Path(args.model), Path(args.lexicon), Path(args.out_dir), args.top_n, args.top_k, args.tag
    )
    print(result.drop(columns="terms_in_top_dep", errors="ignore").round(4).to_string())


if __name__ == "__main__":
    main()
