"""Limpieza + anonimización: raw/ → interim/.

Aplica a cada fuente:
  1. Lectura del formato crudo.
  2. Anonimización (URLs, menciones, emails, teléfonos).
  3. Proyección al esquema unificado (Parquet).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml
from tqdm import tqdm

from src.utils.logging import get_logger
from src.utils.seeds import set_seed

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Anonimización
# ---------------------------------------------------------------------------

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
REDDIT_USER_RE = re.compile(r"(?<![\w/])/?u/[A-Za-z0-9_-]+")
HASHTAG_RE = re.compile(r"#\w+")
EMAIL_RE = re.compile(r"\b[\w._%+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"\+?\d[\d\s().-]{7,}\d")
WS_RE = re.compile(r"\s+")
RT_RE = re.compile(r"^RT\s+", re.IGNORECASE)


def anonymize(text: str) -> str:
    """Quita URLs, menciones, emails, teléfonos y normaliza whitespace."""
    if not isinstance(text, str):
        return ""
    t = URL_RE.sub("", text)
    t = EMAIL_RE.sub("", t)
    t = MENTION_RE.sub("", t)
    t = REDDIT_USER_RE.sub("", t)
    t = PHONE_RE.sub("", t)
    t = HASHTAG_RE.sub("", t)
    t = RT_RE.sub("", t)
    t = WS_RE.sub(" ", t).strip()
    return t


def hash_id(*parts: str) -> str:
    """Hash determinístico para doc_id / user_id anonimizados."""
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Loaders por fuente
# ---------------------------------------------------------------------------


def _parse_twitter_date(s: str) -> str | None:
    """Parsea 'Thu Nov 21 03:45:28 +0000 2013' → ISO8601 UTC."""
    try:
        from datetime import datetime

        return datetime.strptime(s, "%a %b %d %H:%M:%S %z %Y").astimezone(timezone.utc).isoformat()
    except Exception:
        return None


def load_coello_guilarte(raw_dir: Path) -> pd.DataFrame:
    """Carga los 2 archivos del corpus Coello-Guilarte.

    `tweets_Español_depresivos.json` es un TAR (no JSON) que contiene
    `tweets_cuentas_inaoe.json` (JSONL). `tweets_Español_no-depresivos.json`
    es JSONL directo. Mismo formato en ambos: una línea por usuario
    con un array `tweets`.
    """
    zip_path = next(raw_dir.glob("*.zip"), None)
    if zip_path is None:
        raise FileNotFoundError(f"No se encontró zip de Coello-Guilarte en {raw_dir}")
    extract_dir = raw_dir / "_extracted"
    extract_dir.mkdir(exist_ok=True)

    # Extraer zip con filenames correctos (los nombres tienen \u00f1 → ñ).
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            target_name = member.replace("Espa#U00f1ol", "Español")
            zf.extract(member, path=extract_dir)
            src = extract_dir / member
            dst = extract_dir / target_name
            if src != dst and src.exists():
                src.rename(dst)

    dep_path = extract_dir / "tweets_Español_depresivos.json"
    nodep_path = extract_dir / "tweets_Español_no-depresivos.json"

    # El archivo depresivo es un TAR dentro del zip; los no-depresivos es JSONL.
    if tarfile.is_tarfile(dep_path):
        with tarfile.open(dep_path) as tf:
            tf.extractall(path=extract_dir / "_dep")
        dep_jsonl = extract_dir / "_dep" / "tweets_cuentas_inaoe.json"
    else:
        dep_jsonl = dep_path

    rows: list[dict[str, Any]] = []
    for path, label in [(dep_jsonl, 2), (nodep_path, 0)]:
        if not path.exists():
            log.warning("no existe %s — saltando", path)
            continue
        with open(path, encoding="utf-8") as fh:
            for line in tqdm(fh, desc=f"leyendo {path.name}", unit=" users"):
                line = line.strip()
                if not line:
                    continue
                try:
                    user_obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                user_handle = user_obj.get("user", "anon")
                user_id = hash_id("coello", user_handle)
                for tw in user_obj.get("tweets", []):
                    text = tw.get("text", "")
                    ts = _parse_twitter_date(tw.get("created_at", "")) or ""
                    rows.append(
                        {
                            "doc_id": hash_id("coello", str(tw.get("id", ""))),
                            "user_id": user_id,
                            "source": "coello_guilarte",
                            "timestamp": ts,
                            "text_raw": text,
                            "text_clean": anonymize(text),
                            "label": label,
                            "label_source": "crowd_annotation",
                            "lang": "es",
                        }
                    )
    return pd.DataFrame(rows)


def load_synthetic(raw_dir: Path) -> pd.DataFrame:
    """Carga el mini-corpus sintético."""
    p = raw_dir / "data.jsonl"
    rows: list[dict[str, Any]] = []
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            text = r["text"]
            rows.append(
                {
                    "doc_id": hash_id("synthetic", r["user_id"], text),
                    "user_id": hash_id("synthetic_user", r["user_id"]),
                    "source": "synthetic",
                    "timestamp": r.get("timestamp", ""),
                    "text_raw": text,
                    "text_clean": anonymize(text),
                    "label": int(r["label"]),
                    "label_source": "synthetic",
                    "lang": "es",
                }
            )
    return pd.DataFrame(rows)


def load_hf_jsonl(raw_dir: Path, source_name: str, label_map: dict | None = None) -> pd.DataFrame:
    """Carga un JSONL genérico desde HF y mapea labels si hace falta."""
    p = raw_dir / "data.jsonl"
    rows: list[dict[str, Any]] = []
    with open(p, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            text = r.get("text") or r.get("sentence") or r.get("tweet") or ""
            label = r.get("label", r.get("emotion", 0))
            if isinstance(label, str) and label_map:
                label = label_map.get(label, 0)
            rows.append(
                {
                    "doc_id": hash_id(source_name, str(i)),
                    "user_id": hash_id(source_name, str(r.get("user_id", i))),
                    "source": source_name,
                    "timestamp": r.get("timestamp", r.get("date", "")),
                    "text_raw": text,
                    "text_clean": anonymize(text),
                    "label": int(label) if label is not None else 0,
                    "label_source": "hf_dataset",
                    "lang": r.get("lang", "es"),
                }
            )
    return pd.DataFrame(rows)


def _iter_jsonl(p: Path) -> Iterable[dict[str, Any]]:
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _row(
    source: str,
    key: str,
    user: str,
    text: str,
    label: int,
    label_source: str,
    lang: str,
    timestamp: str = "",
    orig_split: str = "",
) -> dict[str, Any]:
    return {
        "doc_id": hash_id(source, key),
        "user_id": hash_id(source + "_user", user),
        "source": source,
        "timestamp": timestamp,
        "text_raw": text,
        "text_clean": anonymize(text),
        "label": label,
        "label_source": label_source,
        "lang": lang,
        "orig_split": orig_split,
    }


def _utc_iso(value: Any) -> str:
    """Epoch (int/float/str numérico) o fecha ISO → ISO8601 UTC. Vacío si no parsea."""
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OverflowError):
        pass
    try:
        ts = pd.Timestamp(value)
        if pd.isna(ts):
            return ""
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        return ts.isoformat()
    except (TypeError, ValueError):
        return ""


def load_kaggle_sdd(raw_dir: Path) -> pd.DataFrame:
    """Kaggle SDD v13: r/depression y r/SuicideWatch → 2, r/teenagers → 0.

    El subreddit queda en `label_source` para poder excluir SuicideWatch
    (riesgo suicida ≠ depresión) en experimentos de sensibilidad.
    Sin autor en el CSV → cada post es su propio "usuario".
    """
    zip_path = raw_dir / "kaggle_sdd.zip"
    with zipfile.ZipFile(zip_path) as zf, zf.open("SuicideAndDepression_Detection.csv") as fh:
        df = pd.read_csv(fh)
    labels = {"depression": 2, "SuicideWatch": 2, "teenagers": 0}
    rows = [
        _row("kaggle_sdd", str(i), str(i), str(text), labels[cls],
             f"subreddit_membership:r/{cls}", "en")
        for i, (text, cls) in enumerate(zip(df["text"], df["class"]))
        if cls in labels
    ]
    return pd.DataFrame(rows)


def load_reddit_mh_posts(raw_dir: Path) -> pd.DataFrame:
    """solomonk/reddit_mental_health_posts: r/depression → 2, resto (ADHD, OCD, ...) → 0.

    Los otros subreddits de salud mental son negativos "difíciles": controlan
    que el modelo aprenda depresión y no "habla de salud mental" en general.
    """
    rows = []
    for r in _iter_jsonl(raw_dir / "data.jsonl"):
        body = (r.get("body") or "").strip()
        if body in {"[removed]", "[deleted]"}:
            body = ""
        text = "\n\n".join(t for t in [(r.get("title") or "").strip(), body] if t)
        if not text:
            continue
        sub = str(r.get("subreddit", ""))
        author = str(r.get("author") or "")
        if author in {"", "[deleted]", "None"}:
            author = f"anon_{r.get('id')}"
        rows.append(
            _row("reddit_mh_posts", str(r.get("id")), author, text,
                 2 if sub.lower() == "depression" else 0,
                 f"subreddit_membership:r/{sub}", "en", _utc_iso(r.get("created_utc")))
        )
    return pd.DataFrame(rows)


def load_depression_reddit(raw_dir: Path) -> pd.DataFrame:
    """mrjunos/depression-reddit-cleaned: label 1 → 2, 0 → 0."""
    rows = [
        _row("depression_reddit", str(i), str(i), str(r["text"]), 2 if int(r["label"]) == 1 else 0,
             "subreddit_membership", "en", orig_split=r.get("_split", ""))
        for i, r in enumerate(_iter_jsonl(raw_dir / "data.jsonl"))
    ]
    return pd.DataFrame(rows)


def load_swmh(raw_dir: Path) -> pd.DataFrame:
    """AIMH/SWMH: `self.depression` → 2, resto de subreddits → 0."""
    rows = []
    for i, r in enumerate(_iter_jsonl(raw_dir / "data.jsonl")):
        sub = str(r.get("label", "")).replace("self.", "")
        rows.append(
            _row("swmh", str(i), str(i), str(r.get("text", "")),
                 2 if sub.lower() == "depression" else 0,
                 f"subreddit_membership:r/{sub}", "en", orig_split=r.get("_split", ""))
        )
    return pd.DataFrame(rows)


def load_prevenia_es(raw_dir: Path) -> pd.DataFrame:
    """PrevenIA: Label 1 (intención suicida) → 2, 0 → 0. Ya en español."""
    rows = [
        _row("prevenia_es", str(i), str(i), str(r.get("Text", "")), 2 if int(r["Label"]) == 1 else 0,
             f"suicide_intent:{r.get('dataset', '')}", "es", orig_split=r.get("_split", ""))
        for i, r in enumerate(_iter_jsonl(raw_dir / "data.jsonl"))
    ]
    return pd.DataFrame(rows)


LOADERS = {
    "kaggle_sdd": load_kaggle_sdd,
    "reddit_mh_posts": load_reddit_mh_posts,
    "depression_reddit": load_depression_reddit,
    "swmh": load_swmh,
    "prevenia_es": load_prevenia_es,
}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def process_one(
    source: str,
    raw_root: Path,
    interim_root: Path,
    label_map: dict | None,
) -> Path | None:
    """Procesa una fuente: raw → interim. Devuelve la ruta del parquet o None."""
    raw_dir = raw_root / source
    if not raw_dir.exists():
        log.warning("no existe raw/%s — saltando", source)
        return None

    log.info("procesando %s ...", source)
    if source == "coello_guilarte":
        df = load_coello_guilarte(raw_dir)
    elif source == "synthetic":
        df = load_synthetic(raw_dir)
    elif source in LOADERS:
        df = LOADERS[source](raw_dir)
    elif source in {"emoevales", "swmh_es"}:
        df = load_hf_jsonl(raw_dir, source, label_map=label_map)
    else:
        log.warning("fuente %s no implementada — saltando", source)
        return None

    if df.empty:
        log.warning("fuente %s vacía — saltando", source)
        return None

    # Fuentes nuevas: textos vacíos tras anonimizar no aportan nada (ni se traducen).
    # Coello-Guilarte se deja intacto para no alterar las tablas de EDA ya reportadas.
    if source in LOADERS:
        df = df[df["text_clean"].str.len() > 0].reset_index(drop=True)

    out_dir = interim_root / source
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "data.parquet"
    df.to_parquet(out_path, index=False)
    log.info("→ %s : %d filas", out_path, len(df))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="raw", type=Path, default=Path("./data/raw"))
    parser.add_argument("--out", dest="interim", type=Path, default=Path("./data/interim"))
    parser.add_argument("--config", type=Path, default=Path("./configs/data.yaml"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    label_map = cfg.get("data", {}).get("label_map", None)

    sources = [
        s
        for s, meta in cfg["data"]["sources"].items()
        if meta.get("enabled", False) and (args.raw / s / "manifest.json").exists()
    ]

    written: list[Path] = []
    for src in sources:
        try:
            p = process_one(src, args.raw, args.interim, label_map)
            if p:
                written.append(p)
        except Exception as exc:
            log.error("falló %s: %s", src, exc)

    log.info("OK. %d archivos escritos en %s", len(written), args.interim)


if __name__ == "__main__":
    main()
