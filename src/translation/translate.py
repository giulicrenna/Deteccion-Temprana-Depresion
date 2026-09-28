"""Traducción EN → ES de `interim/<fuente>/data.parquet` → `data_es.parquet`.

Subcomandos:
  estimate     volumen (docs, caracteres, tokens aprox.) por fuente a traducir.
  run          traduce una fuente con un backend de configs/translation.yaml.
  list-models  lista los modelos que ofrece un backend OpenAI-compatible.

Diseño:
  - Se traduce `text_clean` (ya anonimizado): ningún handle/URL/email sale
    hacia una API externa.
  - Caché JSONL append-only en interim/<fuente>/translations/<sistema>.jsonl
    → el proceso es reanudable (Ctrl+C y volver a correr sigue donde quedó).
  - `--limit N` traduce una muestra estratificada por label (seed fija), útil
    para pilotos y para traducir sólo un subconjunto.
  - Cada traducción lleva `qc_flags` (rechazos del LLM, ratio de longitud
    anómalo, restos de inglés) para revisión manual.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml
from tqdm import tqdm

from src.utils.logging import get_logger
from src.utils.seeds import set_seed

log = get_logger(__name__)

SYSTEM_PROMPT = """Sos un traductor profesional inglés→{target} especializado en salud mental.
Traducí el post de Reddit que te pasa el usuario respetando estas reglas:
1. Fidelidad total: no resumas, no agregues, no omitas, no suavices ni censures
   (incluí insultos, contenido sobre autolesiones o suicidio tal cual).
2. Mantené el registro informal, la persona gramatical (1ra persona singular
   incluida), el tiempo verbal, la intensidad emocional y las palabras
   absolutas ("always"→"siempre", "nothing"→"nada", "never"→"nunca").
3. Conservá emojis, saltos de línea, siglas y nombres de subreddits.
4. No agregues notas, advertencias, líneas de ayuda ni comentarios.
Respondé ÚNICAMENTE con la traducción."""

REFUSAL_RE = re.compile(
    r"(no puedo (ayudar|traducir|cumplir)|i can.?t (help|assist|translate)|i.?m sorry|"
    r"lo siento, pero|l[ií]nea de (ayuda|prevenci[oó]n)|crisis (hotline|line)|988)",
    re.IGNORECASE,
)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
EN_STOP = {"the", "and", "is", "are", "you", "that", "with", "have", "this", "was", "it's", "don't"}
SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+|\n+")


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def load_cfg(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def sources_to_translate(data_cfg: dict[str, Any]) -> list[str]:
    return [
        s
        for s, m in data_cfg["data"]["sources"].items()
        if m.get("enabled", False) and m.get("translate", False)
    ]


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Muestra `n` filas estratificando por label (proporciones originales)."""
    if n >= len(df):
        return df
    out = df.groupby("label").sample(frac=n / len(df), random_state=seed)
    return out.sample(frac=1.0, random_state=seed).head(n)


def qc_flags(src: str, tgt: str) -> list[str]:
    """Heurísticas baratas para marcar traducciones a revisar a mano."""
    flags = []
    if not tgt.strip():
        return ["empty"]
    ratio = len(tgt) / max(1, len(src))
    if ratio < 0.6 or ratio > 2.0:
        flags.append(f"len_ratio={ratio:.2f}")
    if REFUSAL_RE.search(tgt) and not REFUSAL_RE.search(src):
        flags.append("possible_refusal_or_added_note")
    words = re.findall(r"[a-z']+", tgt.lower())
    if words and sum(w in EN_STOP for w in words) / len(words) > 0.08:
        flags.append("english_residue")
    return flags


def read_cache(path: Path) -> dict[str, dict[str, Any]]:
    done: dict[str, dict[str, Any]] = {}
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    done[r["doc_id"]] = r
    return done


# ---------------------------------------------------------------------------
# Backend HF (MarianMT / NLLB) — local en GPU
# ---------------------------------------------------------------------------

def split_segments(text: str, max_words: int) -> list[str]:
    """Parte en oraciones; oraciones muy largas se cortan cada `max_words` palabras."""
    segs: list[str] = []
    for s in SENT_SPLIT_RE.split(text):
        s = s.strip()
        if not s:
            continue
        words = s.split()
        for i in range(0, len(words), max_words):
            segs.append(" ".join(words[i : i + max_words]))
    return segs


class HFTranslator:
    def __init__(self, cfg: dict[str, Any]):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        kw = {"src_lang": cfg["src_lang"]} if "src_lang" in cfg else {}
        self.tok = AutoTokenizer.from_pretrained(cfg["model"], **kw)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(cfg["model"]).to(self.device).eval()
        if self.device == "cuda":
            self.model = self.model.half()
        self.gen_kw: dict[str, Any] = {"num_beams": cfg.get("num_beams", 4), "max_new_tokens": 400}
        if "tgt_lang" in cfg:
            self.gen_kw["forced_bos_token_id"] = self.tok.convert_tokens_to_ids(cfg["tgt_lang"])
        log.info("HF backend %s en %s", cfg["model"], self.device)

    def _translate_segments(self, segs: list[str]) -> list[str]:
        import torch

        bs = self.cfg.get("batch_size", 16)
        order = sorted(range(len(segs)), key=lambda i: len(segs[i]))  # menos padding
        out = [""] * len(segs)
        for i in range(0, len(order), bs):
            idx = order[i : i + bs]
            enc = self.tok([segs[j] for j in idx], return_tensors="pt", padding=True,
                           truncation=True, max_length=512).to(self.device)
            with torch.inference_mode():
                gen = self.model.generate(**enc, **self.gen_kw)
            for j, t in zip(idx, self.tok.batch_decode(gen, skip_special_tokens=True)):
                out[j] = t
        return out

    def translate_docs(self, texts: list[str]) -> list[str]:
        max_w = self.cfg.get("max_segment_words", 120)
        per_doc = [split_segments(t, max_w) for t in texts]
        flat = [s for segs in per_doc for s in segs]
        flat_tr = iter(self._translate_segments(flat))
        return [" ".join(next(flat_tr) for _ in segs) for segs in per_doc]


# ---------------------------------------------------------------------------
# Backend OpenAI-compatible (NVIDIA NIM / Nemotron, Ollama, vLLM, OpenRouter...)
# ---------------------------------------------------------------------------

class RateLimiter:
    def __init__(self, rpm: int):
        self.interval = 60.0 / rpm if rpm else 0.0
        self.lock = threading.Lock()
        self.next_t = 0.0

    def wait(self) -> None:
        if not self.interval:
            return
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next_t)
            self.next_t = t + self.interval
        time.sleep(max(0.0, t - now))


class OpenAITranslator:
    def __init__(self, cfg: dict[str, Any], target: str, max_chars: int):
        import requests

        self.cfg = cfg
        self.session = requests.Session()
        key_env = cfg.get("api_key_env") or ""
        key = os.environ.get(key_env, "") if key_env else ""
        if key_env and not key:
            raise SystemExit(f"Falta la variable de entorno {key_env} (ponela en .env).")
        if key:
            self.session.headers["Authorization"] = f"Bearer {key}"
        self.url = cfg["base_url"].rstrip("/") + "/chat/completions"
        prefix = cfg.get("system_prefix", "")
        self.system = (prefix + "\n" if prefix else "") + SYSTEM_PROMPT.format(target=target)
        self.limiter = RateLimiter(int(cfg.get("rpm", 0)))
        self.max_chars = max_chars

    def _chat(self, text: str) -> str:
        body = {
            "model": self.cfg["model"],
            "messages": [
                {"role": "system", "content": self.system},
                {"role": "user", "content": text},
            ],
            "temperature": self.cfg.get("temperature", 0.0),
            "max_tokens": self.cfg.get("max_tokens", 4096),
        }
        for attempt in range(6):
            self.limiter.wait()
            try:
                r = self.session.post(self.url, json=body, timeout=180)
            except Exception as exc:  # red caída, timeout
                log.warning("error de red (%s), reintento %d", exc, attempt + 1)
                time.sleep(2**attempt)
                continue
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(min(60, 2**attempt * 5))
                continue
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"] or ""
            return THINK_RE.sub("", content).strip()
        raise RuntimeError(f"sin respuesta tras reintentos (último status {r.status_code})")

    def translate_one(self, text: str) -> str:
        if len(text) <= self.max_chars:
            return self._chat(text)
        # Posts muy largos: por párrafos, agrupados hasta max_chars.
        chunks, cur = [], ""
        for para in text.split("\n"):
            if cur and len(cur) + len(para) > self.max_chars:
                chunks.append(cur)
                cur = ""
            cur += para + "\n"
        chunks.append(cur)
        return "\n".join(self._chat(c.strip()) for c in chunks if c.strip())

    def list_models(self) -> list[str]:
        r = self.session.get(self.cfg["base_url"].rstrip("/") + "/models", timeout=60)
        r.raise_for_status()
        return sorted(m["id"] for m in r.json()["data"])


# ---------------------------------------------------------------------------
# Subcomandos
# ---------------------------------------------------------------------------

def cmd_estimate(args: argparse.Namespace) -> None:
    data_cfg = load_cfg(args.data_config)
    rows = []
    for src in sources_to_translate(data_cfg):
        p = args.interim / src / "data.parquet"
        if not p.exists():
            log.warning("falta %s (correr make_dataset)", p)
            continue
        t = pd.read_parquet(p, columns=["text_clean"])["text_clean"]
        chars = int(t.str.len().sum())
        rows.append({
            "source": src,
            "docs": len(t),
            "chars": chars,
            "words": int(t.str.split().str.len().sum()),
            "tokens_en_aprox": chars // 4,
            "chars_mediana": int(t.str.len().median()),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No hay fuentes para estimar.")
    total = df.drop(columns=["source", "chars_mediana"]).sum()
    df.loc[len(df)] = {"source": "TOTAL", **total.to_dict(), "chars_mediana": None}
    print(df.to_string(index=False))


def cmd_list_models(args: argparse.Namespace) -> None:
    tr_cfg = load_cfg(args.config)
    b = tr_cfg["backends"][args.backend]
    t = OpenAITranslator(b, tr_cfg["target_language"], tr_cfg.get("llm_max_chars", 6000))
    print("\n".join(t.list_models()))


def _write_output(df: pd.DataFrame, done: dict[str, dict[str, Any]], out: Path) -> None:
    res = pd.DataFrame([done[d] for d in df["doc_id"] if d in done])
    res.to_parquet(out, index=False)
    n_flag = int((res["qc_flags"].str.len() > 0).sum()) if len(res) else 0
    log.info("→ %s : %d traducciones (%d con qc_flags)", out, len(res), n_flag)


def cmd_run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    tr_cfg = load_cfg(args.config)
    bcfg = dict(tr_cfg["backends"][args.backend])
    if args.model:
        bcfg["model"] = args.model
    system_name = f"{args.backend}:{bcfg['model']}"

    src_dir = args.interim / args.source
    df = pd.read_parquet(src_dir / "data.parquet", columns=["doc_id", "label", "text_clean"])
    if args.limit:
        df = stratified_sample(df, args.limit, args.seed)
    cache_path = src_dir / "translations" / (re.sub(r"[^\w.-]+", "_", system_name) + ".jsonl")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    done = read_cache(cache_path)
    todo = df[~df["doc_id"].isin(done)]
    log.info("%s: %d docs (%d ya traducidos, %d pendientes) con %s",
             args.source, len(df), len(df) - len(todo), len(todo), system_name)

    def record(doc_id: str, src: str, tgt: str) -> dict[str, Any]:
        return {"doc_id": doc_id, "text_es": tgt, "mt_system": system_name,
                "qc_flags": ",".join(qc_flags(src, tgt))}

    with open(cache_path, "a", encoding="utf-8") as fh:
        if bcfg["type"] == "hf":
            tr = HFTranslator(bcfg)
            t0 = time.time()  # sin contar la carga del modelo
            chunk = args.chunk
            for i in tqdm(range(0, len(todo), chunk), desc="traduciendo", unit="chunk"):
                part = todo.iloc[i : i + chunk]
                outs = tr.translate_docs(part["text_clean"].tolist())
                for d, s, o in zip(part["doc_id"], part["text_clean"], outs):
                    done[d] = record(d, s, o)
                    fh.write(json.dumps(done[d], ensure_ascii=False) + "\n")
                fh.flush()
        else:
            tr = OpenAITranslator(bcfg, tr_cfg["target_language"], tr_cfg.get("llm_max_chars", 6000))
            t0 = time.time()
            with ThreadPoolExecutor(max_workers=int(bcfg.get("concurrency", 4))) as ex:
                futs = {ex.submit(tr.translate_one, s): (d, s)
                        for d, s in zip(todo["doc_id"], todo["text_clean"])}
                for f in tqdm(as_completed(futs), total=len(futs), desc="traduciendo"):
                    d, s = futs[f]
                    try:
                        o = f.result()
                    except Exception as exc:
                        log.error("falló %s: %s", d, exc)
                        continue
                    done[d] = record(d, s, o)
                    fh.write(json.dumps(done[d], ensure_ascii=False) + "\n")
                    fh.flush()

    dt = time.time() - t0
    if len(todo):
        log.info("throughput: %.2f docs/s (%.0f s para %d docs)", len(todo) / dt, dt, len(todo))
    _write_output(df, done, src_dir / args.out_name)


def main(argv: Iterable[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="src.translation.translate")
    p.add_argument("--interim", type=Path, default=Path("./data/interim"))
    p.add_argument("--data-config", type=Path, default=Path("./configs/data.yaml"))
    p.add_argument("--config", type=Path, default=Path("./configs/translation.yaml"))
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("estimate")

    lm = sub.add_parser("list-models")
    lm.add_argument("--backend", required=True)

    r = sub.add_parser("run")
    r.add_argument("--source", required=True)
    r.add_argument("--backend", required=True, help="clave de configs/translation.yaml")
    r.add_argument("--model", default=None, help="override del modelo del backend")
    r.add_argument("--limit", type=int, default=0, help="muestra estratificada de N docs")
    r.add_argument("--chunk", type=int, default=64, help="docs por lote (backend hf)")
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--out-name", default="data_es.parquet",
                   help="usar otro nombre para pilotos (no lo toma merge_corpora)")

    args = p.parse_args(list(argv) if argv is not None else None)
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    {"estimate": cmd_estimate, "list-models": cmd_list_models, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
