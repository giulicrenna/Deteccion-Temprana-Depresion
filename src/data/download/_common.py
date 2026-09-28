"""Utilidades comunes para los scripts de descarga.

Reglas de hierro (no negociables):
  - CERO autenticación: nunca se leen headers `Authorization`, `Bearer`, etc.
  - Todo script que se conecta a la red expone `download(target_dir) -> dict`
    con los campos: source, license, sha256, download_date, path, n_files.
  - Si el archivo ya está en target_dir y coincide con un manifest previo,
    se salta la descarga (idempotencia).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional

from src.utils.logging import get_logger

log = get_logger(__name__)

MANIFEST_NAME = "manifest.json"


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Calcula el SHA256 de un archivo en chunks (default 1 MB)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            buf = fh.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Descarga HTTP
# ---------------------------------------------------------------------------

def _http_download(url: str, dest: Path, timeout: int = 60, max_retries: int = 3) -> None:
    """Descarga un URL a `dest` con urllib estándar (sin auth, sin headers secretos).

    Reintenta 3 veces con backoff exponencial. Cualquier 4xx/5xx raise.
    """
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            log.info("descargando %s → %s (intento %d/%d)", url, dest, attempt, max_retries)
            req = urllib.request.Request(url, headers={"User-Agent": "tesis-depresion/0.1"})
            with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest, "wb") as out:
                shutil.copyfileobj(resp, out)
            return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_err = exc
            if attempt < max_retries:
                wait = 2 ** attempt
                log.warning("falló intento %d: %s. Reintentando en %ds...", attempt, exc, wait)
                time.sleep(wait)
    raise RuntimeError(f"No se pudo descargar {url}: {last_err}")


def download_file(
    url: str,
    dest: Path,
    expected_sha256: Optional[str] = None,
    timeout: int = 60,
) -> str:
    """Descarga a `dest` y devuelve el SHA256 del archivo final.

    Si `expected_sha256` está dado y no coincide, raise.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and expected_sha256:
        actual = sha256_file(dest)
        if actual == expected_sha256:
            log.info("ya existe %s con sha256 OK — skip", dest)
            return actual
    _http_download(url, dest, timeout=timeout)
    actual = sha256_file(dest)
    if expected_sha256 and actual != expected_sha256:
        raise ValueError(
            f"SHA256 mismatch en {dest}: esperado {expected_sha256}, real {actual}"
        )
    return actual


def download_kaggle_dataset(
    slug: str, dest: Path, version: Optional[int] = None, timeout: int = 300
) -> str:
    """Baja un dataset PÚBLICO de Kaggle como zip, sin credenciales.

    Usa el endpoint `api/v1/datasets/download/<owner>/<dataset>` que Kaggle
    expone para datasets públicos (el mismo del snippet `curl` del sitio).
    `version` fija una versión concreta (reproducibilidad). Devuelve el SHA256.
    """
    url = f"https://www.kaggle.com/api/v1/datasets/download/{slug}"
    if version is not None:
        url += f"?datasetVersionNumber={version}"
    return download_file(url, dest, timeout=timeout)


# ---------------------------------------------------------------------------
# Hugging Face Hub
# ---------------------------------------------------------------------------

def download_hf_dataset(
    target_dir: Path,
    source: str,
    license: str,
    hf_id: str,
    config: Optional[str] = None,
    gated: bool = False,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Baja TODOS los splits de un dataset HF a `target_dir/data.jsonl`.

    Cada fila lleva un campo `_split` con el split original. Para datasets
    gated, el token NO se lee en este código: `datasets` usa el login local
    (`huggingface-cli login`) si existe.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / "data.jsonl"
    if is_already_downloaded(target_dir) and out_path.exists():
        return read_manifest(target_dir) or {}

    from datasets import load_dataset  # import lazy

    try:
        dsd = load_dataset(hf_id, config) if config else load_dataset(hf_id)
    except Exception as exc:
        if gated:
            raise NotImplementedError(
                f"{hf_id} es gated en Hugging Face. Pasos:\n"
                f"  1. Abrir https://huggingface.co/datasets/{hf_id} y aceptar las condiciones.\n"
                "  2. Loguearse localmente: `huggingface-cli login` (token read).\n"
                "  3. Re-correr este script.\n"
                f"Error original: {exc}"
            ) from exc
        raise

    n_rows: dict[str, int] = {}
    with open(out_path, "w", encoding="utf-8") as fh:
        for split, ds in dsd.items():
            n_rows[split] = len(ds)
            for row in ds:
                row = {k: (str(v) if isinstance(v, (_dt.date, _dt.datetime)) else v) for k, v in row.items()}
                row["_split"] = split
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    return write_manifest(
        target_dir=target_dir,
        source=source,
        license=license,
        sha256=sha256_file(out_path),
        path=str(out_path),
        n_files=1,
        extra={"hf_id": hf_id, "hf_config": config, "n_rows": n_rows, **(extra or {})},
    )


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def write_manifest(
    target_dir: Path,
    source: str,
    license: str,
    sha256: str,
    path: str,
    n_files: int,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Genera un manifest.json con metadatos de la descarga."""
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": source,
        "license": license,
        "sha256": sha256,
        "download_date": _dt.datetime.utcnow().isoformat() + "Z",
        "path": path,
        "n_files": n_files,
    }
    if extra:
        manifest.update(extra)
    (target_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def read_manifest(target_dir: Path) -> Optional[dict[str, Any]]:
    """Lee un manifest.json si existe, sino None."""
    p = target_dir / MANIFEST_NAME
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def is_already_downloaded(target_dir: Path) -> bool:
    """True si hay un manifest.json en target_dir y los archivos referenciados existen."""
    m = read_manifest(target_dir)
    if not m:
        return False
    p = Path(m.get("path", ""))
    return p.exists() and p.stat().st_size > 0


def verify_manifest(target_dir: Path) -> bool:
    """Re-hashea el archivo principal y compara contra el manifest."""
    m = read_manifest(target_dir)
    if not m:
        return False
    p = Path(m["path"])
    if not p.exists():
        return False
    return sha256_file(p) == m.get("sha256", "")


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def make_cli(
    module_name: str,
    run: Callable[[Path], dict[str, Any]],
    description: str = "",
) -> None:
    """Crea un entry-point CLI uniforme para un script de descarga.

    Uso en cada download script:
        if __name__ == "__main__":
            make_cli(__name__, lambda out: download(out))
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog=module_name,
        description=description or f"Download script for {module_name}",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Directorio destino (ej: ./data/raw/coello_guilarte)",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    log.info("== %s ==", module_name)
    manifest = run(args.out)
    log.info("manifest: %s", json.dumps(manifest, ensure_ascii=False))


__all__ = [
    "MANIFEST_NAME",
    "sha256_file",
    "download_file",
    "download_kaggle_dataset",
    "download_hf_dataset",
    "write_manifest",
    "read_manifest",
    "is_already_downloaded",
    "verify_manifest",
    "make_cli",
]
