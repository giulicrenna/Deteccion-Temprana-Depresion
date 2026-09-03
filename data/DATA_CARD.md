# DATA_CARD — Proveniencia, licencias y anonimización

> Documento de trazabilidad ética. Cualquier persona que abra el repo
> tiene que poder responder **de dónde viene cada documento** y **bajo
> qué licencia se puede usar**.

## Resumen de fuentes

| # | Fuente | URL pública | Licencia | Idioma | Plataforma | Tamaño | Estado | Decisión |
|---|---|---|---|---|---|---|---|---|
| 1 | Coello-Guilarte 2019 (CrossLingualDepression) | https://ccc.inaoep.mx/~mmontesg/resources/CrossLingualDepression.zip | Research use, citation required | es | Twitter | 53MB + 124MB (177MB total) | ✅ wget | **Incluido** |
| 2 | MentalRiskES (muestra GitHub) | https://github.com/sinai-uja/corpusMentalRiskES | Gated — autores | es | Reddit-like | ~5MB (zip cifrado) | ⚠️ cifrado | **Stub** (gated) |
| 3 | MentalRiskES (Zenodo 8055604) | https://zenodo.org/record/8055604 | — | es | — | — | ⚠️ es PRECOM-SM, no MentalRiskES | **Stub** |
| 4 | ReDSM5 paraphrase sample | https://huggingface.co/datasets/irlab-udc/redsm5 | MIT | en/es | Reddit | 25 entries | ✅ HF Hub (público) | **Incluido** |
| 5 | EmoEvalEs | https://huggingface.co/datasets (varios candidatos) | Research use | es | Twitter | chico | ⚠️ candidato a chequear | **Incluido (best-effort)** |
| 6 | SWMH-ES | https://huggingface.co/datasets (varios candidatos) | Mixed | es | Reddit (traducido) | variable | ⚠️ candidato a chequear | **Incluido (best-effort)** |
| 7 | Figshare mental health ES | https://figshare.com/articles/dataset/28498766 | Research use | es | Twitter | variable | ❌ WAF challenge | **Stub** |
| 8 | MentalRiskES completo (45k) | autores (amarmol@ujaen.es / amontejo@ujaen.es) | Gated | es | Reddit-like | 45k mensajes | ⚠️ gated | **Stub** |
| 9 | Leis et al. 2019 | Kaggle / mail a F. Ronzano | gated | es | Twitter | — | ⚠️ gated | **Stub** |
| 10 | DAIC-WOZ | https://dcapswoz.ict.usc.edu | DUA | en | entrevistas | grande | ⚠️ gated | **Stub** |
| 11 | RSDD (Bucuram 2025) | — | — | es | Twitter | — | ❌ N/AV | **Stub** |
| 12 | Mini-corpus sintético | `src/data/synthetic/` | Generated | es | — | 60 mensajes | ✅ local | **Incluido (sólo dev)** |

## Decisión de corpus base para la tesis

> **Coello-Guilarte 2019 es el corpus principal** porque es el único
> que cumple TODAS las siguientes condiciones simultáneamente:
> 1. Descargable sin autenticación (wget plano).
> 2. En español.
> 3. Binario (depresivo / no-depresivo) — mapea a las 2 clases
>    operativas del modelo.
> 4. Citado en publicaciones con revisión por pares.
> 5. Volumen suficiente para entrenar y validar (≥10k usuarios únicos
>    una vez procesados).

El resto se incluye como **complemento** (EmoEvalEs, ReDSM5, SWMH-ES,
mini-corpus sintético) y los **stubs** documentan el camino para crecer
hacia corpus más grandes si en el futuro se obtiene acceso formal.

## Anonimización

Aplicada en `src/data/make_dataset.py` antes de escribir a `interim/`:

| Patrón | Regex | Reemplazo |
|---|---|---|
| URLs | `https?://\S+\|www\.\S+` | vacío |
| Menciones | `@\w+` | vacío |
| Emails | `[\w._%+-]+@[\w.-]+\.[A-Za-z]{2,}` | vacío |
| Teléfonos | `\+?\d[\d\s().-]{7,}\d` | vacío |
| Hashtags | `#\w+` | vacío (configurable) |
| "RT" prefijo | `^RT\s+` | vacío |
| Whitespace | `\s+` | un solo espacio |

**Limitaciones reconocidas:**

- No se aplica NER para detectar nombres propios. Esto puede dejar
  nombres de personas en el texto si el usuario los escribió. Es una
  decisión consciente para v1 (los falsos negativos de NER son
  peligrosos). En v2, agregar `es_core_news_md` + revisión manual de
  muestra.
- Los `user_id` se hashean con SHA-256 (primeros 16 chars) → no se
  puede revertir al handle original.
- Los `doc_id` también son SHA-256 → no se puede revertir al id de
  Twitter.

## Contacto de los autores de cada fuente

- Coello-Guilarte: ver paper en https://ccc.inaoep.mx/~mmontesg/
- MentalRiskES: Ana Martín-Maldonado <amarmol@ujaen.es>, Ángel Montejo-Ráez <amontejo@ujaen.es>
- Leis: Francesco Ronzano <francesco.ronzano@upf.edu>
- DAIC-WOZ: ver https://dcapswoz.ict.usc.edu

## Ethical considerations (resumen)

- **Solo datos públicos** (Twitter, Reddit-like). No scrapeamos DMs.
- **No subimos datos crudos a git** (`data/raw/`, `interim/`,
  `processed/` están en `.gitignore` excepto `.gitkeep`).
- **El modelo final nunca debería usarse como screening clínico
  unilateral**: la sección 8 de la tesis discute este punto.
- La anonimización se hace **antes** de cualquier análisis → no se
  filtra información personal a artefactos versionados.

## Reproducibilidad

- Hashes SHA-256 de cada archivo crudo se guardan en
  `data/raw/<fuente>/manifest.json`.
- Split user-level estratificado (no document-level) en
  `data/processed/splits/split_manifest.json` → siempre los mismos
  `user_id` van al mismo fold.
- `set_seed(42)` se llama en cada script de procesamiento.

## Histórico de cambios

| Fecha | Cambio | Autor |
|---|---|---|
| 2026-09-03 | Creación inicial del DATA_CARD | Crenna, Pace |
