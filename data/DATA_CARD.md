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
| 13 | Suicide and Depression Detection **v13** (Komati et al., IEEE 2021) | https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch | CC BY-SA 4.0 | en | Reddit | 348.103 posts | ✅ endpoint público Kaggle | **Incluido tras traducción** |
| 14 | Reddit Mental Health Posts | https://huggingface.co/datasets/solomonk/reddit_mental_health_posts | sin declarar | en | Reddit | 151.274 posts | ✅ HF público | **Incluido tras traducción** |
| 15 | Depression: Reddit Dataset (Cleaned) | https://huggingface.co/datasets/mrjunos/depression-reddit-cleaned | CC BY 4.0 | en | Reddit | 7.731 posts | ✅ HF público | **Incluido tras traducción** (texto en minúsculas y sin puntuación) |
| 16 | SWMH (Ji et al., 2021) | https://huggingface.co/datasets/AIMH/SWMH | CC BY-NC 4.0 | en | Reddit | ~54k posts | ⚠️ gated (aprobación automática) | **Incluido tras login + traducción** |
| 17 | PrevenIA spanish-suicide-intent | https://huggingface.co/datasets/PrevenIA/spanish-suicide-intent | CC BY 4.0 | es | Twitter/Reddit (parte traducida) | 189.064 textos | ✅ HF público | **Descargado, no se mergea** (constructo = intención suicida) |

**Correcciones 2026-09-27:** los IDs de HF de EmoEvalEs (#5) y SWMH-ES (#6)
no existen (HTTP 401) → desactivados; SWMH-ES se reemplaza por SWMH (#16)
+ traducción propia. ReDSM5 (#4) es gated con aprobación manual y no tiene
config pública `redsm5-sample`; el dataset completo tiene 1.484 posts
anotados por psicólogos con síntomas DSM-5 (loader pendiente de ver el
formato real).

## Mapeo de etiquetas (fuentes nuevas)

`label`: 0 = control, 1 = moderado, 2 = depresivo. El subreddit/origen
queda en `label_source` para poder filtrar en análisis de sensibilidad.

| Fuente | → 2 | → 0 | Tipo de etiqueta |
|---|---|---|---|
| kaggle_sdd v13 | r/depression, r/SuicideWatch | r/teenagers | pertenencia a subreddit (débil) |
| reddit_mh_posts | r/depression | r/ADHD, r/OCD, r/aspergers, r/ptsd (negativos "difíciles") | pertenencia a subreddit |
| depression_reddit | `label=1` | `label=0` | pertenencia a subreddit |
| swmh | r/depression | r/SuicideWatch, r/Anxiety, r/bipolar, r/offmychest | pertenencia a subreddit |
| prevenia_es | intención suicida | no | anotación del dataset original |

Limitación: la pertenencia a un subreddit no es un diagnóstico clínico;
es una etiqueta ruidosa (self-reported / community membership).

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
| Emails (se aplica **antes** que menciones) | `[\w._%+-]+@[\w.-]+\.[A-Za-z]{2,}` | vacío |
| Menciones | `@\w+` | vacío |
| Usuarios de Reddit | `/?u/[A-Za-z0-9_-]+` | vacío |
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

## Traducción automática (corpus en inglés)

- Se traduce `text_clean` (post-anonimización) con `src/translation/translate.py`.
- Cada fila traducida registra `mt_system` (`backend:modelo`) y `qc_flags`.
  Hay que registrar además la **fecha** de la corrida: los modelos servidos
  por API pueden cambiar.
- En el corpus final `text_clean` es español y `text_orig` el original.
- Amenazas a la validez a discutir en la tesis:
  1. *Translationese*: evaluar sobre texto nativo en español (Coello /
     MentalRiskES) para que el modelo no aprenda "traducido vs nativo".
  2. Español pro-drop: la traducción omite "yo" → contar 1ra persona
     también por morfología verbal.
  3. Posibles rechazos o notas agregadas por el LLM en contenido suicida
     (`qc_flags=possible_refusal_or_added_note`).
- Validación prevista: evaluación humana de una muestra (~200 posts) y
  métrica sin referencia (p. ej. CometKiwi).

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
| 2026-09-27 | Fuentes Reddit en inglés (#13–17), mapeo de etiquetas, fix anonimización de emails, usuarios de Reddit, sección de traducción | Crenna, Pace |
