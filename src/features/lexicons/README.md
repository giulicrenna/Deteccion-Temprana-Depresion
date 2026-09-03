# Lexicons

CSVs hardcoded con categorías léxicas en español, basados en Leis et al.
(2019) y扩充 (扩充 = extendido) manualmente.

## Archivos

- `leis_lexicon.csv` — listas de palabras por categoría (1ra persona,
  absolutistas, negatividad, positividad). Consumido por
  `src/features/liwc_counts.py`.
- `polarity_es.csv` — lexicon de polaridad fallback (cuando VADER no
  funciona en español). Consumido por `src/features/polarity.py`.

## Formato

Cada CSV tiene dos columnas:

| category / term | term / polarity |
|---|---|
| `first_person_singular` | `yo` |
| `absolutist` | `siempre` |
| `negative_emotion` | `triste` |
| ... | ... |

## Cómo extender

Agregá filas al CSV. Las listas son case-insensitive (los scripts hacen
`.lower()` antes de matchear). Para polaridad, usá `positive` o `negative`
exactamente como valores.

## Limitaciones

- No es LIWC-ES oficial (que requiere licencia paga). Es una aproximación
  razonable para español rioplatense/peninsular basada en Leis 2019.
- Cobertura léxica: ~50 palabras deprimidas + ~30 controles. Aceptable
  para experimentación exploratoria; insuficiente para producción.
