# Mini-corpus sintético

60 mensajes en español hardcoded a mano (30 depresivos + 30 controles).
**NO usar para entrenar modelos finales** — solo para desarrollo del
pipeline (`make data` end-to-end sin depender de acceso formal a
ningún corpus).

## Estructura

Cada línea del JSONL tiene:

| Campo | Tipo | Descripción |
|---|---|---|
| `user_id` | str | Identificador del usuario (6 usuarios por clase, 5 msgs c/u) |
| `text` | str | Mensaje en español |
| `label` | int | 0 = control, 2 = depresivo |
| `label_source` | str | Siempre `"synthetic"` |
| `timestamp` | str | ISO8601 (todos en 2025-01-01) |

## Marcadores intencionales

- Mensajes depresivos: alta densidad de 1ra persona, vocabulario
  absolutista, negatividad (siguiendo Leis 2019).
- Mensajes control: vocabulario positivo, actividades concretas,
 第三者 (gente alrededor).

## Cómo regenerar

`make data` lo lee automáticamente desde
`src/data/synthetic/sample.jsonl` a través de
`src/data/download/download_synthetic.py`.
