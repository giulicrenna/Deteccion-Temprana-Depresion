# configs/

YAML files de configuración desacoplados del código. Cualquier ruta,
umbral o parámetro del pipeline se lee desde acá.

## Archivos

- `data.yaml` — rutas, versiones de datasets, splits, mapa de etiquetas,
  reglas de anonimización. Lo consumen `make_dataset.py`, `merge_corpora.py`,
  `build_splits.py`.
- `preprocessing.yaml` — parámetros de limpieza, tokenización, features
  (LIWC, temporales, polaridad).

## Cómo agregar un config nuevo

1. Creá `configs/<etapa>.yaml` siguiendo el mismo estilo (snake_case,
   comentarios al tope).
2. Cargalo en el script con:
   ```python
   from pathlib import Path
   import yaml
   cfg = yaml.safe_load(Path("configs/data.yaml").read_text())
   ```
3. Documentá acá abajo qué script lo consume.

## Consumidores

| Script | Configs |
|---|---|
| `src/data/make_dataset.py` | `data.yaml` (sección `anonymize`, `data.sources`) |
| `src/data/merge_corpora.py` | `data.yaml` (sección `data.sources`, `label_map`) |
| `src/data/build_splits.py` | `data.yaml` (sección `splits`) |
| `src/features/*` | `preprocessing.yaml` |
| `src/models/*` (etapa 4) | `configs/model/*.yaml` (próximamente) |
