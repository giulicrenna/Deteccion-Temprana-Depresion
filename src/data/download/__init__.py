"""Scripts de descarga de fuentes públicas de PLN para detección de depresión.

Cada script expone `download(target_dir: Path) -> dict` con metadatos
para el manifest. Los que requieren gestión manual (gated) levantan
NotImplementedError con instrucciones claras.
"""
