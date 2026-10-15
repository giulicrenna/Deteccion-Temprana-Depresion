import re
from pathlib import Path

INF = Path(__file__).resolve().parents[1] / "informe"
tex = (INF / "main.tex").read_text(encoding="utf-8")
bib = (INF / "references.bib").read_text(encoding="utf-8")
citadas = {k.strip() for g in re.findall(r"\\cite\{([^}]*)\}", tex) for k in g.split(",")}
entradas = set(re.findall(r"^@\w+\{([^,]+),", bib, re.M))
print(len(citadas), "citadas;", len(entradas), "entradas")
print("sin entrada:", sorted(citadas - entradas))
print("no citadas:", sorted(entradas - citadas))
print("citadas:", sorted(citadas))
