# Presentación de Defensa de Tesina de Grado en LaTeX Beamer

**Título:** *Detección temprana de indicios de depresión en español mediante procesamiento del lenguaje natural: evaluación empírica de modelos contextuales y líneas base clásicas*  
**Autores:** Giuliano Crenna (DNI 43.841.488) · Bruno Emmanuel Pace (DNI 43.127.593)  
**Carrera:** Licenciatura en Ciencia de Datos  
**Institución:** Universidad del Gran Rosario (UGR) — Área de Tecnología y Negocios  
**Docentes / Evaluadores:** Jorge Manuel Fernández Acuña, Briant Alcides Gauna, Alejandro Mario Hernández  
**Año:** 2026

---

## 1. Estructura del Directorio

El directorio `/presentacion_defensa/` es completamente autónomo y modular. No depende de rutas externas al proyecto:

```text
presentacion_defensa/
├── main.tex                    # Archivo principal orquestador de Beamer
├── main.pdf                    # PDF compilado y verificado (379 KB, listo para proyectar)
├── README.md                   # Este manual de compilación y estructura
├── notas_expositor.md          # Guía oral diapositiva a diapositiva con preguntas del tribunal
├── references.bib              # Bibliografía académica verificada en formato BibTeX
├── sections/                   # Contenido modular dividido por capítulos temáticos
│   ├── 01_introduccion.tex     # Diapositivas 1-3: Portada, Contexto/Salud pública y Problema
│   ├── 02_marco_y_objetivos.tex# Diapositivas 4-6: Objetivos, Hipótesis y Metodología CRISP-DM
│   ├── 03_datos_y_preprocesamiento.tex # Diapositivas 7-9: Corpus 1.3M, Partición por usuario y Pipeline
│   ├── 04_modelos_y_arquitectura.tex   # Diapositivas 10-11: Jerarquía de modelos y Arquitectura BETO
│   ├── 05_resultados_principales.tex   # Diapositivas 12-15: Métricas, Curvas ROC y Contraste formal H1/H0
│   ├── 06_transferencia_y_sesgo.tex    # Diapositivas 16-18: Transferencia EN->ES, Sesgo diatópico y Nivel usuario
│   └── 07_conclusiones_y_cierre.tex    # Diapositivas 19-23: Discusión, Limitaciones, Futuro, Conclusiones y Cierre
├── diagramas/                  # Diagramas vectoriales TikZ independientes
│   ├── beto.tex                # Diagrama de flujo del Transformer BETO
│   ├── diseno_experimental.tex # Matriz experimental 2x2 (Mono/Multi vs Test)
│   ├── particion.tex           # Esquema de partición estricta por usuario (70/10/20)
│   ├── pipeline.tex            # Pipeline integral de datos a evaluación
│   └── proceso_investigacion.tex # Ciclo CRISP-DM iterativo en 6 fases
└── figures/                    # Gráficos de alta resolución y logos
    ├── logo_ugr.png            # Isologo institucional de la UGR
    ├── roc_test_nativo.png     # Curvas ROC comparativas en el test nativo
    ├── beto_confusion_matrix.png # Matriz de confusión normalizada de BETO
    ├── baseline_downsampled_top_features.png # Coeficientes léxicos de la Regresión Logística
    └── eda_02_distribucion_clases.png # Histograma de balance de clases (1,31M)
```

---

## 2. Instrucciones de Compilación

La presentación fue desarrollada con soporte multi-motor mediante el paquete `iftex`. Es 100% compatible con **pdfLaTeX**, **XeLaTeX**, **LuaLaTeX** y **Tectonic**.

### Opción A: Compilación local con Tectonic (Recomendada)
Tectonic compila el documento en un solo paso resolviendo automáticamente referencias y paquetes sin necesidad de configurar una distribución TeX Live completa:
```bash
cd presentacion_defensa
tectonic main.tex
```
Generará directamente el archivo `main.pdf`.

### Opción B: Compilación estándar con latexmk
Si se dispone de TeX Live o MacTeX instalado en el sistema:
```bash
cd presentacion_defensa
latexmk -pdf main.tex
```
Para limpiar los archivos auxiliares generados:
```bash
latexmk -c
```

### Opción C: Compilación con pdfLaTeX tradicional
Requiere dos pasadas para actualizar la numeración de diapositivas y las referencias cruzadas:
```bash
cd presentacion_defensa
pdflatex main.tex
pdflatex main.tex
```

### Opción D: Compilación en Overleaf
1. Comprimir el directorio `presentacion_defensa/` en un archivo `.zip`.
2. En Overleaf, seleccionar **New Project** $\to$ **Upload Project** y subir el archivo comprimido.
3. Asegurarse de que `main.tex` esté configurado como el archivo principal (*Main document*).
4. El compilador de Overleaf puede ser tanto **pdfLaTeX** como **XeLaTeX**. Compila de forma nativa e inmediata.

---

## 3. Visualización de las Notas del Expositor en Beamer

Cada diapositiva cuenta con un bloque `\note{...}` integrado en el código LaTeX. Para proyectar en pantalla dividida (diapositiva para el público y notas para el expositor en pantalla secundaria), agregar en el preámbulo de `main.tex`:

```latex
\usepackage{pgfpages}
\setbeameroption{show notes on second screen=right}
```

O para imprimir una versión con notas debajo de cada diapositiva:
```latex
\setbeameroption{show notes}
```

Para una preparación exhaustiva fuera del aula, consultar el documento maestro:
👉 [`notas_expositor.md`](notas_expositor.md)

---

## 4. Filosofía del Diseño Visual y Narrativa

- **Relación de aspecto 16:9 (`aspectratio=169`):** Optimizado para proyectores modernos y pantallas panorámicas de auditorios universitarios.
- **Paleta institucional UGR:** Azul profundo (`#0B3D91`), Celeste institucional (`#3FA9F5`), Verde éxito (`#1E7E34`), Naranja advertencia (`#E07B00`) y Gris pizarra (`#555E68`).
- **Control de densidad:** Sin bloques de texto saturados. Cada diapositiva comunica una sola idea rectora apoyada en tablas concisas, diagramas vectoriales o gráficos generados a partir de los datos experimentales reales.
- **Rigor científico:** No se ocultan las limitaciones metodológicas (transferencia negativa hacia el nativo, atajos diatópicos en el corpus abierto, o equivalencia de modelos a nivel de perfil de usuario).

