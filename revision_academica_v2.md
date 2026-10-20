# Reevaluación académica y metodológica del informe (versión 2)

**Título:** *Detección temprana de depresión mediante procesamiento de lenguaje natural y aprendizaje automático: Evaluación de modelos contextuales y transferencia translingüística en redes sociales*
**Autores:** Tec. Univ. Giuliano Crenna · Tec. Univ. Bruno Emmanuel Pace
**Documento evaluado:** `informe/main.tex` compilado a `temp/build/main.pdf` (56 páginas, 16/10/2026)
**Revisión previa:** `revision_academica.md` (sobre la versión de 48 páginas)
**Evidencia nueva:** `reports/tables/` (generada sin reentrenar con `src.evaluation.run_validation` y `src.evaluation.dialect_control`)

---

## A. Diagnóstico general

La versión revisada resuelve todos los puntos críticos de la revisión anterior. Además incorpora tres análisis que el informe previo solo mencionaba como limitaciones o trabajo futuro:

- **Intervalos de confianza:** se calcularon por *bootstrap* de usuarios, junto con diferencias pareadas entre modelos.
- **Control del atajo dialectal:** se agregó un control experimental con contraste aleatorio, aplicado también a BETO.
- **Evaluación por usuario:** las predicciones se agregaron a nivel de usuario.

Estos análisis no solo dan más rigor al informe: cambian dos conclusiones.

1. **La hipótesis principal se sostiene a nivel de publicación, no de usuario.** Las diferencias pareadas entre BETO y las líneas base tienen intervalos que excluyen el cero (F1-macro +0,062 [0,038; 0,081]). Al agregar por usuario, en cambio, todos los modelos separan casi perfectamente a los 64 usuarios (AUC ≥ 0,97) y BETO deja de distinguirse de las líneas base.
2. **BETO también aprende el atajo dialectal.** Al enmascarar los marcadores regionales, BETO pierde lo mismo que la Regresión Logística (ΔAUC ≈ −0,02). Esto responde la pregunta que la versión anterior dejaba abierta. El atajo, sin embargo, no explica la ventaja de BETO: con el texto enmascarado la ventaja se mantiene igual.

> **Dictamen:** el documento está en condiciones de presentarse. Las observaciones que quedan son menores o corresponden a trabajo futuro ya declarado.

---

## B. Matriz de cumplimiento del correo del evaluador (actualizada)

| Observación del evaluador | Estado anterior | Estado actual | Evidencia en el informe |
| :--- | :--- | :--- | :--- |
| 1. No explicitar la métrica en la hipótesis principal | Parcial (lista cerrada de 4 métricas) | **Resuelta** | Sección 1.3: "batería integral de métricas de clasificación para clases desbalanceadas". |
| 2. Prudencia con las hipótesis secundarias | Resuelta con desvío | **Resuelta** | Antigua Hs2 fusionada con la principal. La antigua Hs4 (ahora Hs3) quedó acotada a "modelos lineales interpretables" y contrastada con pruebas formales (Tabla 12). |
| 3. Definir interpretabilidad y replicabilidad | Resuelta | **Resuelta y precisada** | Sección 4.10: dos niveles (global en lineales; auditoría de atajos en todos). Sección 4.11: enlace al repositorio y comando único de validación. Conclusión 6: alcance explícito. |
| 4. Objetivo 3 desdoblado y comparación tras la evaluación | Pendiente | **Resuelta** | Objetivo 3: solo entrenamiento. Objetivo 4: evaluación con IC. Objetivo 5: comparación y auditoría de sesgos. |
| 5. Profundizar el estado del arte en español | Parcial | **Resuelta** | Sección 2.6: alcance y limitación de Leis et al., MentalRiskES y Coello-Guilarte. |
| 6. Variación lingüística latinoamericana | Resuelta solo en resultados | **Resuelta** | Planteada en 1.2, medida en el análisis exploratorio (3.6, Tabla 5), cuantificada en 5.6 (Tabla 13). |
| 7. Viabilidad general | Resuelta | **Resuelta** | Sin cambios. |

---

## C. Estado de las correcciones de la revisión anterior

### Críticas

| Corrección | Estado |
| :--- | :--- |
| Secuencia y formulación del Objetivo 3 | Hecha |
| Hipótesis principal con lista cerrada de métricas | Hecha |
| Desarticulación narrativa del sesgo dialectal | Hecha: aparece en 1.2, en el análisis exploratorio, en la metodología (4.9) y en los resultados (5.6) |
| Ambigüedad de la unidad de análisis | Hecha: formalizada en 1.2, evaluada en 4.8 y 5.9 |

### Importantes

| Corrección | Estado |
| :--- | :--- |
| Profundizar la sección 2.6 | Hecha. Se evitaron afirmaciones no verificables sobre los métodos de cada antecedente. |
| Acotar Hs4 (ahora Hs3) | Hecha. El resultado pasó de "no sostenida" a "sostenida parcialmente", con pruebas formales (emoción negativa: Fisher p = 0,003). |
| Nota de la SVM en la Tabla 7 (ahora Tabla 8) | Hecha, y además se completaron los valores: F1 dep. 0,544 y AUC 0,754 (sobre la función de decisión). |

### Menores

| Corrección | Estado |
| :--- | :--- |
| "notebooks notebooks" en el Anexo A.1 | Corregido |
| Tilde en "Hernández" | Corregida |
| Dominio de la fórmula TF-IDF (tf > 0) | Aclarado |
| Abreviatura F1-m | Se mantuvo una abreviatura (F1-M) por espacio, definida en el epígrafe de la Tabla 11 |

### Cuestiones metodológicas de la sección E

| Cuestión | Estado |
| :--- | :--- |
| Bootstrap de usuarios | Hecho: 2 000 réplicas, estratificado por etiqueta, con diferencias pareadas |
| Experimento de control dialectal | Hecho sin reentrenar: enmascarado de 175 términos y un patrón, por categoría, con control aleatorio |
| Agregación por usuario | Hecha: score medio y voto mayoritario |
| Explicabilidad de BETO con SHAP | Diferida, con justificación. Reemplazada parcialmente por la prueba de sensibilidad. |

---

## D. Observaciones pendientes

### Importantes

1. ~~**La hipótesis principal y Hs2 son casi redundantes.**~~ **Resuelta:** se fusionaron. La hipótesis principal incorpora la comparación BETO frente a TF-IDF; se quitó el mecanismo ("al capturar negación, orden…"), que el trabajo no pone a prueba. Las secundarias se renumeraron: Hs1 aumento translingüístico, Hs2 rasgos psicolingüísticos, Hs3 consistencia psicolingüística. Los archivos de salida conservan el nombre `hs4_*.csv`, que ahora corresponde a Hs3.
2. **La separación casi perfecta por usuario compromete la validez de constructo del corpus nativo.** Que la Regresión Logística alcance AUC 0,989 por usuario, y que enmascarar los marcadores explícitos apenas la reduzca (0,986), indica que ambos grupos de Coello-Guilarte difieren por muchas señales además de la sintomatología. El informe lo discute en 5.9. Convendría que también figure en "Validez de constructo" (5.11) y en las conclusiones, porque condiciona la lectura de todas las métricas sobre ese corpus.
3. **Variabilidad de entrenamiento no cuantificada.** Los intervalos cubren solo el muestreo de usuarios de prueba. La degradación de BETO multi frente a mono en el test nativo (−0,016 [−0,035; −0,004]) es pequeña y podría cambiar de signo con otra semilla. El informe ya lo declara en "Validez estadística"; no requiere cambios de texto.

### Menores

4. **Reproducibilidad de la LR multi.** El Anexo A.1 ahora remite a `temp/ablation_per_source.py`, que vive en una carpeta de archivos temporales. Hay además dos referencias al módulo inexistente `src.models.train_baseline_multi`: el objetivo `make baseline-multi` del `Makefile` y el test `test_build_baseline_multi_pipeline`, que falla. Para un anexo de reproducción conviene mover ese script a `src/` y corregir esas dos referencias.
5. **Hs2 (rasgos psicolingüísticos) sin intervalo.** La diferencia del modelo combinado (0,644 frente a 0,653) no tiene intervalo porque sus rasgos temporales requieren recomputarse. El texto ya la presenta como "no mejora" y no como empeoramiento.
6. **Comparaciones múltiples.** Se reportan varias diferencias pareadas sin corrección por multiplicidad. Las diferencias centrales (BETO frente a líneas base) tienen intervalos muy alejados del cero y sobrevivirían a cualquier corrección razonable. Una frase en la Sección 4.7 bastaría.
7. **Versión de scikit-learn.** Los modelos `.joblib` se serializaron con la versión 1.5.1 y la validación corrió con la 1.9.0. Las métricas reproducen exactamente las tablas originales, pero `requirements.txt` debería fijar la versión con que se entrenaron.

---

## E. Hipótesis: resultado actualizado

| Hipótesis | Versión anterior | Versión actual | Evidencia clave |
| :--- | :--- | :--- | :--- |
| Principal (incluye la antigua Hs2) | Sostenida con matices | Sostenida a nivel de documento | Δ F1-macro BETO − LR = +0,062 [0,038; 0,081]; todas las diferencias pareadas en el test nativo son positivas; no se distingue a nivel de usuario |
| Hs1 Aumento translingüístico | No sostenida (nativo) | No sostenida (nativo), con IC | Δ BETO multi − mono = −0,016 [−0,035; −0,004] |
| Hs2 Rasgos psicolingüísticos (antes Hs3) | No sostenida | No sostenida | Sin intervalo (observación 5) |
| Hs3 Consistencia psicolingüística (antes Hs4) | No sostenida | Sostenida parcialmente | Emoción negativa enriquecida (p = 0,003); primera persona y absolutismo no significativos |

---

## F. Veredicto

El informe cumple todas las observaciones del evaluador y las correcciones críticas e importantes de la revisión anterior. La incorporación de intervalos por usuario, el control dialectal y la evaluación por usuario lo vuelven sustancialmente más sólido. Además, las conclusiones son ahora más prudentes y están mejor sustentadas: la ventaja de BETO es real a nivel de publicación, el atajo dialectal afecta a todos los modelos y la evaluación por usuario sobre este corpus no permite estimar la capacidad clínica.

La redundancia entre la hipótesis principal y Hs2 (observación 1) ya se resolvió. Antes de la entrega se recomienda reflejar la observación 2 en la sección de validez de constructo. El resto de las observaciones es opcional.
