# Revisión académica y metodológica del informe de Trabajo Final de Carrera

**Título evaluado:** *Detección temprana de depresión mediante procesamiento de lenguaje natural y aprendizaje automático: Evaluación de modelos contextuales y transferencia translingüística en redes sociales*  
**Autores:** Tec. Univ. Giuliano Crenna · Tec. Univ. Bruno Emmanuel Pace  
**Carrera:** Licenciatura en Ciencia de Datos — Universidad del Gran Rosario  
**Documento base de revisión:** `main.pdf` (versión de 48 páginas, fecha 14 de octubre de 2026)  
**Evaluador de referencia:** Jorge Manuel Fernández Acuña  

---

## A. Diagnóstico general

El documento visible en `main.pdf` no constituye un mero esquema preliminar, sino un informe de investigación empírica sumamente avanzado y maduro (48 páginas, estructurado como tesis de grado formal con 6 capítulos, referencias bibliográficas y anexos técnicos). Contiene un volumen de trabajo experimental destacable: la consolidación de un corpus de más de 1,3 millones de textos, un pipeline de traducción automática neuronal con MarianMT (`opus-mt-en-es`), particiones estrictas a nivel de usuario (*user-level*), el ajuste fino de un modelo de lenguaje contextual en español (BETO) sobre GPU, y una batería de ablaciones por fuente y dominio.

Desde el punto de vista científico y metodológico, el trabajo tiene un nivel sobresaliente: no oculta los resultados negativos (el aumento translingüístico no mejoró el dominio nativo; los rasgos manuales tipo LIWC no superaron a los n-gramas) y demuestra una enorme lucidez analítica al desenmascarar un fenómeno de **aprendizaje por atajos (*shortcut learning*) de base dialectal** en el corpus nativo.

Sin embargo, el informe arrastra ciertos **desajustes heredados de la transición entre la propuesta inicial y el reporte final**, además de observaciones puntuales del evaluador que quedaron sin resolver:
1. **Objetivo 3 sin corregir:** sigue agrupando entrenamiento y comparación en un solo ítem y ubicándose antes de la evaluación métrica, contraviniendo la observación explícita del evaluador.
2. **Rigidez en la Hipótesis Principal:** para responder al comentario de no depender solo de $F_1$, se incluyó una lista taxativa de cuatro métricas, lo que rigidiza la hipótesis y genera contradicciones con modelos donde no todas las métricas están calculadas (como SVM).
3. **Desconexión narrativa del sesgo dialectal:** el hallazgo empírico más relevante del trabajo (que los clasificadores aprenden sesgos geográficos en lugar de marcadores clínicos) irrumpe como una sorpresa en la Sección 5.5 sin haber sido planteado como riesgo ni en el problema (1.2) ni en el análisis exploratorio (3.6).
4. **Estado del arte en español sintético:** la revisión bibliográfica en español cita las referencias pertinentes, pero las despacha en oraciones aisladas, sin profundizar en sus alcances y limitaciones como solicitó el evaluador.

> [!NOTE]
> **Dictamen global:** El documento está **muy cerca de su versión definitiva**, pero **requiere una ronda de armonización textual y metodológica focalizada** antes de ser presentado formalmente ante el comité evaluador.

---

## B. Matriz de cumplimiento del correo del evaluador

| Observación del evaluador (Jorge) | Evidencia encontrada en el informe (`main.pdf`) | Estado | Acción recomendada |
| :--- | :--- | :--- | :--- |
| **1. Hipótesis principal:** Evitar explicitar la métrica de comparación `"(medido en F1-score sobre la clase positiva)"`, ya que la superioridad podría probarse mediante varias métricas. | **Sección 1.3 (pág. 2):** Se reemplazó la frase por `"...medido mediante F1-macro, F1 de la clase depresiva, AUC-ROC y coeficiente Kappa de Cohen."` | **Parcialmente resuelta (con rigidez metodológica)** | El cambio reemplazó una métrica por una lista cerrada de cuatro, sin seguir la sugerencia de *evitar explicitar* la métrica en el enunciado. En la Tabla 7, la SVM no computa ni $F_1$ depresivo ni AUC, lo que falsearía la hipótesis en sentido estricto. Conviene usar una fórmula abarcativa (`"evaluado mediante una batería integral de métricas de clasificación para clases desbalanceadas"`). |
| **2. Hipótesis secundarias:** Precaución con no comprometer propuestas que excedan el tiempo disponible; libertad para eliminarlas si lo ven así. | **Sección 1.3 y 1.5:** Se mantuvieron y ampliaron a cuatro ($H_{s1}$ a $H_{s4}$). En la **Sección 1.5** se transparentó que la explicabilidad con SHAP/LIME sobre BETO no se implementó por costo computacional y quedó para trabajo futuro. | **Resuelta con desvío documentado** | Mantener las cuatro hipótesis dado que fueron testeadas empíricamente en la Sección 5.6 (Tabla 9). No obstante, se debe acotar la redacción de $H_{s4}$ para especificar que la consistencia se evaluó sobre coeficientes de modelos lineales interpretables. |
| **3. Objetivo general:** Definir en Materiales y Métodos la *interpretabilidad* y la *replicabilidad* para poder concluir si se alcanzaron. | **Secciones 4.7 y 4.8 (págs. 21–22):** Se definieron operacionalmente ambas (4.7: interpretabilidad mediante coeficientes y contraste clínico; 4.8: replicabilidad en 7 puntos verificables). En **Sección 6.1 (conclusiones 4 y 6)** se concluye explícitamente sobre ambas. | **Resuelta** | Mantener ambas secciones. Como precisión técnica, explicitar en 4.7 y 6.1 que la interpretabilidad se alcanzó a nivel global para modelos lineales y auditoría de atajos, quedando pendiente la interpretabilidad post-hoc de BETO. |
| **4. Objetivo específico 3:** Son dos objetivos en uno (entrenar por un lado; comparar por otro); la comparación debe ir tras calcular las métricas. | **Sección 1.4 (pág. 3):** El Objetivo 3 sigue redactado textualmente como `"Entrenar y comparar líneas base clásicas [...] y un modelo BETO ajustado"`, antes del Objetivo 4 (`"Evaluar el rendimiento..."`). | **Pendiente / No resuelta** | Desdoblar el objetivo: dejar el Objetivo 3 exclusivamente para el entrenamiento y diseño de la jerarquía de modelos, y mover la comparación a un objetivo posterior al cálculo de métricas (trasladándolo al Objetivo 5). |
| **5. Estado del arte en español:** Profundizar la revisión en español para fundamentar la escasez de modelos en este campo, citando trabajos existentes, alcances y limitaciones. | **Sección 2.6 (pág. 8):** Se citan Leis et al. [4, 18], MentalRiskES [14, 31], Coello-Guilarte [15], Alemán Acuña et al. [12], Mieles Toloza et al. [13] e Yánez Soffia [33]. En la **Tabla 1 (pág. 10)** se resumen accesos. | **Parcialmente resuelta** | Si bien las referencias existen y son correctas, su tratamiento en 2.6 es meramente enumerativo (una oración por trabajo). Falta describir brevemente qué hicieron (modelos, muestras) y cuáles fueron sus limitaciones (muestras pequeñas, métodos lineales, corpus cerrados, falta de transformers). |
| **6. Variación lingüística latinoamericana:** ¿Cómo abordar los matices dialectales en los corpus? ¿Pueden afectar al modelo? | **Sección 5.5, 5.8 y 5.9 (págs. 29–32):** Se detectó que el modelo léxico aprendió atajos diatópicos (*wey* en controles; *vos*, *joder*, *piñera* en depresivos). Se le dedicó la subsección completa 5.8. | **Resuelta en resultados; pendiente de articulación previa** | Fue abordada de forma sobresaliente en la discusión empírica, pero está ausente en el planteamiento inicial del problema (1.2) y en la exploración (3.6). Conviene mencionarla en la introducción como un factor de riesgo a controlar. |
| **7. Viabilidad general:** Evaluar si los desafíos y alcances son razonables para un Trabajo Final de Carrera. | **Sección 1.5 (pág. 3):** Se documentaron recortes sensatos de alcance (clasificación binaria en vez de 3 clases, validación simple 70/10/20 por costo de GPU en vez de 5-fold CV, descarte de modelos multilingües pesados). | **Resuelta** | El alcance ejecutado es más que suficiente para un TFC de grado universitario. |

---

## C. Correcciones prioritarias

### 1. Críticas (afectan la coherencia lógica y la validez metodológica)
- **Secuencia y formulación del Objetivo 3:** Es contradictorio comparar modelos antes de evaluarlos cuantitativamente con métricas. Debe separarse el entrenamiento de la comparación y resecuenciar los objetivos 3, 4 y 5.
- **Formulación cerrada de la Hipótesis Principal:** Exigir en el enunciado de la hipótesis que se verifique taxativamente en cuatro métricas específicas ($F_1$-macro, $F_1$ clase 2, AUC y $\kappa$) es epistemológicamente riesgoso. En la Tabla 7, la línea base de SVM carece de $F_1$ depresivo y de AUC; si se tomara la hipótesis al pie de la letra, no se podría contrastar formalmente frente a SVM.
- **Desarticulación narrativa del sesgo geográfico/dialectal:** El principal hallazgo del trabajo (aprendizaje por atajos diatópicos) aparece de manera abrupta en la pág. 29. Si no se introduce en la formulación del problema (1.2) como una pregunta de investigación o un riesgo metodológico del español, parece un tropiezo accidental en lugar de una auditoría metodológica rigurosa.
- **Ambigüedad en la unidad de análisis (usuario vs. documento):** En 1.2 se plantea `"identificar perfiles con alta probabilidad..."` (nivel usuario), pero en 1.3 se hipotetiza clasificar `"textos de usuarios"` (nivel publicación). En 5.7 se explica lúcidamente que los tweets individuales heredan la etiqueta del usuario, lo que impone un techo bajo a la precisión por post. Esta distinción debe quedar formalizada desde el capítulo 1.

### 2. Importantes (mejoran la precisión académica)
- **Profundización del estado del arte en español (Sección 2.6):** Explicitar los límites técnicos de los antecedentes en español (p. ej., Leis et al. empleó regresión logística y SVM con lexicones LIWC sobre 90 usuarios; Coello-Guilarte usó alineamientos léxicos sin transformers; MentalRiskES es de acceso restringido y enfocado en Telegram). Esto blinda la justificación de por qué BETO es necesario.
- **Acotación del alcance de interpretabilidad en $H_{s4}$:** La hipótesis secundaria 4 habla de consistencia de los marcadores en general, pero en la práctica solo se contrastaron los coeficientes de la Regresión Logística (la interpretabilidad de BETO se postergó en 1.5). Debe explicitarse que $H_{s4}$ evalúa modelos interpretables lineales.
- **Justificación de valores ausentes en Tabla 7:** La nota al pie de la Tabla 7 indica que la SVM lineal no incluye $F_1$ por clase `"porque no produce probabilidades calibradas"`. Esto es conceptualmente inexacto: el $F_1$ por clase solo requiere predicciones binarias discretas ($\hat{y} \in \{0, 2\}$); lo que requiere probabilidades es únicamente el AUC-ROC. Se debe aclarar que fue un dato no registrado en el script inicial.

### 3. Menores (detalles de redacción, estilo y formato)
- **Cacofonía en Anexo A.1 (pág. 40):** `"desde los notebooks notebooks/04_beto/..."` (palabra repetida).
- **Ortografía en portada (pág. 1):** El docente figura como `"Alejandro Mario Hernandez"` (falta la tilde en Hernández).
- **Consistencia en fórmulas matemáticas (Sección 2.3, pág. 6):** La ecuación (1) de TF-IDF define la variante sublineal $1 + \log(\text{tf})$, pero matemáticamente requiere aclarar que aplica para $\text{tf} > 0$ (o $\text{tf} = 0 \implies 0$).
- **Uniformidad en tablas de métricas:** En Tabla 7 se usa `F1-macro`, mientras que en Tabla 8 se abrevia como `F1-m`. Conviene unificar.

---

## D. Propuestas de redacción concretas

### 1. Hipótesis Principal
* **Ubicación:** Sección 1.3, página 2 de `main.pdf`.
* **Texto actual:**
  > *"Es posible desarrollar un modelo de aprendizaje automático supervisado que, a partir de texto escrito en español y de representaciones extraídas mediante técnicas de PLN, clasifique textos de usuarios con sintomatología depresiva frente a controles con un rendimiento superior al de los modelos de referencia clásicos, medido mediante F1-macro, F1 de la clase depresiva, AUC-ROC y coeficiente Kappa de Cohen."*
* **Problema:** Sobreespecificación de métricas en el enunciado teórico (observación desoída de Jorge); inconsistencia formal con la línea base de SVM donde no se reportan todas ellas.
* **Propuesta corregida:**
  > *"Es posible desarrollar un modelo de aprendizaje automático supervisado que, a partir de texto escrito en español y de representaciones contextuales del lenguaje natural, clasifique publicaciones de usuarios con indicios de sintomatología depresiva frente a controles con un rendimiento superior al de los modelos de referencia clásicos, evaluado mediante una batería integral de métricas de clasificación para clases desbalanceadas."*

### 2. Hipótesis Secundaria 4 ($H_{s4}$)
* **Ubicación:** Sección 1.3, página 2 de `main.pdf`.
* **Texto actual:**
  > *"$H_{s4}$ : Consistencia psicolingüística. Los marcadores que más influyen en las predicciones coinciden con los descritos en la literatura psicológica (primera persona del singular, lenguaje absolutista, emociones negativas)."*
* **Problema:** No delimita a qué arquitectura aplica. Como en BETO no se corrió SHAP/LIME, en la Sección 5.6 se rechaza basándose únicamente en la regresión logística.
* **Propuesta corregida:**
  > *"$H_{s4}$ : Consistencia psicolingüística. Los marcadores léxicos de mayor peso predictivo en modelos lineales interpretables coinciden con los descritos en la literatura psicológica (primera persona del singular, lenguaje absolutista, emociones negativas)."*

### 3. Objetivos Específicos (3, 4 y 5)
* **Ubicación:** Sección 1.4, páginas 2–3 de `main.pdf`.
* **Texto actual:**
  > *"3. Entrenar y comparar líneas base clásicas (TF-IDF con Regresión Logística y SVM), un modelo con rasgos psicolingüísticos y un modelo BETO ajustado.*  
  > *4. Evaluar el rendimiento con métricas adecuadas para clases desbalanceadas, sobre particiones a nivel de usuario, y discutir errores y sesgos.*  
  > *5. Contrastar el escenario mono-corpus con el multi-corpus para estimar el efecto del aumento translingüístico."*
* **Problema:** Se mezclan actividades distintas en el objetivo 3 y se sitúa la comparación antes de la evaluación métrica.
* **Propuesta corregida:**
  > *"3. Entrenar una jerarquía de modelos de complejidad creciente: líneas base léxicas basadas en TF-IDF (Regresión Logística y SVM lineal), un modelo combinado con rasgos psicolingüísticos y temporales explícitos, y un modelo de lenguaje contextual preentrenado (BETO) adaptado mediante ajuste fino.*  
  > *4. Evaluar el rendimiento predictivo mediante una batería multidimensional de métricas sobre particiones ciego estratificadas a nivel de usuario para prevenir fugas de información.*  
  > *5. Comparar el desempeño relativo entre arquitecturas y contrastar el escenario mono-corpus frente al multi-corpus para cuantificar el impacto de la transferencia translingüística y auditar la presencia de sesgos."*

### 4. Planteamiento del problema (incorporación de variación diatópica y unidad textual)
* **Ubicación:** Sección 1.2, página 1 de `main.pdf`.
* **Texto actual:**
  > *"El problema que motiva este trabajo es, por lo tanto, simultáneamente computacional y metodológico: ¿es viable construir un modelo que, a partir de texto escrito en español, identifique perfiles con alta probabilidad de presentar sintomatología depresiva con un desempeño razonable, respetando la privacidad de los usuarios, siendo interpretable para profesionales de la salud mental y evaluado de forma que sus resultados sean confiables? Una pregunta derivada es si la escasez de datos en español puede mitigarse traduciendo automáticamente corpus en inglés."*
* **Problema:** Desconexión con la unidad de clasificación real (publicación vs. perfil) y omisión del factor dialectal latinoamericano.
* **Propuesta corregida:**
  > *"El problema que motiva este trabajo es, por lo tanto, simultáneamente computacional y metodológico: ¿es viable construir un modelo que, a partir de texto en español a nivel de documento, clasifique publicaciones individuales para asistir en la identificación temprana de usuarios con sintomatología depresiva, manteniendo un desempeño robusto ante la variación dialectal del idioma, respetando la privacidad y garantizando criterios de interpretabilidad y reproducibilidad? Dos preguntas derivadas son si la escasez de recursos anotados en español puede mitigarse mediante aumento translingüístico con traducción automática neuronal y en qué medida las diferencias regionales y de registro entre corpus introducen atajos espurios de clasificación."*

### 5. Brechas del Estado del Arte en español
* **Ubicación:** Sección 2.6 (párrafo final), página 8 de `main.pdf`.
* **Texto actual:**
  > *"Brechas identificadas. La producción en español sigue siendo escasa; pocos trabajos utilizan modelos específicos de la lengua y la reproducibilidad se ve limitada por la falta de código y datos abiertos [12, 13, 33]. Además, el uso de corpus traducidos para aumentar datos en salud mental en español ha sido poco evaluado."*
* **Problema:** Fundamentación bibliográfica muy escueta que no explicita los alcances y límites de los trabajos previos.
* **Propuesta corregida:**
  > *"Brechas identificadas. Si bien la literatura hispanohablante cuenta con contribuciones valiosas, su alcance presenta limitaciones metodológicas estructurales: los trabajos fundacionales de Leis et al. [4, 18] se concentran en clasificadores clásicos (SVM y regresión lineal) sobre muestras reducidas de Twitter y sin modelos contextuales; el corpus MentalRiskES [14, 31], aunque incorpora desafíos con transformers, posee un acceso restringido y se enfoca en mensajería de Telegram; y el trabajo de Coello-Guilarte et al. [15] explora alineamientos léxicos bilingües pero sobre un corpus proveniente de una sola plataforma y sin balance geolectal. Son escasos los estudios que evalúen modelos específicos como BETO en detección de depresión en redes y la reproducibilidad se ve comprometida por la falta de código abierto [12, 13, 33]. Finalmente, la viabilidad de utilizar traducción automática neuronal para mitigar la escasez de datos en español no había sido contrastada de forma sistemática con particiones estrictas por usuario."*

### 6. Nota explicativa de la SVM en Tabla 7
* **Ubicación:** Sección 5.1 (pie de Tabla 7), página 23 de `main.pdf`.
* **Texto actual:**
  > *"La SVM lineal no produce probabilidades calibradas y su registro no incluye el F1 por clase ni el AUC."*
* **Problema:** Atribución inexacta de la ausencia de $F_1$ a la falta de probabilidades calibradas.
* **Propuesta corregida:**
  > *"La SVM lineal se optimizó mediante pérdida de bisagra (hinge loss) sin estimación probabilística, por lo que no permite calcular el AUC-ROC. El F1 de la clase depresiva no fue registrado en el pipeline original de líneas base, si bien su F1-macro (0,653) y coeficiente Kappa (0,318) confirman un rendimiento equivalente al de la Regresión Logística."*

### 7. Detalle de estilo en Anexos
* **Ubicación:** Anexo A.1, primer párrafo tras el bloque de código, página 40 de `main.pdf`.
* **Texto actual:**
  > *"Los modelos mono de BETO y el modelo combinado se entrenan desde los notebooks notebooks/04_beto/02_finetune_beto_local.ipynb y notebooks/03_baseline/03_combined_features.ipynb..."*
* **Propuesta corregida:**
  > *"Los modelos mono de BETO y el modelo combinado se entrenan a partir de los cuadernos ubicados en `notebooks/04_beto/02_finetune_beto_local.ipynb` y `notebooks/03_baseline/03_combined_features.ipynb`..."*

---

## E. Cuestiones metodológicas pendientes

Es fundamental separar las decisiones que deben plasmarse en el texto de esta versión de aquellas que corresponden a extensiones de investigación:

### Necesarias para esta entrega
1. **Articular el problema del sesgo geolectal desde el inicio:** No requiere correr experimentos nuevos; solo exige conectar coherentemente el problema (1.2) con el hallazgo ya documentado en 5.5 y 5.8.
2. **Armonizar el alcance de la interpretabilidad:** Reconocer formalmente en el marco metodológico (4.7) que la interpretabilidad post-hoc mediante SHAP/LIME sobre la arquitectura de BETO quedó acotada a trabajo futuro por restricciones computacionales, habiéndose logrado la interpretabilidad analítica y de auditoría a través de los modelos lineales.
3. **Transparentar el salto de granularidad (texto individual vs. perfil de usuario):** Dejar explícito en la metodología que, aunque el objetivo de aplicación clínica es la detección a nivel de usuario, la formulación empírica clasifica textos individuales con etiquetas heredadas.

### Diferibles al desarrollo posterior de la tesina (trabajo futuro)
1. **Explicabilidad profunda de BETO con XAI:** Implementar SHAP o mapas de autoatención sobre el codificador de BETO para verificar formalmente si el modelo contextual aprendió o no los mismos atajos dialectales que la regresión logística.
2. **Agregación temporal por usuario:** Diseñar una capa de agregación posterior (por ejemplo, clasificar ventanas temporales de publicaciones o calcular la tasa de positividad acumulada por usuario a lo largo de semanas), emulando los protocolos de CLEF eRisk.
3. **Experimento de control dialectal:** Enmascarar entidades y modismos regionales (topónimos como *Puebla*, *Veracruz*, y términos dialectales como *vos*, *wey*, *joder*) y reentrenar los modelos para cuantificar qué porcentaje de la ventaja métrica provenía del atajo geográfico.
4. **Validación con significación estadística:** Calcular intervalos de confianza mediante remuestreo (*bootstrap*) sobre los 64 usuarios del conjunto de prueba nativo.

---

## F. Veredicto final

El trabajo desarrollado por Bruno y Giuliano es de una **calidad técnica, metodológica y ética sobresaliente**, muy por encima del estándar habitual de un plan de trabajo y plenamente representativo de una tesina de grado rigurosa en Ciencia de Datos. Los resultados negativos y el descubrimiento del sesgo diatópico demuestran honestidad científica y pensamiento crítico.

Para que el informe quede listo para su presentación definitiva ante Jorge y el comité evaluador, la lista de cambios concretos a realizar se resume en seis puntos:

1. **Reestructurar los Objetivos 3, 4 y 5** en la Sección 1.4, separando el entrenamiento de la comparación y colocándola tras la evaluación métrica.
2. **Flexibilizar la Hipótesis Principal** en la Sección 1.3, eliminando la enumeración taxativa de cuatro métricas para adoptar una fórmula general sobre clases desbalanceadas.
3. **Acotar la Hipótesis Secundaria 4 ($H_{s4}$)** indicando que la consistencia psicolingüística se evalúa sobre modelos interpretables lineales.
4. **Integrar la variación dialectal en el Planteamiento del Problema (1.2)**, de modo que el hallazgo empírico de las Secciones 5.5 y 5.8 responda a una pregunta planteada desde el comienzo.
5. **Enriquecer el párrafo de brechas en el Estado del Arte (2.6)**, explicitando brevemente los alcances y limitaciones técnicas de los antecedentes en español citados (Leis et al., MentalRiskES y Coello-Guilarte).
6. **Subsanar erratas menores:** la duplicación de palabras en el Anexo A.1, la tilde en la portada ("Hernández") y la aclaración conceptual sobre la SVM en la nota de la Tabla 7.
