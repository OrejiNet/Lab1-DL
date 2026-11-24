# Referencias

Bibliografía del Laboratorio 1 — *Fundamentos de Aprendizaje Profundo y Aplicaciones*, 2-2026.
Grupo 6 · Cristopher Angulo Ahumada.

Cada entrada indica **dónde se usa** en el trabajo. Todas las referencias fueron verificadas
contra la fuente editorial o el repositorio del preprint; se incluye el DOI cuando existe.

---

## 1. Nota sobre el alcance de estas citas

Las 23 variables construidas en la Fase B **no provienen de una lista tomada de un trabajo
previo**. Son construcciones estándar de la física de colisionadores —separación angular ΔR,
cocientes de masas invariantes, energía transversal total, recuento de jets con etiquetado *b*—
aplicadas a las variables que el dataset ya provee. Las referencias de la sección 3 documentan
**el origen canónico de cada construcción**, de modo que el criterio con que se eligieron las
variables sea verificable y no quede como una decisión arbitraria.

El único trabajo que define el dataset y sus 28 variables originales es Baldi, Sadowski y
Whiteson (2014), referencia [1].

---

## 2. Dataset y problema físico

**[1]** Baldi, P., Sadowski, P. y Whiteson, D. (2014). *Searching for exotic particles in
high-energy physics with deep learning*. **Nature Communications 5**, artículo 4308.
DOI: [10.1038/ncomms5308](https://doi.org/10.1038/ncomms5308) ·
preprint [arXiv:1402.4735](https://arxiv.org/abs/1402.4735)

> Artículo que introduce el dataset HIGGS y define sus 28 variables: 21 cinemáticas de bajo nivel
> y 7 masas invariantes de alto nivel. Es la fuente de la hipótesis central que el laboratorio
> reproduce a escala reducida: que una red profunda puede extraer de las variables crudas del
> detector la información que los físicos obtenían construyendo variables derivadas a mano.
>
> *Se usa en:* justificación del dataset (notebook 01, sección 1), descripción de las variables
> (notebook 01, sección 1.1), fundamento del feature engineering (notebook 02, sección 3.1).

**[2]** Whiteson, D. (2014). *HIGGS* [dataset]. **UCI Machine Learning Repository**, ID 280.
Donado el 11 de febrero de 2014. Licencia CC BY 4.0.
<https://archive.ics.uci.edu/dataset/280/higgs>

> Fuente de la que se descarga el archivo `HIGGS.csv.gz` (2,6 GB, 11.000.000 de filas). La ficha
> documenta que las primeras 21 variables son mediciones de bajo nivel, que las últimas 7 son
> derivadas por físicos, y que el benchmark original reserva las últimas 500.000 observaciones
> como conjunto de prueba —dato que motivó el muestreo uniforme por bloques del notebook 01.
>
> *Se usa en:* descarga (notebook 01, celda 3), estrategia de submuestreo (notebook 01, celda 4).

---

## 3. Fundamento de las variables construidas (Fase B)

**[3]** Cacciari, M., Salam, G. P. y Soyez, G. (2008). *The anti-k_t jet clustering algorithm*.
**Journal of High Energy Physics 2008**(04), 063.
DOI: [10.1088/1126-6708/2008/04/063](https://doi.org/10.1088/1126-6708/2008/04/063) ·
preprint [arXiv:0802.1189](https://arxiv.org/abs/0802.1189)

> Define la distancia angular entre objetos en el plano (rapidez, ángulo azimutal) mediante
> ΔR²_ij = (y_i − y_j)² + (φ_i − φ_j)², que es la métrica estándar de separación en física de
> altas energías y la base de las ocho variables de la familia **separación angular**
> (`dR_*`, `deta_j1j2`, `dphi_j1j2`, `dR_lep_j*`).
>
> *Se usa en:* familia de separación angular (notebook 02, sección 3.1) y tratamiento de la
> periodicidad de φ (notebook 02, sección 3.2).

**[4]** CMS Collaboration (2018). *Identification of heavy-flavour jets with the CMS detector in
pp collisions at 13 TeV*. **Journal of Instrumentation 13**, P05011.
DOI: [10.1088/1748-0221/13/05/P05011](https://doi.org/10.1088/1748-0221/13/05/P05011) ·
preprint [arXiv:1712.07158](https://arxiv.org/abs/1712.07158)

> Documenta el etiquetado de jets de sabor pesado —el *b-tagging*— y su papel en la
> identificación de decaimientos a pares de quarks *b*. Es el fundamento de las dos variables de
> la familia **etiquetado b** (`n_btags`, `suma_btags`), que agregan la información dispersa en
> las cuatro variables `jet*_b_tag` individuales.
>
> *Se usa en:* familia de etiquetado *b* (notebook 02, sección 3.1).

---

## 4. Arquitectura, inicialización y regularización

**[5]** Glorot, X. y Bengio, Y. (2010). *Understanding the difficulty of training deep feedforward
neural networks*. **Proceedings of AISTATS 2010**, PMLR 9, 249–256.
<https://proceedings.mlr.press/v9/glorot10a.html>

> Inicialización Xavier/Glorot, aplicada a la capa de salida lineal de la FFNN y, en la variante
> original del código base, a la meta-capa del Stacking.
>
> *Se usa en:* inicialización de pesos (notebook 02, sección B.2; notebook 03, sección 4.3).

**[6]** He, K., Zhang, X., Ren, S. y Sun, J. (2015). *Delving deep into rectifiers: surpassing
human-level performance on ImageNet classification*. **Proceedings of ICCV 2015**, 1026–1034.
preprint [arXiv:1502.01852](https://arxiv.org/abs/1502.01852)

> Inicialización He/Kaiming con `std = √(2/fan_in)`, que compensa la reducción de varianza que
> introduce ReLU al anular los valores negativos. Se aplica a todas las capas ocultas.
>
> *Se usa en:* inicialización de pesos (notebook 02, sección B.2).

**[7]** Ioffe, S. y Szegedy, C. (2015). *Batch normalization: accelerating deep network training by
reducing internal covariate shift*. **Proceedings of ICML 2015**, PMLR 37, 448–456.
<https://proceedings.mlr.press/v37/ioffe15.html>

> Batch Normalization, presente en cada bloque oculto. Justifica además el uso de `bias=False`
> en las capas lineales que la preceden: BatchNorm resta la media del lote e incorpora su propio
> parámetro de desplazamiento, lo que vuelve redundante el sesgo de la capa anterior.
>
> *Se usa en:* arquitectura FFNN (notebook 02, sección B.2).

**[8]** Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I. y Salakhutdinov, R. (2014).
*Dropout: a simple way to prevent neural networks from overfitting*. **Journal of Machine Learning
Research 15**, 1929–1958. <https://jmlr.org/papers/v15/srivastava14a.html>

> Dropout, el hiperparámetro que la búsqueda de Optuna identificó como el más influyente del
> espacio explorado (88,5 % de la importancia relativa).
>
> *Se usa en:* arquitectura FFNN (notebook 02, sección B.2) y análisis de la búsqueda
> (notebook 02, sección 3.1 del resumen).

**[9]** Loshchilov, I. y Hutter, F. (2019). *Decoupled weight decay regularization*. **ICLR 2019**.
preprint [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)

> AdamW. Desacopla la regularización L2 del paso adaptativo, lo que resulta pertinente dada la
> colinealidad documentada en el notebook 01 (correlaciones de hasta 0,896 entre masas
> invariantes).
>
> *Se usa en:* motor de entrenamiento (notebook 02, sección B.3).

---

## 5. Búsqueda de hiperparámetros

**[10]** Akiba, T., Sano, S., Yanase, T., Ohta, T. y Koyama, M. (2019). *Optuna: a
next-generation hyperparameter optimization framework*. **Proceedings of the 25th ACM SIGKDD
International Conference on Knowledge Discovery & Data Mining**, 2623–2631.
DOI: [10.1145/3292500.3330701](https://doi.org/10.1145/3292500.3330701) ·
preprint [arXiv:1907.10902](https://arxiv.org/abs/1907.10902)

> Framework de la búsqueda de hiperparámetros, incluidos el muestreador TPE y el `MedianPruner`
> que interrumpió 7 de los 30 intentos.
>
> *Se usa en:* búsqueda progresiva (notebook 02, sección B.4).

---

## 6. Ensambles

**[11]** Breiman, L. (1996). *Bagging predictors*. **Machine Learning 24**(2), 123–140.
DOI: [10.1007/BF00058655](https://doi.org/10.1007/BF00058655)

> Bagging. Fundamenta el entrenamiento de los modelos base sobre muestras Bootstrap distintas y
> el argumento de que la reducción de varianza del ensamble depende de que sus miembros cometan
> errores distintos.
>
> *Se usa en:* modelos base (notebook 02, sección B.5); discusión del ensamble (notebook 03).

**[12]** Efron, B. (1979). *Bootstrap methods: another look at the jackknife*. **The Annals of
Statistics 7**(1), 1–26. DOI: [10.1214/aos/1176344552](https://doi.org/10.1214/aos/1176344552)

> Origen del remuestreo Bootstrap, del que se desprende la cobertura esperada del 63,2 % de
> filas únicas por muestra —verificada empíricamente en los tres modelos base— y el 36,8 %
> *Out-Of-Bag* restante.
>
> *Se usa en:* muestreo Bootstrap (notebook 02, sección 8.1).

**[13]** Wolpert, D. H. (1992). *Stacked generalization*. **Neural Networks 5**, 241–259.
Publicado por Elsevier; volumen y páginas verificados, DOI no confirmado.

> Stacking. Introduce la idea de entrenar un segundo modelo —el meta-aprendiz— que aprende cómo
> combinar las predicciones de los modelos de primer nivel, en lugar de promediarlas con pesos
> fijos.
>
> *Se usa en:* meta-red de Stacking (notebook 03, sección 4).

---

## 7. Análisis no supervisado y métricas

**[14]** van der Maaten, L. y Hinton, G. (2008). *Visualizing data using t-SNE*. **Journal of
Machine Learning Research 9**, 2579–2605. <https://jmlr.org/papers/v9/vandermaaten08a.html>

> t-SNE, empleado para comprobar si señal y fondo forman regiones separadas al preservar las
> relaciones de vecindad local. El costo cuadrático que describe el artículo es la razón por la
> que la proyección se calcula sobre 5.000 puntos.
>
> *Se usa en:* análisis no supervisado (notebook 01, celda 9).

**[15]** Rousseeuw, P. J. (1987). *Silhouettes: a graphical aid to the interpretation and
validation of cluster analysis*. **Journal of Computational and Applied Mathematics 20**, 53–65.
DOI: [10.1016/0377-0427(87)90125-7](https://doi.org/10.1016/0377-0427(87)90125-7)

> Índice de silueta, con el que se evaluó la calidad del agrupamiento de K-Means (0,186 con k=2 y
> por debajo de 0,045 para k=3, 4 y 5).
>
> *Se usa en:* análisis no supervisado (notebook 01, celda 9).

**[16]** Hubert, L. y Arabie, P. (1985). *Comparing partitions*. **Journal of Classification 2**(1),
193–218. DOI: [10.1007/BF01908075](https://doi.org/10.1007/BF01908075)

> Índice de Rand ajustado (ARI), corrección por azar del índice de Rand. Es la métrica con la que
> se estableció que el agrupamiento de K-Means no guarda relación con la etiqueta física
> (ARI = 0,0085).
>
> *Se usa en:* análisis no supervisado (notebook 01, celda 9).

---

## 8. Software

**[17]** Paszke, A. *et al.* (2019). *PyTorch: an imperative style, high-performance deep learning
library*. **Advances in Neural Information Processing Systems 32 (NeurIPS 2019)**.
preprint [arXiv:1912.01703](https://arxiv.org/abs/1912.01703)

> Framework de las redes, el entrenamiento y los checkpoints. Versión empleada: `2.15.0+cpu`.

**[18]** Pedregosa, F. *et al.* (2011). *Scikit-learn: machine learning in Python*. **Journal of
Machine Learning Research 12**, 2825–2830. <https://jmlr.org/papers/v12/pedregosa11a.html>

> `StandardScaler`, `train_test_split` estratificado, PCA, t-SNE, K-Means y las métricas
> (ROC-AUC, PR-AUC, F1, matriz de confusión, silueta, ARI).

**[19]** YData (2024). *ydata-profiling* [software]. <https://github.com/ydataai/ydata-profiling>

> Generación del reporte comparativo de particiones mediante `compare([train, val, test])`,
> entregable exigido por el enunciado.

---

## 9. Documento del curso

**[20]** Muñoz, F. (2026). *Laboratorio 1: Pipeline de Deep Learning sobre Datos Masivos*
[enunciado del curso]. Fundamentos de Aprendizaje Profundo y Aplicaciones, Magíster en
Informática, 2-2026. Copia en [`docs/enunciado_laboratorio_1.pdf`](enunciado_laboratorio_1.pdf).

> Define la Opción B, los rangos de volumen y dimensionalidad, la cantidad de modelos base según
> el tamaño del grupo, y los entregables. La autorización para usar HIGGS con 28 columnas, por
> debajo del mínimo de 40, fue otorgada expresamente por el profesor; el *feature engineering* de
> la Fase B eleva el conjunto a 51 variables y satisface el requisito.
