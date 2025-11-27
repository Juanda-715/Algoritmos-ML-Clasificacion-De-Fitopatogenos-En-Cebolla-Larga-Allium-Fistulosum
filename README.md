**Clasificación de fitopatógenos en cebolla larga (Allium fistulosum)**

==================================================================

**NOTA: En este repositorio no se encuentra subido el dataset
Ni el archivo de caracteristícas .NPZ**


==================================================================

**Resumen**:


Este proyecto implementa, entrena y compara distintos algoritmos de machine learning

para la detección y clasificación de enfermedades en cebolla larga a partir de un

dataset de imágenes.

==================================================================

Link de Drive del Proyecto Completo: https://drive.google.com/drive/folders/1FBIKcmoI7_0FVo_eOfoUhAOrifghxm0d?usp=drive_link

Link del Video Explicatorio: https://www.youtube.com/watch?v=LAt3KUxgebY

==================================================================

Se utilizan modelos clásicos:

\- Regresión Logística

\- K-Means

\- KNN

\- Árbol de Decisión

\- Random Forest

\- SVM

\- MLPClassifier



Además, se cuenta con un pipeline de preprocesamiento y extracción de características

en Python, pensado para ser reutilizable en trabajos futuros.



------------------------------------------------------------------

**1. ESTRUCTURA DEL PROYECTO**

------------------------------------------------------------------



├── Arbol\_Decision/          # Resultados y modelos del Árbol de Decisión (.pkl, gráficas, etc.)

├── K-Means/                 # Resultados y modelos de K-Means

├── KNN/                     # Resultados y modelos de KNN

├── MLPClassifier/           # Resultados y modelos del MLPClassifier

├── Random\_Forest/           # Resultados y modelos de Random Forest

├── Regresion\_Logistica/     # Resultados y modelos de Regresión Logística

├── SVM/                     # Resultados y modelos de SVM

├── Onion\_Desease\_Dataset/   # Dataset original de imágenes por clase

├── Imagenes\_Prueba/         # 4 imágenes independientes para pruebas de predicción

│

├── 0.0.Parametrizacion.ipynb      # Notebook de parametrización general del proyecto

├── 0.1.Extractor\_Caracteristicas.ipynb

│                                  # Notebook de extracción de características (carga de imágenes,

│                                  # redimensionado, flatten, normalización, guardado de features, etc.)

├── 1.Regresion\_Logistica.ipynb    # Entrenamiento y análisis de Regresión Logística

├── 2.K-Means.ipynb                # Entrenamiento y análisis de K-Means

├── 3.KNN.ipynb                    # Entrenamiento y análisis de KNN

├── 4.Arbol\_Decision.ipynb         # Entrenamiento y análisis de Árbol de Decisión

├── 5.Random\_Forest.ipynb          # Entrenamiento y análisis de Random Forest

├── 6.SVM.ipynb                    # Entrenamiento y análisis de SVM

├── 7.MLPClassifier.ipynb          # Entrenamiento y análisis de MLPClassifier

│

├── onion\_d eseat.csv              # Dataset principal (rutas de imágenes + etiquetas de clase)

├── onion\_dataset\_full.csv         # Versión extendida / completa del dataset (si aplica)

├── train\_dataset.csv              # Partición de entrenamiento

├── val\_dataset.csv                # Partición de validación

├── test\_dataset.csv               # Partición de prueba

├── onion\_features.npz             # Matriz de características preprocesadas (features extraídas)

│

└── README.txt / README.md         # Este archivo





Carpetas principales:



\- Arbol\_Decision/

&nbsp;   Resultados y modelos del Árbol de Decisión (.pkl, gráficas, etc.).



\- K-Means/

&nbsp;   Resultados y modelos de K-Means.



\- KNN/

&nbsp;   Resultados y modelos de KNN.



\- MLPClassifier/

&nbsp;   Resultados y modelos del MLPClassifier.



\- Random\_Forest/

&nbsp;   Resultados y modelos de Random Forest.



\- Regresion\_Logistica/

&nbsp;   Resultados y modelos de Regresión Logística.



\- SVM/

&nbsp;   Resultados y modelos de SVM.



\- Onion\_Desease\_Dataset/

&nbsp;   Dataset original de imágenes organizadas por clase de fitopatógeno.



\- Imagenes\_Prueba/

&nbsp;   Cuatro imágenes de prueba independientes para evaluar los modelos entrenados.





Notebooks principales (Jupyter):



\- 0.0.Parametrizacion.ipynb

&nbsp;   Notebook de parametrización general del proyecto

&nbsp;   (rutas, tamaños de imagen, particiones del dataset, etc.).



\- 0.1.Extractor\_Caracteristicas.ipynb

&nbsp;   Extracción de características a partir de las imágenes:

&nbsp;   carga, redimensionado, normalización, flatten, guardado de features.



\- 1.Regresion\_Logistica.ipynb

&nbsp;   Entrenamiento, evaluación y análisis de Regresión Logística.



\- 2.K-Means.ipynb

&nbsp;   Entrenamiento, evaluación y análisis de K-Means.



\- 3.KNN.ipynb

&nbsp;   Entrenamiento, evaluación y análisis de KNN.



\- 4.Arbol\_Decision.ipynb

&nbsp;   Entrenamiento, evaluación y análisis del Árbol de Decisión.



\- 5.Random\_Forest.ipynb

&nbsp;   Entrenamiento, evaluación y análisis de Random Forest.



\- 6.SVM.ipynb

&nbsp;   Entrenamiento, evaluación y análisis de SVM.



\- 7.MLPClassifier.ipynb

&nbsp;   Entrenamiento, evaluación y análisis de MLPClassifier.





Archivos de datos:



\- onion\_dataset.csv

&nbsp;   Dataset principal: rutas de imágenes y etiquetas de clase.



\- onion\_dataset\_full.csv

&nbsp;   Versión extendida/completa del dataset (si aplica).



\- train\_dataset.csv

&nbsp;   Conjunto de entrenamiento.



\- val\_dataset.csv

&nbsp;   Conjunto de validación.



\- test\_dataset.csv

&nbsp;   Conjunto de prueba.



\- onion\_features.npz

&nbsp;   Matriz de características preprocesadas generada por el extractor

&nbsp;   (por ejemplo, píxeles normalizados ya aplanados).





------------------------------------------------------------------

**2. CONTENIDO DE LAS CARPETAS DE MODELOS**

------------------------------------------------------------------



Cada carpeta de modelo (Arbol\_Decision, KNN, Random\_Forest, SVM, MLPClassifier,

Regresion\_Logistica, K-Means) incluye:



\- El modelo entrenado en formato .pkl (serializado con joblib).

\- Imágenes generadas durante el análisis:

&nbsp; matrices de confusión, curvas de aprendizaje, histogramas, etc.

\- Otros archivos auxiliares usados por los notebooks correspondientes.





------------------------------------------------------------------

**3. REQUISITOS**

------------------------------------------------------------------



Se necesita Python >3.8 y las siguientes librerías (mínimo):



\- numpy

\- pandas

\- scikit-learn

\- matplotlib

\- seaborn

\- pillow (PIL)

\- joblib



Ejemplo de instalación rápida (ajustar según el entorno):



&nbsp;   pip install numpy pandas scikit-learn matplotlib seaborn pillow joblib tensorflow





------------------------------------------------------------------

**4. IMPORTANTE SOBRE LAS RUTAS**

------------------------------------------------------------------



Todos los scripts y notebooks usan rutas RELATIVAS respecto a la carpeta raíz

de este proyecto.



Para que el código funcione correctamente:



1\. Abrir la carpeta raíz del proyecto en el IDE (VSCode, PyCharm, JupyterLab, etc.).

2\. Ejecutar los notebooks desde esa ubicación.

3\. No cambiar la estructura de carpetas ni mover archivos individualmente.



Si se mueve el proyecto a otro lugar del disco, debe copiarse o trasladarse

la carpeta completa manteniendo la misma estructura interna.





------------------------------------------------------------------

**5. FLUJO RECOMENDADO DE EJECUCIÓN**

------------------------------------------------------------------



Paso 1: Parametrización

-----------------------

Abrir y ejecutar:



\- 0.0.Parametrizacion.ipynb



En este notebook se definen parámetros generales:

rutas de carpetas, tamaño de imagen, particiones del dataset, etc.



Paso 2: Extracción de características

-------------------------------------

Ejecutar:



\- 0.1.Extractor\_Caracteristicas.ipynb



Este notebook:

\- Carga las imágenes desde Onion\_Desease\_Dataset.

\- Realiza preprocesado (redimensionado, normalización, flatten).

\- Genera y guarda la matriz de características en onion\_features.npz.

\- Genera y guarda los archivos train\_dataset.csv, val\_dataset.csv, test\_dataset.csv.



Paso 3: Entrenamiento de modelos

--------------------------------

Ejecutar los notebooks:



\- 1.Regresion\_Logistica.ipynb

\- 2.K-Means.ipynb

\- 3.KNN.ipynb

\- 4.Arbol\_Decision.ipynb

\- 5.Random\_Forest.ipynb

\- 6.SVM.ipynb

\- 7.MLPClassifier.ipynb



Cada notebook:



\- Carga las características y etiquetas.

\- Entrena el modelo correspondiente.

\- Ajusta o fija hiperparámetros según el caso.

\- Evalúa el modelo usando accuracy, precision, recall, F1-score y matrices de confusión.

\- Guarda el modelo entrenado (.pkl) en su carpeta.

\- Genera y guarda las gráficas de análisis (curvas de entrenamiento, importancia de características, etc.).



Paso 4: Pruebas con imágenes nuevas

-----------------------------------

En cada notebook se incluye una función del estilo:



\- predecir\_imagen(...)



Esta función permite cargar imágenes desde:

\- Imagenes\_Prueba/

\- u otra ruta válida,



y obtener la clase predicha por el modelo entrenado.





------------------------------------------------------------------

**6. PROPÓSITO DEL PROYECTO**

------------------------------------------------------------------



Este proyecto tiene dos objetivos principales:



1\. Estudiar y comparar el desempeño de diferentes algoritmos de machine learning

&nbsp;  para la detección y clasificación de fitopatógenos en cebolla larga

&nbsp;  (Allium fistulosum) utilizando un dataset de imágenes.



2\. Dejar una base de código en Python:

&nbsp;  - Estructurada en notebooks.

&nbsp;  - Documentada y organizada por modelos.

&nbsp;  - Reutilizable y fácil de extender para futuras investigaciones,

&nbsp;    nuevos datasets o modificaciones en la arquitectura de los modelos.



------------------------------------------------------------------

Fin del README.



