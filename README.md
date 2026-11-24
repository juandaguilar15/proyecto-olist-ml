# Olist ML — Análisis de comercio electrónico con Ciencia de Datos

**Proyecto académico | Universidad de Cundinamarca — sede Fusagasugá**  
**Asignatura:** Ciencia de Datos  
**Integrantes:** Juan David Aguilar Páez y Nicolás Mauricio Vásquez Guaqueta

Este proyecto analiza el conjunto de datos público **Brazilian E-Commerce Public Dataset by Olist** para abordar dos preguntas relacionadas con el comercio electrónico:

1. **¿Qué pedidos presentan mayor riesgo de retraso en la entrega?** Se experimenta con modelos supervisados de clasificación y evaluaciones temporales.
2. **¿Qué perfiles de compradores se pueden identificar a partir de su comportamiento de compra?** Se construyen indicadores RFM y RM y se aplica agrupamiento mediante K-Means.

El proyecto tiene un **alcance académico y exploratorio**. Sus resultados ayudan a comprender patrones históricos, pero no constituyen un sistema operativo de predicción ni una solución validada para producción.

## Contenido

- [Objetivos](#objetivos)
- [Datos y metodología](#datos-y-metodología)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Instalación y ejecución](#instalación-y-ejecución)
- [Resultados principales](#resultados-principales)
- [Funciones reutilizables y pruebas](#funciones-reutilizables-y-pruebas)
- [Diccionario de datos](#diccionario-de-datos)
- [Limitaciones](#limitaciones)
- [Fuente de los datos](#fuente-de-los-datos)

## Objetivos

**Objetivo general:** analizar datos históricos de comercio electrónico utilizando técnicas de Ciencia de Datos y aprendizaje automático para estudiar los retrasos en entregas y el comportamiento de compra.

**Objetivos específicos:**

- Comprender la estructura, calidad y relaciones de las tablas originales de Olist.
- Explorar patrones temporales, geográficos y logísticos de los pedidos.
- Integrar y transformar los datos para construir características predictoras.
- Comparar modelos supervisados para identificar pedidos con mayor riesgo de retraso.
- Segmentar compradores mediante indicadores de recencia, frecuencia y gasto.
- Interpretar los resultados y reconocer sus limitaciones.

## Datos y metodología

El análisis utiliza las **nueve tablas originales de Olist**, relacionadas mediante identificadores de pedidos, compradores, vendedores, productos y otros registros. Los cuadernillos siguen este orden:

| Cuadernillo | Propósito |
|---|---|
| [`01_comprension_datos.ipynb`](notebooks/01_comprension_datos.ipynb) | Reconocer los archivos, columnas, tipos de datos, relaciones y características iniciales. |
| [`02_analisis_exploratorio.ipynb`](notebooks/02_analisis_exploratorio.ipynb) | Examinar la distribución de los pedidos, los retrasos y las variables logísticas y geográficas. |
| [`03_preparacion_datos.ipynb`](notebooks/03_preparacion_datos.ipynb) | Integrar tablas, definir el retraso, preparar conjuntos de entrenamiento y prueba y construir las versiones V1, V2 y V3. |
| [`04_modelos_supervisados.ipynb`](notebooks/04_modelos_supervisados.ipynb) | Entrenar y comparar clasificadores, validar por períodos y estudiar métricas, umbrales e interpretabilidad. |
| [`05_segmentacion_clientes.ipynb`](notebooks/05_segmentacion_clientes.ipynb) | Construir indicadores RFM/RM, comparar agrupamientos K-Means e interpretar los segmentos. |
| [`06_evaluacion_integral.ipynb`](notebooks/06_evaluacion_integral.ipynb) | Reunir los resultados exportados, discutir los hallazgos, sus aplicaciones y las limitaciones. |

### Predicción de retrasos

Para los pedidos entregados, la variable objetivo `retraso` toma el valor **1** cuando la **fecha calendario** de entrega real es posterior a la fecha estimada, y **0** cuando se entrega en la fecha prevista o antes. Se construyen características por pedido y se comparan tres versiones experimentales:

- **V1:** conjunto inicial de variables.
- **V2:** incorpora `dias_prometidos`, que representa el plazo estimado para la entrega.
- **V3:** incorpora información de distancia logística, incluida `distancia_maxima_km`, calculada a partir de coordenadas geográficas mediante Haversine.

Se evalúan regresión logística, árbol de decisión, Random Forest, XGBoost y CatBoost, con especial atención a **Average Precision (AP)** y a la variación de los resultados entre ventanas temporales.

### Segmentación de compradores

Se calculan indicadores de **recencia (R)**, **frecuencia (F)** y **valor monetario (M)**. Se comparan agrupamientos basados en RM y RFM, utilizando transformaciones y estandarización de las variables antes de aplicar K-Means. La selección de segmentos se interpreta según su comportamiento de compra; **los números de segmento son etiquetas del algoritmo, no niveles de calidad de los clientes**.

## Estructura del repositorio

```text
proyecto-olist-ml/
├── data/
│   ├── raw/                         # Nueve CSV originales de Olist
│   └── processed/                   # train.csv, test.csv, train_v2.csv y train_v3.csv
├── docs/
│   └── diccionario_datos_olist_ml.pdf
├── notebooks/
│   ├── 01_comprension_datos.ipynb
│   ├── 02_analisis_exploratorio.ipynb
│   ├── 03_preparacion_datos.ipynb
│   ├── 04_modelos_supervisados.ipynb
│   ├── 05_segmentacion_clientes.ipynb
│   └── 06_evaluacion_integral.ipynb
├── resultados/                     # CSV generados por los análisis
├── src/
│   ├── __init__.py
│   ├── preparacion.py
│   ├── validacion.py
│   ├── evaluacion.py
│   ├── preprocesamiento.py
│   └── segmentacion.py
├── tests/
│   └── test_modulos.py
├── requirements.txt
├── .gitignore
└── README.md
```

> La estructura destaca los archivos necesarios para reproducir el análisis. Las carpetas adicionales de entregables, si existen, no son necesarias para entender el flujo principal.

Los **CSV originales y los cuatro conjuntos procesados se incluyeron en el repositorio** para facilitar su revisión. Los datos de `data/processed/` pueden volver a generarse mediante el cuadernillo 03. La carpeta `resultados/` almacena las salidas usadas por la evaluación integral.

**No se incluye una carpeta `modelos/` ni archivos `.joblib` como entregables.** El entrenamiento y la evaluación se realizan directamente en los cuadernillos. Esta decisión describe el alcance de esta entrega académica, no una recomendación general contra la serialización de modelos.

## Instalación y ejecución

Las instrucciones siguientes están orientadas a **Windows y PowerShell**, aunque el proyecto puede adaptarse a otros sistemas. Se recomienda Python **3.13**, VS Code con la extensión Jupyter y Git.

### 1. Clonar el repositorio

```powershell
git clone https://github.com/juandaguilar15/proyecto-olist-ml.git
cd proyecto-olist-ml
```

### 2. Crear y activar un entorno virtual

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Instalar dependencias

```powershell
python -m pip install -r requirements.txt
python -m pip check
```

Entre las bibliotecas utilizadas se encuentran **Pandas, NumPy, Matplotlib, scikit-learn, XGBoost y CatBoost**. Las versiones concretas dependen del archivo `requirements.txt` incluido en la entrega.

### 4. Abrir los cuadernillos

```powershell
code .
```

En VS Code, abre cada archivo desde la carpeta `notebooks/` y selecciona el intérprete del entorno `.venv` como kernel de Jupyter.

**Importante:** las rutas de los notebooks son relativas a la carpeta `notebooks/` (por ejemplo, `../data/raw/`). Ejecútalos con **`notebooks/` como directorio de trabajo**. Los cuadernillos que usan funciones de `src` añaden la carpeta raíz del proyecto a la ruta de importaciones.

### 5. Ejecutar en orden

1. **01:** comprensión de los datos.
2. **02:** análisis exploratorio.
3. **03:** integración e ingeniería de características; genera o actualiza `data/processed/`.
4. **04:** entrenamiento y evaluación supervisada; genera `resultados/validacion_temporal_v3.csv` y `resultados/comparacion_algoritmos_v3.csv`.
5. **05:** segmentación; genera `resultados/segmentacion_clientes.csv`, `resultados/resumen_segmentos_rm.csv` y `resultados/resumen_segmentos_rfm.csv`.
6. **06:** evaluación integral, que lee los CSV producidos por los cuadernillos 04 y 05.

Los archivos de datos originales pueden consultarse en `data/raw/`. Si los datos no estuvieran disponibles en una copia del proyecto, se pueden obtener desde la fuente pública de Olist indicada al final.

## Resultados principales

### 1. Modelos supervisados: Average Precision temporal

En la evaluación temporal exploratoria de **V3** se obtuvieron los siguientes valores:

| Período | Regresión logística V3 | Random Forest V3 |
|---|---:|---:|
| Noviembre de 2017 | 0,1659 | 0,1135 |
| Febrero de 2018 | 0,2795 | 0,2593 |
| Abril de 2018 | 0,1379 | 0,1295 |

La regresión logística obtuvo mayor AP que Random Forest en los tres períodos revisados. Estas cifras **no deben interpretarse como una evaluación final independiente de todos los experimentos**, ni como prueba de buen desempeño en datos futuros o producción. El desempeño cambia entre meses y la calibración de las probabilidades sigue siendo una limitación.

### 2. Aprendizaje no supervisado: perfiles RM (K = 3)

Se analizaron **93.357 compradores**. La segmentación RM con tres grupos permitió identificar estos perfiles:

| Segmento | Compradores | Porcentaje | Recencia mediana (días) | Gasto mediano (R$) | Interpretación |
|---|---:|---:|---:|---:|---|
| 0 | 16.380 | 17,55 % | 37 | 106,32 | Compradores relativamente recientes. |
| 1 | 43.791 | 46,91 % | 270 | 67,13 | Compradores menos recientes y de menor gasto. |
| 2 | 33.186 | 35,55 % | 251 | 206,87 | Compradores menos recientes y de mayor gasto. |

El análisis identificó que aproximadamente **97 %** de los compradores realizó una sola compra registrada, por lo que la frecuencia aporta poca diferenciación a gran parte de la población. El modelo RFM completo se estudia como comparación complementaria.

> Los segmentos describen patrones observados en este conjunto de datos histórico. No equivalen a categorías comerciales validadas ni implican resultados causales.

## Funciones reutilizables y pruebas

Los cuadernillos contienen la exploración, los experimentos y las interpretaciones. La carpeta `src/` reúne funciones compartidas que ayudan a evitar repetición de código y a separar responsabilidades:

| Archivo | Responsabilidad principal | Cuadernillo que lo utiliza |
|---|---|---|
| `src/preparacion.py` | Preparar la variable objetivo, agregaciones de artículos e información de envíos. | 03 |
| `src/validacion.py` | Dividir períodos y evaluar ventanas temporales. | 04 |
| `src/evaluacion.py` | Calcular métricas para modelos de clasificación. | 04 |
| `src/preprocesamiento.py` | Crear transformaciones de variables para los modelos. | 04 |
| `src/segmentacion.py` | Transformar indicadores RFM y RM. | 05 |

Las llamadas a las funciones se encuentran dentro de los notebooks; **no es necesario ejecutar los módulos de `src` como programas independientes**.

Para ejecutar las pruebas automatizadas desde la raíz del repositorio:

```powershell
python -m unittest discover -s tests -v
```

En la revisión de la refactorización se superaron **5 pruebas unitarias**. También se comprobaron los principales resultados de AP de V3 y los tamaños de los segmentos RM después de integrar los módulos. Estas comprobaciones no sustituyen una validación exhaustiva de todos los cuadernillos en cualquier entorno.

## Diccionario de datos

El documento [`docs/diccionario_datos_olist_ml.pdf`](docs/diccionario_datos_olist_ml.pdf) presenta las tablas originales, sus campos y las principales variables construidas para el análisis. Sirve como referencia para comprender las relaciones y transformaciones realizadas durante la preparación de datos.

## Limitaciones

- **Datos históricos:** los patrones no representan necesariamente el comportamiento actual del comercio electrónico.
- **Desbalance de clases:** la proporción de pedidos retrasados cambia entre períodos; por ello es importante interpretar AP frente a la tasa base.
- **Generalización:** los experimentos V2 y V3 requieren una validación independiente adicional antes de afirmar mejoras generalizables.
- **Calibración:** las puntuaciones de riesgo no deben asumirse automáticamente como probabilidades bien calibradas.
- **Frecuencia de compra:** la mayoría de compradores registra una sola compra, lo que limita el análisis clásico de frecuencia en RFM.
- **Interpretación comercial:** los segmentos no han sido validados mediante campañas, intervenciones o resultados operativos reales.

## Fuente de los datos

**Olist — Brazilian E-Commerce Public Dataset by Olist**, publicado en Kaggle:  
https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

El conjunto de datos es externo y se utiliza con fines académicos. Las variables derivadas, experimentos y conclusiones corresponden al trabajo realizado en este repositorio.

---

**Estado:** proyecto académico de análisis y experimentación reproducible mediante cuadernillos Jupyter.  
**Repositorio:** https://github.com/juandaguilar15/proyecto-olist-ml
