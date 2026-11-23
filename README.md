
# Proyecto Olist ML

## Análisis de datos y Machine Learning aplicado al comercio electrónico

Proyecto académico de Ciencia de Datos desarrollado con el conjunto **Brazilian E-Commerce Public Dataset by Olist**.

El trabajo integra análisis exploratorio, preparación de datos, aprendizaje supervisado y no supervisado para investigar dos problemas del comercio electrónico:

- **Predicción de retrasos:** identificar pedidos con mayor riesgo de entregarse después de la fecha estimada.
- **Segmentación de clientes:** identificar perfiles de compradores según su recencia, frecuencia y gasto monetario.

## Integrantes del proyecto

- **Juan David Aguilar Páez**
- **Nicolás Vásquez**

**Universidad:** Universidad de Cundinamarca  
**Programa:** Ingeniería de Sistemas  
**Asignatura:** Ciencia de Datos

## Objetivos

### Objetivo general

Analizar datos históricos de comercio electrónico mediante técnicas de Ciencia de Datos y Machine Learning para identificar patrones relacionados con los retrasos en las entregas y el comportamiento de compra.

### Objetivos específicos

1. Comprender, explorar y preparar los datos de Olist.
2. Construir y evaluar modelos supervisados para predecir retrasos.
3. Aplicar técnicas de aprendizaje no supervisado para segmentar compradores.
4. Comparar resultados, identificar limitaciones y proponer posibles aplicaciones.

## Metodología

El proyecto se organiza en seis cuadernillos Jupyter:

| Cuadernillo | Contenido |
|---|---|
| `01_comprension_datos.ipynb` | Comprensión de las tablas y relaciones del conjunto Olist |
| `02_analisis_exploratorio.ipynb` | Exploración de variables, pedidos y patrones |
| `03_preparacion_datos.ipynb` | Limpieza, integración y preparación para Machine Learning |
| `04_modelos_supervisados.ipynb` | Predicción de retrasos y validación temporal |
| `05_segmentacion_clientes.ipynb` | Análisis RFM, K-Means y comparación de segmentos |
| `06_evaluacion_integral.ipynb` | Integración de resultados, conclusiones y limitaciones |

## Tecnologías

- Python
- Pandas y NumPy
- Matplotlib
- Scikit-learn
- XGBoost y CatBoost
- Jupyter Notebook
- Visual Studio Code
- Git y GitHub


## Instalación y ejecución

### Requisitos previos

Para reproducir el proyecto se recomienda disponer de:

- Python 3.13.
- Git.
- Visual Studio Code con las extensiones Python y Jupyter.
- Conexión a Internet para descargar el conjunto de datos e instalar las dependencias.

Los comandos siguientes están orientados a Windows y PowerShell.

### 1. Clonar el repositorio

Descargar el proyecto desde GitHub:

```powershell
git clone https://github.com/juandaguilar15/proyecto-olist-ml.git
cd proyecto-olist-ml
```

Sustituir `URL_DEL_REPOSITORIO` por la dirección HTTPS real del repositorio.

### 2. Crear un entorno virtual

Desde la carpeta principal del proyecto:

```powershell
py -3.13 -m venv .venv
```

Activarlo:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Instalar las dependencias

Con el entorno virtual activado:

```powershell
python -m pip install -r requirements.txt
```

Comprobar las dependencias instaladas:

```powershell
python -m pip check
```

### 4. Descargar los datos originales

Descargar el conjunto **Brazilian E-Commerce Public Dataset by Olist** desde Kaggle:

https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

Descomprimir el conjunto y colocar los nueve archivos CSV originales dentro de:

```text
data/raw/
```

Los archivos deben conservar sus nombres originales.

### 5. Abrir el proyecto en Visual Studio Code

Desde la carpeta principal del proyecto:

```powershell
code .
```

Abrir la carpeta `notebooks/` y seleccionar el intérprete de Python del entorno virtual `.venv` como kernel de los cuadernillos.

Los notebooks utilizan rutas relativas como `../data/raw/`, por lo que deben ejecutarse con `notebooks/` como directorio de trabajo.

### 6. Ejecutar los cuadernillos

Ejecutar los siguientes notebooks en este orden:

1. `01_comprension_datos.ipynb`: comprensión inicial.
2. `02_analisis_exploratorio.ipynb`: análisis exploratorio.
3. `03_preparacion_datos.ipynb`: preparación y generación de archivos procesados.
4. `04_modelos_supervisados.ipynb`: entrenamiento y evaluación de clasificación.
5. `05_segmentacion_clientes.ipynb`: agrupamiento y perfiles de compradores.
6. `06_evaluacion_integral.ipynb`: integración e interpretación final.

Los cuadernillos 04 y 05 generan archivos CSV dentro de `resultados/`, utilizados posteriormente por el cuadernillo 06.

### 7. Consideraciones de reproducibilidad

- Los datos originales y procesados no se incluyen en el repositorio porque están excluidos mediante `.gitignore`.
- Es necesario ejecutar la preparación antes de los modelos supervisados.
- Los resultados pueden presentar diferencias menores debido a versiones, condiciones de ejecución o aleatoriedad.
- Los experimentos supervisados V3 son exploratorios y requieren validación independiente adicional.
- La ejecución completa desde un entorno recién creado todavía no ha sido verificada.


## Resultados principales

### Aprendizaje supervisado

Se evaluaron algoritmos de clasificación para identificar pedidos con riesgo de retraso.

En las validaciones temporales exploratorias de la versión V3 se obtuvieron los siguientes resultados:

| Período | AP regresión logística V3 | AP Random Forest V3 |
|---|---:|---:|
| Noviembre de 2017 | 0,1659 | 0,1135 |
| Febrero de 2018 | 0,2795 | 0,2593 |
| Abril de 2018 | 0,1379 | 0,1295 |

La regresión logística obtuvo mayor Average Precision en los tres períodos.

**Limitación:** las mejoras V3 requieren una evaluación final independiente posterior al desarrollo experimental. Los resultados no demuestran que el modelo esté listo para producción.

### Aprendizaje no supervisado

Se construyeron indicadores RFM para **93.357 compradores** con pedidos entregados e información de pago.

Se compararon dos alternativas:

- **RM:** recencia y gasto monetario.
- **RFM:** recencia, frecuencia y gasto monetario.

El modelo RM con K=3 se seleccionó experimentalmente por su interpretabilidad:

| Segmento RM | Compradores | Perfil |
|---|---:|---|
| 0 | 16.380 | Compradores recientes |
| 1 | 43.791 | Compradores menos recientes de bajo gasto |
| 2 | 33.186 | Compradores menos recientes de alto gasto |

Se identificó que el **97 % de los compradores realizó una sola compra registrada**.

El modelo complementario RFM con K=3 separó a los 2.801 compradores recurrentes en un segmento exclusivo.

## Estructura del repositorio

```text


proyecto-olist-ml/
├── data/
│   ├── raw/                      # Datos originales de Olist
│   └── processed/                # Datos preparados para los modelos
├── notebooks/
│   ├── 01_comprension_datos.ipynb
│   ├── 02_analisis_exploratorio.ipynb
│   ├── 03_preparacion_datos.ipynb
│   ├── 04_modelos_supervisados.ipynb
│   ├── 05_segmentacion_clientes.ipynb
│   └── 06_evaluacion_integral.ipynb
├── presentacion/
│   ├── 01_Comprensión de Datos Olist .pdf
│   ├── 02_Análisis Exploratorio de Retrasos Logísticos — Olist.pdf
│   ├── 03_Preparación de Datos para Machine Learning — Olist.pdf
│   ├── 04_modelos_supervisados.ipynb.pdf
│   ├── 05_Segmentación de Clientes con RFM y K-Means.pdf
│   ├── 06_evaluacion_integral_presentacion.pdf
│   ├── Presentacion_Final_Olist_ML.pdf
│   └── Presentacion_Olist_ML_Final.pdf
├── resultados/
│   ├── comparacion_algoritmos_v3.csv
│   ├── validacion_temporal_v3.csv
│   ├── resumen_segmentos_rm.csv
│   ├── resumen_segmentos_rfm.csv
│   └── segmentacion_clientes.csv
├── src/
│   ├── __init__.py
│   └── preparacion.py
├── .gitignore
├── README.md
└── requirements.txt


```


## Limitaciones

- Los datos corresponden a un período histórico.
- El problema de retrasos presenta desbalance de clases.
- El rendimiento supervisado varía entre períodos.
- La calibración de probabilidades no está consolidada.
- La frecuencia de recompra registrada es baja.
- La utilidad comercial de los segmentos no ha sido validada.

## Próximos pasos

- Revisar la reproducibilidad y las dependencias del proyecto.
- Completar la documentación académica.
- Organizar los materiales para la sustentación.
- Explorar evaluaciones adicionales con datos independientes.

## Fuente de datos

Brazilian E-Commerce Public Dataset by Olist, disponible en Kaggle.

## Estado del proyecto

Proyecto académico en desarrollo. Los resultados corresponden a análisis exploratorios y experimentales.
