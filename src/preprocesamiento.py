"""Construcción reutilizable del preprocesamiento para modelos supervisados.

Integrantes: Juan David Aguilar Páez y Nicolás Vásquez.
"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def crear_preprocesamiento(numericas, categoricas=()):
    """Replica el patrón del cuadernillo 04: mediana + StandardScaler;
    para variables categóricas, OneHotEncoder(handle_unknown='ignore').

    No ajusta transformaciones: se debe colocar el resultado dentro de un
    Pipeline entrenado solo sobre las filas históricas de cada ventana.
    """
    numericas = list(numericas)
    categoricas = list(categoricas)
    if not numericas and not categoricas:
        raise ValueError("Debe indicarse al menos una columna")
    if len(set(numericas + categoricas)) != len(numericas + categoricas):
        raise ValueError("Hay columnas duplicadas entre variables numéricas/categóricas")
    transformaciones = []
    if numericas:
        transformaciones.append((
            "num",
            Pipeline([
                ("imputador", SimpleImputer(strategy="median")),
                ("escalador", StandardScaler()),
            ]),
            numericas,
        ))
    if categoricas:
        transformaciones.append((
            "cat", OneHotEncoder(handle_unknown="ignore"), categoricas
        ))
    return ColumnTransformer(transformaciones)
