"""Transformaciones repetidas en segmentación de clientes RFM y RM.

Integrantes: Juan David Aguilar Páez y Nicolás Vásquez.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler


def transformar_indicadores_rfm(datos, columnas, *, escalador=None):
    """Aplica log1p y StandardScaler, reproduciendo el cuadernillo 05.

    Al entrenar: `X, scaler = transformar_indicadores_rfm(df, columnas)`.
    Para nuevos compradores: pasar `escalador=scaler` ya ajustado, sin fit.
    Conserva el orden de las columnas proporcionadas.
    """
    columnas = list(columnas)
    if not columnas:
        raise ValueError("La lista de columnas no puede estar vacía")
    faltantes = sorted(set(columnas) - set(datos.columns))
    if faltantes:
        raise KeyError(f"Faltan columnas: {faltantes}")
    valores = datos[columnas]
    if valores.isna().any().any():
        raise ValueError("Hay indicadores RFM nulos; revisa su preparación")
    if (valores < 0).any().any():
        raise ValueError("Los indicadores RFM deben ser no negativos")
    transformados = np.log1p(valores)
    if escalador is None:
        escalador = StandardScaler()
        X = escalador.fit_transform(transformados)
    else:
        X = escalador.transform(transformados)
    return X, escalador
