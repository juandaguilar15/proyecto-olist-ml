"""Validación temporal y comparaciones repetibles de Olist ML.

Integrantes: Juan David Aguilar Páez y Nicolás Vásquez.
"""
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score


def dividir_ventana_temporal(datos: pd.DataFrame, inicio, *, incluir_validacion=True):
    """Replica el corte usado en el cuadernillo 04.

    Histórico: compra < inicio Y entrega real < inicio.
    Validación: inicio <= compra < primer día del mes siguiente.
    La fecha de entrega se usa para delimitar el histórico ya conocido,
    nunca como característica de predicción.
    """
    inicio = pd.Timestamp(inicio)
    if pd.isna(inicio):
        raise ValueError("Fecha de inicio inválida")
    if inicio.day != 1 or inicio.hour or inicio.minute or inicio.second:
        raise ValueError("El inicio debe ser el primer día del mes a las 00:00")
    faltantes = [c for c in ("order_purchase_timestamp", "order_delivered_customer_date") if c not in datos]
    if faltantes:
        raise KeyError(f"Faltan columnas: {faltantes}")
    compra = pd.to_datetime(datos["order_purchase_timestamp"])
    entrega = pd.to_datetime(datos["order_delivered_customer_date"])
    historico = datos.loc[(compra < inicio) & (entrega < inicio)].copy()
    if not incluir_validacion:
        return historico
    fin = inicio + pd.DateOffset(months=1)
    validacion = datos.loc[(compra >= inicio) & (compra < fin)].copy()
    return historico, validacion


def evaluar_ventanas_temporales(datos, modelos, variables_por_modelo, meses, *, objetivo="retraso", nombre_columna="Modelo"):
    """Entrena clones y evalúa AP por cada ventana temporal y cada modelo.

    `modelos`: diccionario {nombre: estimador sklearn sin entrenar o entrenado}.
    `variables_por_modelo`: diccionario {nombre: lista de columnas}.
    Cada estimador se clona y se ajusta sobre el histórico de esa ventana.
    Devuelve columnas Mes, <nombre_columna>, AP, Tasa_retraso.
    No cambia los estimadores recibidos ni guarda modelos.
    """
    if objetivo not in datos:
        raise KeyError(f"Falta la variable objetivo '{objetivo}'")
    if set(modelos) != set(variables_por_modelo):
        raise ValueError("Los nombres de modelos y sus conjuntos de variables deben coincidir")
    resultados = []
    for inicio in meses:
        historico, periodo = dividir_ventana_temporal(datos, inicio)
        if historico.empty or periodo.empty:
            raise ValueError(f"Ventana sin suficientes registros: {pd.Timestamp(inicio).strftime('%Y-%m')}")
        for nombre, estimador_base in modelos.items():
            columnas = list(variables_por_modelo[nombre])
            modelo = clone(estimador_base)
            modelo.fit(historico[columnas], historico[objetivo])
            puntuacion = modelo.predict_proba(periodo[columnas])[:, 1]
            resultados.append({
                "Mes": pd.Timestamp(inicio).strftime("%Y-%m"),
                nombre_columna: nombre,
                "AP": average_precision_score(periodo[objetivo], puntuacion),
                "Tasa_retraso": periodo[objetivo].mean(),
            })
    return pd.DataFrame(resultados)
