"""Métricas compartidas de clasificación para Olist ML.

Integrantes: Juan David Aguilar Páez y Nicolás Vásquez.
"""
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)


def evaluar_modelo(nombre, y_real, predicciones, probabilidades):
    """Devuelve exactamente las métricas y nombres del cuadernillo 04.

    `probabilidades` representa la puntuación de la clase positiva (retraso=1).
    """
    return {
        "Modelo": nombre,
        "Accuracy": accuracy_score(y_real, predicciones),
        "Precision": precision_score(y_real, predicciones, zero_division=0),
        "Recall": recall_score(y_real, predicciones, zero_division=0),
        "F1": f1_score(y_real, predicciones, zero_division=0),
        "AP": average_precision_score(y_real, probabilidades),
    }
