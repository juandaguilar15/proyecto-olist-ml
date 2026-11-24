"""Pruebas sintéticas, sin entrenar ni modificar datos reales de Olist."""
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.validacion import dividir_ventana_temporal, evaluar_ventanas_temporales
from src.evaluacion import evaluar_modelo
from src.preprocesamiento import crear_preprocesamiento
from src.segmentacion import transformar_indicadores_rfm


class TestModulos(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "order_purchase_timestamp": pd.to_datetime([
                "2017-08-01", "2017-08-12", "2017-09-03", "2017-09-12",
                "2017-10-05", "2017-10-12", "2017-11-05", "2017-11-07",
                "2017-11-10", "2017-11-15", "2017-11-20", "2017-11-21",
            ]),
            "order_delivered_customer_date": pd.to_datetime([
                "2017-08-05", "2017-08-17", "2017-09-08", "2017-09-19",
                "2017-10-10", "2017-10-18", "2017-11-12", "2017-11-11",
                "2017-11-15", "2017-11-22", "2017-11-28", "2017-12-02",
            ]),
            "peso": [2,4,6,8,10,12,14,16,18,20,22,24],
            "retraso": [0,1,0,1,0,1,0,0,1,1,0,1],
        })

    def test_corte_temporal_y_no_mutacion(self):
        df_original = self.df.copy(deep=True)
        hist, val = dividir_ventana_temporal(self.df, "2017-11-01")
        self.assertEqual(len(hist), 6)
        self.assertEqual(len(val), 6)
        self.assertTrue((hist["order_delivered_customer_date"] < "2017-11-01").all())
        pd.testing.assert_frame_equal(self.df, df_original)

    def test_metricas(self):
        r = evaluar_modelo("modelo", [0, 1, 0, 1], [0, 1, 0, 0], [0.1, 0.9, 0.2, 0.4])
        self.assertEqual(list(r.keys()), ["Modelo", "Accuracy", "Precision", "Recall", "F1", "AP"])
        self.assertAlmostEqual(r["Accuracy"], .75)

    def test_preprocesamiento(self):
        pre = crear_preprocesamiento(["peso"], [])
        X = pre.fit_transform(self.df)
        self.assertEqual(X.shape, (12, 1))
        self.assertAlmostEqual(float(np.mean(X)), 0, places=8)

    def test_segmentacion_equivalente(self):
        df = pd.DataFrame({"recencia_dias": [10, 20, 30], "gasto_total": [50., 200., 400.]})
        X, esc = transformar_indicadores_rfm(df, ["recencia_dias", "gasto_total"])
        np.testing.assert_allclose(X, esc.transform(np.log1p(df)))
        X2, esc2 = transformar_indicadores_rfm(df.iloc[:1], ["recencia_dias", "gasto_total"], escalador=esc)
        np.testing.assert_allclose(X2, X[:1])
        self.assertIs(esc2, esc)

    def test_comparacion_temporal(self):
        base = Pipeline([("pre", crear_preprocesamiento(["peso"])),
                         ("clf", LogisticRegression(max_iter=1000, random_state=42))])
        salida = evaluar_ventanas_temporales(self.df,
                       {"Logística": base}, {"Logística": ["peso"]}, ["2017-11-01"])
        self.assertEqual(list(salida.columns), ["Mes", "Modelo", "AP", "Tasa_retraso"])
        self.assertEqual(len(salida), 1)
        self.assertTrue(0 <= salida.iloc[0]["AP"] <= 1)
        self.assertFalse(hasattr(base.named_steps["clf"], "coef_"))


if __name__ == "__main__":
    unittest.main()
