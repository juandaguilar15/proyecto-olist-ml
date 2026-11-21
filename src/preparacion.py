import pandas as pd


def preparar_objetivo(pedidos):
    fechas = [
        "order_delivered_customer_date",
        "order_estimated_delivery_date"
    ]

    datos = pedidos.loc[pedidos["order_status"] == "delivered"].copy()
    datos[fechas] = datos[fechas].apply(pd.to_datetime)
    datos = datos.dropna(subset=fechas)

    datos["retraso"] = (
        datos["order_delivered_customer_date"].dt.normalize()
        > datos["order_estimated_delivery_date"].dt.normalize()
    ).astype(int)

    return datos

def preparar_articulos(items, productos):
    articulos = items.merge(
        productos[["product_id", "product_weight_g"]],
        on="product_id",
        how="left",
        validate="many_to_one"
    )

    return articulos.groupby("order_id").agg(
        cantidad_items=("product_id", "size"),
        peso_total_g=("product_weight_g", lambda x: x.sum(min_count=len(x))),
        costo_envio=("freight_value", "sum")
    ).reset_index()

def preparar_envios(items, vendedores, pedidos, clientes):
    ubicaciones = (
        items[["order_id", "seller_id"]]
        .merge(
            vendedores[["seller_id", "seller_state"]],
            on="seller_id",
            validate="many_to_one"
        )
        .merge(
            pedidos[["order_id", "customer_id"]],
            on="order_id",
            validate="many_to_one"
        )
        .merge(
            clientes[["customer_id", "customer_state"]],
            on="customer_id",
            validate="many_to_one"
        )
    )

    ubicaciones["estado_diferente"] = (
        ubicaciones["seller_state"] != ubicaciones["customer_state"]
    )

    return (
        ubicaciones.groupby("order_id")["estado_diferente"]
        .any()
        .rename("envio_entre_estados")
        .reset_index()
    )