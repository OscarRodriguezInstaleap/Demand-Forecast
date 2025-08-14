# Versión 0.8 - Añade BLOQUE 4 (Productividad) y BLOQUE 6 (Fechas especiales) - 2025-08-14

import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly

st.set_page_config(page_title="Forecast App", layout="wide")
st.title("📦 Forecast de Demanda por Tienda - Instaleap")

# === BLOQUE 0: Selector de Modo de Visualización ===
modo = st.sidebar.selectbox("Selecciona modo de uso", ["Vista completa", "Solo Forecast", "Vista experimental"])

# === BLOQUE 1: Carga y limpieza de datos ===
st.sidebar.header("Paso 1: Carga tu archivo CSV")
archivo = st.sidebar.file_uploader("Selecciona el archivo CSV", type=["csv"])

if archivo is not None:
    try:
        df = pd.read_csv(archivo)
        st.subheader("Vista previa del archivo cargado")
        st.dataframe(df.head())

        columnas_necesarias = ['estado', 'slot_from', 'items', 'numero_pedido', 'Tienda']
        if not all(col in df.columns for col in columnas_necesarias):
            st.error(f"❌ El archivo debe contener las siguientes columnas: {columnas_necesarias}")
        else:
            # Filtro y estandarización
            df = df[df['estado'] == 'FINISHED'].copy()
            df['slot_from'] = pd.to_datetime(df['slot_from'], errors='coerce')
            df = df.dropna(subset=['slot_from'])
            df['fecha'] = df['slot_from'].dt.date
            df['hora'] = df['slot_from'].dt.hour
            df['items'] = pd.to_numeric(df['items'], errors='coerce')
            df = df.dropna(subset=['items'])

            # Base por tienda/fecha/hora
            agrupado = (
                df.groupby(['Tienda', 'fecha', 'hora'])
                .agg(pedidos=('numero_pedido', 'nunique'), items=('items', 'sum'))
                .reset_index()
            )

            st.subheader("🎯 Datos procesados por Tienda / Día / Hora")
            st.dataframe(agrupado)

            # Selector de tienda
            tiendas = agrupado['Tienda'].unique().tolist()
            tienda_seleccionada = st.selectbox("Selecciona una tienda para visualizar:", tiendas)
            df_tienda = agrupado[agrupado['Tienda'] == tienda_seleccionada]

            # === BLOQUE 2: Visualización histórica ===
            if modo in ["Vista completa", "Vista experimental"]:
                st.subheader(f"🟧 Heatmap de pedidos por hora en {tienda_seleccionada}")
                fig_heatmap_pedidos = px.density_heatmap(
                    df_tienda, x='hora', y='fecha', z='pedidos', histfunc='sum', nbinsx=24,
                    labels={'hora': 'Hora del día', 'fecha': 'Fecha', 'pedidos': 'Cantidad de pedidos'},
                    color_continuous_scale='Blues'
                )
                fig_heatmap_pedidos.update_layout(height=400, template='simple_white')
                st.plotly_chart(fig_heatmap_pedidos, use_container_width=True)

                st.subheader(f"🟦 Heatmap de ítems por hora en {tienda_seleccionada}")
                fig_heatmap_items = px.density_heatmap(
                    df_tienda, x='hora', y='fecha', z='items', histfunc='sum', nbinsx=24,
                    labels={'hora': 'Hora del día', 'fecha': 'Fecha', 'items': 'Cantidad de ítems'},
                    color_continuous_scale='Greens'
                )
                fig_heatmap_items.update_layout(height=400, template='simple_white')
                st.plotly_chart(fig_heatmap_items, use_container_width=True)

            # === BLOQUE 4: Métricas de productividad por tienda (Picking y Delivery) ===
            if modo in ["Vista completa", "Vista experimental"]:
                st.subheader("⚙️ Métricas de productividad por tienda")
                # Trabajamos desde la tabla original filtrando la tienda seleccionada
                df_store = df[df['Tienda'] == tienda_seleccionada].copy()

                # Convertir timestamps relevantes a datetime
                time_cols = [
                    'actual_inicio_picking', 'actual_fin_picking',
                    'actual_inicio_delivery', 'actual_fin_delivery'
                ]
                for c in time_cols:
                    if c in df_store.columns:
                        df_store[c] = pd.to_datetime(df_store[c], errors='coerce')

                # --- Picking: consolidar a nivel de orden x picker ---
                if {'picker'}.issubset(df_store.columns):
                    pick_base = (
                        df_store.dropna(subset=['picker'])
                        .groupby(['numero_pedido', 'picker'], as_index=False)
                        .agg(
                            items=('items', 'max'),
                            pick_start=('actual_inicio_picking', 'min'),
                            pick_end=('actual_fin_picking', 'max')
                        )
                    )
                    pick_base['min_picking'] = (pick_base['pick_end'] - pick_base['pick_start']).dt.total_seconds() / 60.0
                    pick_base = pick_base.dropna(subset=['min_picking'])
                    pick_base = pick_base[pick_base['min_picking'] > 0]

                    prod_picker = (
                        pick_base.groupby('picker')
                        .agg(
                            ordenes=('numero_pedido', 'nunique'),
                            items_totales=('items', 'sum'),
                            min_totales=('min_picking', 'sum'),
                            min_promedio_por_orden=('min_picking', 'mean')
                        )
                        .reset_index()
                    )
                    prod_picker['horas_totales'] = prod_picker['min_totales'] / 60.0
                    prod_picker['items_por_hora'] = prod_picker['items_totales'] / prod_picker['horas_totales']
                    prod_picker['ordenes_por_hora'] = prod_picker['ordenes'] / prod_picker['horas_totales']

                    cols_picker = ['picker', 'ordenes', 'items_totales', 'horas_totales', 'items_por_hora', 'ordenes_por_hora', 'min_promedio_por_orden']
                    prod_picker = prod_picker[cols_picker].round({'horas_totales': 2, 'items_por_hora': 2, 'ordenes_por_hora': 2, 'min_promedio_por_orden': 1})

                    st.markdown("##### 👷‍♂️ Picking — productividad por picker")
                    st.dataframe(prod_picker.sort_values('items_por_hora', ascending=False))
                else:
                    st.info("No se encontraron datos de 'picker' en el archivo para calcular productividad de picking.")

                # --- Delivery: consolidar a nivel de orden x driver ---
                if {'driver'}.issubset(df_store.columns):
                    deliv_base = (
                        df_store.dropna(subset=['driver'])
                        .groupby(['numero_pedido', 'driver'], as_index=False)
                        .agg(
                            deliv_start=('actual_inicio_delivery', 'min'),
                            deliv_end=('actual_fin_delivery', 'max')
                        )
                    )
                    deliv_base['min_delivery'] = (deliv_base['deliv_end'] - deliv_base['deliv_start']).dt.total_seconds() / 60.0
                    deliv_base = deliv_base.dropna(subset=['min_delivery'])
                    deliv_base = deliv_base[deliv_base['min_delivery'] > 0]

                    prod_driver = (
                        deliv_base.groupby('driver')
                        .agg(
                            ordenes=('numero_pedido', 'nunique'),
                            min_totales=('min_delivery', 'sum'),
                            min_promedio_por_orden=('min_delivery', 'mean')
                        )
                        .reset_index()
                    )
                    prod_driver['horas_totales'] = prod_driver['min_totales'] / 60.0
                    prod_driver['ordenes_por_hora'] = prod_driver['ordenes'] / prod_driver['horas_totales']

                    cols_driver = ['driver', 'ordenes', 'horas_totales', 'ordenes_por_hora', 'min_promedio_por_orden']
                    prod_driver = prod_driver[cols_driver].round({'horas_totales': 2, 'ordenes_por_hora': 2, 'min_promedio_por_orden': 1})

                    st.markdown("##### 🚚 Delivery — productividad por driver")
                    st.dataframe(prod_driver.sort_values('ordenes_por_hora', ascending=False))
                else:
                    st.info("No se encontraron datos de 'driver' en el archivo para calcular productividad de delivery.")

            # === BLOQUE 6: Ajuste por fechas especiales en forecast ===
            st.sidebar.markdown("### ⚡ Fechas especiales")
            usar_fechas_especiales = st.sidebar.checkbox("¿Aplicar un aumento por fechas especiales?", value=False)

            ajuste_fechas = None
            if usar_fechas_especiales:
                col1, col2 = st.sidebar.columns(2)
                fecha_inicio = col1.date_input("Inicio del evento especial")
                fecha_fin = col2.date_input("Fin del evento especial")
                incremento_pct = st.sidebar.slider("Incremento de demanda esperado (%)", 0, 200, 20)

                ajuste_fechas = {
                    "inicio": pd.to_datetime(fecha_inicio),
                    "fin": pd.to_datetime(fecha_fin),
                    "incremento": incremento_pct / 100
                }

            # === BLOQUE 3: Forecast de demanda ===
            st.subheader("🔮 Forecast de Demanda")
            dias_prediccion = st.number_input("¿Cuántos días quieres predecir? (1 a 31)", min_value=1, max_value=31, value=7)

            df_pred = df_tienda.groupby('fecha').agg({
                'pedidos': 'sum',
                'items': 'sum'
            }).reset_index()

            # Forecast pedidos
            st.markdown("#### 📈 Predicción de Pedidos Totales")
            df_pedidos = df_pred[['fecha', 'pedidos']].rename(columns={'fecha': 'ds', 'pedidos': 'y'})
            model_pedidos = Prophet()
            model_pedidos.fit(df_pedidos)
            future_pedidos = model_pedidos.make_future_dataframe(periods=dias_prediccion)
            forecast_pedidos = model_pedidos.predict(future_pedidos)

            if ajuste_fechas:
                mask = (forecast_pedidos['ds'] >= ajuste_fechas['inicio']) & (forecast_pedidos['ds'] <= ajuste_fechas['fin'])
                forecast_pedidos.loc[mask, ['yhat', 'yhat_lower', 'yhat_upper']] *= (1 + ajuste_fechas['incremento'])

            fig1 = plot_plotly(model_pedidos, forecast_pedidos)
            st.plotly_chart(fig1, use_container_width=True)

            st.markdown("##### Detalle numérico de la predicción de pedidos")
            st.dataframe(
                forecast_pedidos[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
                .tail(dias_prediccion)
                .rename(columns={
                    'ds': 'Fecha',
                    'yhat': 'Predicción',
                    'yhat_lower': 'Límite Inferior',
                    'yhat_upper': 'Límite Superior'
                })
            )

            # Forecast items
            st.markdown("#### 📈 Predicción de Ítems Totales")
            df_items = df_pred[['fecha', 'items']].rename(columns={'fecha': 'ds', 'items': 'y'})
            model_items = Prophet()
            model_items.fit(df_items)
            future_items = model_items.make_future_dataframe(periods=dias_prediccion)
            forecast_items = model_items.predict(future_items)

            if ajuste_fechas:
                mask = (forecast_items['ds'] >= ajuste_fechas['inicio']) & (forecast_items['ds'] <= ajuste_fechas['fin'])
                forecast_items.loc[mask, ['yhat', 'yhat_lower', 'yhat_upper']] *= (1 + ajuste_fechas['incremento'])

            fig2 = plot_plotly(model_items, forecast_items)
            st.plotly_chart(fig2, use_container_width=True)

            st.markdown("##### Detalle numérico de la predicción de ítems")
            st.dataframe(
                forecast_items[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
                .tail(dias_prediccion)
                .rename(columns={
                    'ds': 'Fecha',
                    'yhat': 'Predicción',
                    'yhat_lower': 'Límite Inferior',
                    'yhat_upper': 'Límite Superior'
                })
            )

    except Exception as e:
        st.error(f"Error al procesar el archivo: {e}")
else:
    st.info("⬅️ Por favor carga un archivo CSV para comenzar.")

