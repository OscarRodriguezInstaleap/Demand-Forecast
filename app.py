# Versión 0.6 - Modularizada con selector de modo y bloques independientes - 2025-08-05

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
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
            df = df[df['estado'] == 'FINISHED'].copy()
            df['slot_from'] = pd.to_datetime(df['slot_from'], errors='coerce')
            df = df.dropna(subset=['slot_from'])
            df['fecha'] = df['slot_from'].dt.date
            df['hora'] = df['slot_from'].dt.hour
            df['items'] = pd.to_numeric(df['items'], errors='coerce')
            df = df.dropna(subset=['items'])

            agrupado = (
                df.groupby(['Tienda', 'fecha', 'hora'])
                .agg(pedidos=('numero_pedido', 'nunique'), items=('items', 'sum'))
                .reset_index()
            )

            st.subheader("🎯 Datos procesados por Tienda / Día / Hora")
            st.dataframe(agrupado)

            tiendas = agrupado['Tienda'].unique().tolist()
            tienda_seleccionada = st.selectbox("Selecciona una tienda para visualizar:", tiendas)
            df_tienda = agrupado[agrupado['Tienda'] == tienda_seleccionada]

            # === BLOQUE 2: Visualización histórica (solo en Vista completa o experimental) ===
            if modo in ["Vista completa", "Vista experimental"]:
                st.subheader(f"🟧 Heatmap de pedidos por hora en {tienda_seleccionada}")
                fig_heatmap_pedidos = px.density_heatmap(
                    df_tienda,
                    x='hora',
                    y='fecha',
                    z='pedidos',
                    histfunc='sum',
                    nbinsx=24,
                    labels={'hora': 'Hora del día', 'fecha': 'Fecha', 'pedidos': 'Cantidad de pedidos'},
                    color_continuous_scale='Blues'
                )
                fig_heatmap_pedidos.update_layout(height=400, template='simple_white')
                st.plotly_chart(fig_heatmap_pedidos, use_container_width=True)

                st.subheader(f"🟦 Heatmap de ítems por hora en {tienda_seleccionada}")
                fig_heatmap_items = px.density_heatmap(
                    df_tienda,
                    x='hora',
                    y='fecha',
                    z='items',
                    histfunc='sum',
                    nbinsx=24,
                    labels={'hora': 'Hora del día', 'fecha': 'Fecha', 'items': 'Cantidad de ítems'},
                    color_continuous_scale='Greens'
                )
                fig_heatmap_items.update_layout(height=400, template='simple_white')
                st.plotly_chart(fig_heatmap_items, use_container_width=True)

# === BLOQUE 3: Forecast de demanda (siempre visible en los tres modos) ===
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
fig1 = plot_plotly(model_pedidos, forecast_pedidos)
st.plotly_chart(fig1, use_container_width=True)

# Tabla detallada de predicción de pedidos
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
fig2 = plot_plotly(model_items, forecast_items)
st.plotly_chart(fig2, use_container_width=True)

# Tabla detallada de predicción de ítems
st.markdown("##### Detalle numérico de la predicción de ítems")
st.dataframe(


    except Exception as e:
        st.error(f"Error al procesar el archivo: {e}")
else:
    st.info("⬅️ Por favor carga un archivo CSV para comenzar.")
