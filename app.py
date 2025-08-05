import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly

st.set_page_config(page_title="Forecast App", layout="wide")

st.title("📦 Forecast de Demanda por Tienda - Instaleap")

# Paso 1: Cargar archivo CSV
st.sidebar.header("Paso 1: Carga tu archivo CSV")
archivo = st.sidebar.file_uploader("Selecciona el archivo CSV", type=["csv"])

if archivo is not None:
    try:
        # Leer el archivo
        df = pd.read_csv(archivo)

        # Mostrar primeras filas
        st.subheader("Vista previa del archivo cargado")
        st.dataframe(df.head())

        # Validar columnas necesarias
        columnas_necesarias = ['estado', 'slot_from', 'items', 'numero_pedido', 'Tienda']
        if not all(col in df.columns for col in columnas_necesarias):
            st.error(f"❌ El archivo debe contener las siguientes columnas: {columnas_necesarias}")
        else:
            # Filtrar pedidos finalizados
            df = df[df['estado'] == 'FINISHED'].copy()

            # Convertir slot_from a datetime
            df['slot_from'] = pd.to_datetime(df['slot_from'], errors='coerce')
            df = df.dropna(subset=['slot_from'])
            df['fecha'] = df['slot_from'].dt.date
            df['hora'] = df['slot_from'].dt.hour

            # Asegurar que 'items' es numérico
            df['items'] = pd.to_numeric(df['items'], errors='coerce')
            df = df.dropna(subset=['items'])

            # Agrupar
            agrupado = (
                df.groupby(['Tienda', 'fecha', 'hora'])
                .agg(pedidos=('numero_pedido', 'nunique'), items=('items', 'sum'))
                .reset_index()
            )

            st.subheader("🎯 Datos procesados por Tienda / Día / Hora")
            st.dataframe(agrupado)

            # Selección de tienda para análisis
            tiendas = agrupado['Tienda'].unique().tolist()
            tienda_seleccionada = st.selectbox("Selecciona una tienda para visualizar:", tiendas)

            df_tienda = agrupado[agrupado['Tienda'] == tienda_seleccionada]

            # Mostrar gráfico de pedidos por hora
            st.subheader(f"📊 Pedidos por hora en {tienda_seleccionada}")
            fig_pedidos = px.line(df_tienda, x='hora', y='pedidos', color=df_tienda['fecha'].astype(str),
                                  labels={'hora': 'Hora del día', 'pedidos': 'Pedidos'},
                                  title="Evolución de pedidos por hora")
            st.plotly_chart(fig_pedidos, use_container_width=True)

            # Mostrar gráfico de items por hora
            st.subheader(f"📦 Ítems por hora en {tienda_seleccionada}")
            fig_items = px.line(df_tienda, x='hora', y='items', color=df_tienda['fecha'].astype(str),
                                labels={'hora': 'Hora del día', 'items': 'Items'},
                                title="Evolución de ítems por hora")
            st.plotly_chart(fig_items, use_container_width=True)

            # Forecast
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

            # Forecast items
            st.markdown("#### 📈 Predicción de Ítems Totales")
            df_items = df_pred[['fecha', 'items']].rename(columns={'fecha': 'ds', 'items': 'y'})
            model_items = Prophet()
            model_items.fit(df_items)
            future_items = model_items.make_future_dataframe(periods=dias_prediccion)
            forecast_items = model_items.predict(future_items)
            fig2 = plot_plotly(model_items, forecast_items)
            st.plotly_chart(fig2, use_container_width=True)

    except Exception as e:
        st.error(f"Error al procesar el archivo: {e}")
else:
    st.info("⬅️ Por favor carga un archivo CSV para comenzar.")
