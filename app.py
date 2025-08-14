# Versión 1.1 - UX mejoras (colapsables), 3 heatmaps, top10 gráficos, forecast hasta 90d, dimensionamiento y CSVs - 2025-08-14

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet
from prophet.plot import plot_plotly
from datetime import datetime
import tempfile
import matplotlib.pyplot as plt
import math
import io
import zipfile

st.set_page_config(page_title="Forecast App", layout="wide")
st.title("Forecast de Demanda por Tienda - Instaleap")

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

            with st.expander("Datos procesados por Tienda / Día / Hora (colapsado)", expanded=False):
                st.dataframe(agrupado)

            # Selector de tienda
            tiendas = agrupado['Tienda'].unique().tolist()
            tienda_seleccionada = st.selectbox("Selecciona una tienda para visualizar:", tiendas)
            df_tienda = agrupado[agrupado['Tienda'] == tienda_seleccionada].copy()

            # Guardar rango de fechas para reportes
            fecha_min = pd.to_datetime(df_tienda['fecha']).min()
            fecha_max = pd.to_datetime(df_tienda['fecha']).max()

            # === BLOQUE 2: Visualización histórica ===
            if modo in ["Vista completa", "Vista experimental"]:
                # 2 heatmaps en la misma fila
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader(f"Heatmap de pedidos por hora — {tienda_seleccionada}")
                    fig_heatmap_pedidos = px.density_heatmap(
                        df_tienda, x='hora', y='fecha', z='pedidos', histfunc='sum', nbinsx=24,
                        labels={'hora': 'Hora del día', 'fecha': 'Fecha', 'pedidos': 'Cantidad de pedidos'},
                        color_continuous_scale='Blues'
                    )
                    fig_heatmap_pedidos.update_layout(height=400, template='simple_white')
                    st.plotly_chart(fig_heatmap_pedidos, use_container_width=True)
                with c2:
                    st.subheader(f"Heatmap de items por hora — {tienda_seleccionada}")
                    fig_heatmap_items = px.density_heatmap(
                        df_tienda, x='hora', y='fecha', z='items', histfunc='sum', nbinsx=24,
                        labels={'hora': 'Hora del día', 'fecha': 'Fecha', 'items': 'Cantidad de items'},
                        color_continuous_scale='Greens'
                    )
                    fig_heatmap_items.update_layout(height=400, template='simple_white')
                    st.plotly_chart(fig_heatmap_items, use_container_width=True)

                # 3er heatmap: día de la semana vs hora (pedidos)
                df_tienda_week = df_tienda.copy()
                df_tienda_week['fecha_dt'] = pd.to_datetime(df_tienda_week['fecha'])
                df_tienda_week['dow_num'] = df_tienda_week['fecha_dt'].dt.weekday  # 0=Lunes
                mapa_dow = {0:'Lunes',1:'Martes',2:'Miércoles',3:'Jueves',4:'Viernes',5:'Sábado',6:'Domingo'}
                df_tienda_week['dia_semana'] = df_tienda_week['dow_num'].map(mapa_dow)
                orden_dow = ['Lunes','Martes','Miércoles','Jueves','Viernes','Sábado','Domingo']
                df_tienda_week['dia_semana'] = pd.Categorical(df_tienda_week['dia_semana'], categories=orden_dow, ordered=True)

                st.subheader("Heatmap de pedidos por día de la semana y hora")
                fig_heatmap_dow = px.density_heatmap(
                    df_tienda_week, x='hora', y='dia_semana', z='pedidos', histfunc='sum', nbinsx=24,
                    labels={'hora': 'Hora del día', 'dia_semana': 'Día de la semana', 'pedidos': 'Cantidad de pedidos'},
                    color_continuous_scale='Blues'
                )
                fig_heatmap_dow.update_layout(height=380, template='simple_white')
                st.plotly_chart(fig_heatmap_dow, use_container_width=True)

            # === BLOQUE 4: Métricas de productividad por tienda (Picking y Delivery) ===
            prod_picker = None
            prod_driver = None
            if modo in ["Vista completa", "Vista experimental"]:
                st.subheader("Métricas de productividad por tienda")
                df_store = df[df['Tienda'] == tienda_seleccionada].copy()

                # Convertir timestamps relevantes a datetime si existen
                time_cols = [
                    'actual_inicio_picking', 'actual_fin_picking',
                    'actual_inicio_delivery', 'actual_fin_delivery'
                ]
                for c in time_cols:
                    if c in df_store.columns:
                        df_store[c] = pd.to_datetime(df_store[c], errors='coerce')

                # --- Picking
                if 'picker' in df_store.columns:
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

                    # Top 10 gráfico: barras (ordenes) + línea (items/hora)
                    top_pick = prod_picker.sort_values('ordenes', ascending=False).head(10)
                    fig_top_pick = make_subplots(specs=[[{"secondary_y": True}]])
                    fig_top_pick.add_trace(
                        go.Bar(x=top_pick['picker'], y=top_pick['ordenes'], name='Órdenes'),
                        secondary_y=False
                    )
                    fig_top_pick.add_trace(
                        go.Scatter(x=top_pick['picker'], y=top_pick['items_por_hora'], mode='lines+markers', name='Items/hora'),
                        secondary_y=True
                    )
                    fig_top_pick.update_layout(template='simple_white', height=420, legend_title_text='')
                    fig_top_pick.update_xaxes(title_text='Picker (Top 10)')
                    fig_top_pick.update_yaxes(title_text='Órdenes', secondary_y=False)
                    fig_top_pick.update_yaxes(title_text='Items/hora', secondary_y=True)
                    st.plotly_chart(fig_top_pick, use_container_width=True)

                    with st.expander("Tabla de productividad - Picking (colapsada)", expanded=False):
                        cols_picker = ['picker', 'ordenes', 'items_totales', 'horas_totales', 'items_por_hora', 'ordenes_por_hora', 'min_promedio_por_orden']
                        prodp = prod_picker[cols_picker].round({'horas_totales': 2, 'items_por_hora': 2, 'ordenes_por_hora': 2, 'min_promedio_por_orden': 1})
                        st.dataframe(prodp.sort_values('items_por_hora', ascending=False))
                else:
                    st.info("No se encontraron datos de 'picker' en el archivo para calcular productividad de picking.")

                # --- Delivery
                if 'driver' in df_store.columns:
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

                    # Top 10 gráfico: barras (órdenes) + línea (órdenes/hora)
                    top_drv = prod_driver.sort_values('ordenes', ascending=False).head(10)
                    fig_top_drv = make_subplots(specs=[[{"secondary_y": True}]])
                    fig_top_drv.add_trace(
                        go.Bar(x=top_drv['driver'], y=top_drv['ordenes'], name='Órdenes'),
                        secondary_y=False
                    )
                    fig_top_drv.add_trace(
                        go.Scatter(x=top_drv['driver'], y=top_drv['ordenes_por_hora'], mode='lines+markers', name='Órdenes/hora'),
                        secondary_y=True
                    )
                    fig_top_drv.update_layout(template='simple_white', height=420, legend_title_text='')
                    fig_top_drv.update_xaxes(title_text='Driver (Top 10)')
                    fig_top_drv.update_yaxes(title_text='Órdenes', secondary_y=False)
                    fig_top_drv.update_yaxes(title_text='Órdenes/hora', secondary_y=True)
                    st.plotly_chart(fig_top_drv, use_container_width=True)

                    with st.expander("Tabla de productividad - Delivery (colapsada)", expanded=False):
                        cols_driver = ['driver', 'ordenes', 'horas_totales', 'ordenes_por_hora', 'min_promedio_por_orden']
                        prodd = prod_driver[cols_driver].round({'horas_totales': 2, 'ordenes_por_hora': 2, 'min_promedio_por_orden': 1})
                        st.dataframe(prodd.sort_values('ordenes_por_hora', ascending=False))
                else:
                    st.info("No se encontraron datos de 'driver' en el archivo para calcular productividad de delivery.")

            # === BLOQUE 6: Ajuste por fechas especiales en forecast ===
            st.sidebar.markdown("### Fechas especiales")
            usar_fechas_especiales = st.sidebar.checkbox("¿Aplicar un aumento por fechas especiales?", value=False)
            ajuste_fechas = None
            if usar_fechas_especiales:
                col1, col2 = st.sidebar.columns(2)
                fecha_inicio = col1.date_input("Inicio del evento especial")
                fecha_fin = col2.date_input("Fin del evento especial")
                incremento_pct = st.sidebar.slider("Incremento de demanda esperado (%)", 0, 200, 20)
                ajuste_fechas = {"inicio": pd.to_datetime(fecha_inicio), "fin": pd.to_datetime(fecha_fin), "incremento": incremento_pct / 100}

            # === BLOQUE 3: Forecast de demanda ===
            st.subheader("Forecast de Demanda")
            dias_prediccion = st.number_input("¿Cuántos días quieres predecir? (1 a 90)", min_value=1, max_value=90, value=7)

            df_pred = df_tienda.groupby('fecha').agg({'pedidos': 'sum','items': 'sum'}).reset_index()

            # Forecast pedidos
            st.markdown("#### Predicción de Pedidos Totales")
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
            tabla_ped = forecast_pedidos[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(dias_prediccion).rename(columns={'ds': 'Fecha','yhat': 'Predicción','yhat_lower': 'Límite Inferior','yhat_upper': 'Límite Superior'})
            st.dataframe(tabla_ped)

            # Forecast items
            st.markdown("#### Predicción de Items Totales")
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
            tabla_items = forecast_items[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(dias_prediccion).rename(columns={'ds': 'Fecha','yhat': 'Predicción','yhat_lower': 'Límite Inferior','yhat_upper': 'Límite Superior'})
            st.dataframe(tabla_items)

            # === BLOQUE 7: Dimensionamiento de recursos (correlación demanda-productividad) ===
            st.subheader("Dimensionamiento de recursos (estimado)")
            # Tasas base desde productividad si existen, si no valores por defecto
            items_rate_default = None
            orders_rate_default = None
            if prod_picker is not None and len(prod_picker) > 0:
                items_rate_default = float(prod_picker['items_por_hora'].median())
            if prod_driver is not None and len(prod_driver) > 0:
                orders_rate_default = float(prod_driver['ordenes_por_hora'].median())
            if items_rate_default is None or math.isnan(items_rate_default):
                items_rate_default = 60.0  # fallback
            if orders_rate_default is None or math.isnan(orders_rate_default):
                orders_rate_default = 3.0  # fallback

            cA, cB, cC = st.columns(3)
            items_por_hora = cA.number_input("Rendimiento picker (items/hora)", min_value=1.0, max_value=1000.0, value=float(round(items_rate_default,2)))
            ordenes_por_hora = cB.number_input("Rendimiento driver (órdenes/hora)", min_value=0.5, max_value=50.0, value=float(round(orders_rate_default,2)))
            horas_turno = cC.number_input("Horas por turno", min_value=1.0, max_value=12.0, value=8.0)
            factor_seguridad = st.slider("Factor de seguridad adicional (%)", min_value=0, max_value=100, value=10) / 100.0

            # Construir tabla de dimensionamiento para los próximos dias_prediccion
            dim = pd.DataFrame({
                'Fecha': forecast_pedidos['ds'].tail(dias_prediccion).reset_index(drop=True)
            })
            dim['Pedidos (yhat)'] = forecast_pedidos['yhat'].tail(dias_prediccion).values
            dim['Items (yhat)'] = forecast_items['yhat'].tail(dias_prediccion).values
            dim['Pedidos (low)'] = forecast_pedidos['yhat_lower'].tail(dias_prediccion).values
            dim['Pedidos (high)'] = forecast_pedidos['yhat_upper'].tail(dias_prediccion).values
            dim['Items (low)'] = forecast_items['yhat_lower'].tail(dias_prediccion).values
            dim['Items (high)'] = forecast_items['yhat_upper'].tail(dias_prediccion).values

            # Horas requeridas
            dim['Horas picker (yhat)'] = dim['Items (yhat)'] / max(items_por_hora, 1e-9)
            dim['Horas driver (yhat)'] = dim['Pedidos (yhat)'] / max(ordenes_por_hora, 1e-9)

            # Recursos (con factor de seguridad)
            def ceil_div(x, y):
                return math.ceil(x / y) if y > 0 else 0
            dim['Pickers necesarios (yhat)'] = [ceil_div(h*(1+factor_seguridad), horas_turno) for h in dim['Horas picker (yhat)']]
            dim['Drivers necesarios (yhat)'] = [ceil_div(h*(1+factor_seguridad), horas_turno) for h in dim['Horas driver (yhat)']]

            st.dataframe(dim)

            # === BLOQUE 5: Exportación de reporte PDF (Matplotlib, sin Chrome) ===
            st.sidebar.markdown("### Reporte PDF")
            inc_portada = st.sidebar.checkbox("Incluir portada (tienda / rango de fechas)", True)
            inc_heatmaps = st.sidebar.checkbox("Incluir heatmaps", True)
            inc_forecast = st.sidebar.checkbox("Incluir resumen de forecast", True)
            inc_productividad = st.sidebar.checkbox("Incluir tablas de productividad", True)

            if st.sidebar.button("Generar PDF"):
                try:
                    from fpdf import FPDF  # requiere paquete fpdf2
                except Exception as e:
                    st.error("Para exportar PDF, agrega 'fpdf2' a requirements.txt e intenta de nuevo.")
                else:
                    # --- Helpers Matplotlib -> PNG ---
                    def heatmap_png_from_df(df_src, value_col, title):
                        pivot = df_src.pivot_table(index='fecha', columns='hora', values=value_col, aggfunc='sum').sort_index()
                        if pivot.size == 0:
                            return None
                        fig, ax = plt.subplots(figsize=(10, 4), dpi=200)
                        im = ax.imshow(pivot.values, aspect='auto', interpolation='nearest')
                        ax.set_title(title)
                        ax.set_xlabel('Hora del día')
                        ax.set_ylabel('Fecha')
                        ax.set_xticks(range(len(pivot.columns)))
                        ax.set_xticklabels(pivot.columns, fontsize=7)
                        y_labels = [str(d) for d in pivot.index]
                        step = max(1, len(y_labels)//10)
                        ax.set_yticks(range(0, len(y_labels), step))
                        ax.set_yticklabels(y_labels[0::step], fontsize=7)
                        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                        cbar.ax.tick_params(labelsize=7)
                        tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                        fig.tight_layout()
                        fig.savefig(tmp.name, bbox_inches='tight')
                        plt.close(fig)
                        return tmp.name

                    def prophet_png(model, forecast, title):
                        fig = model.plot(forecast)
                        ax = fig.axes[0]
                        ax.set_title(title)
                        ax.set_xlabel('Fecha')
                        ax.set_ylabel('Demanda')
                        tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                        fig.tight_layout()
                        fig.savefig(tmp.name, bbox_inches='tight', dpi=200)
                        plt.close(fig)
                        return tmp.name

                    # Construcción del PDF
                    pdf = FPDF(orientation='P', unit='mm', format='A4')
                    pdf.set_auto_page_break(auto=True, margin=12)

                    # Portada
                    if inc_portada:
                        pdf.add_page()
                        pdf.set_font('Helvetica', 'B', 18)
                        pdf.cell(0, 12, 'Reporte de Demanda - Instaleap', ln=1)
                        pdf.set_font('Helvetica', size=12)
                        pdf.cell(0, 8, f'Tienda: {tienda_seleccionada}', ln=1)
                        ran = f'Rango historico: {fecha_min.date()} a {fecha_max.date()}'
                        pdf.cell(0, 8, ran, ln=1)
                        pdf.cell(0, 8, f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=1)

                    # Heatmaps
                    if inc_heatmaps:
                        img1 = heatmap_png_from_df(df_tienda, 'pedidos', 'Heatmap - Pedidos')
                        img2 = heatmap_png_from_df(df_tienda, 'items', 'Heatmap - Items')
                        if img1 or img2:
                            pdf.add_page()
                            if img1:
                                pdf.set_font('Helvetica', 'B', 14)
                                pdf.cell(0, 8, 'Heatmap - Pedidos', ln=1)
                                pdf.image(img1, x=10, y=None, w=190)
                                pdf.ln(4)
                            if img2:
                                pdf.set_font('Helvetica', 'B', 14)
                                pdf.cell(0, 8, 'Heatmap - Items', ln=1)
                                pdf.image(img2, x=10, y=None, w=190)

                    # Forecast
                    if inc_forecast:
                        img3 = prophet_png(model_pedidos, forecast_pedidos, 'Forecast - Pedidos (con bandas)')
                        img4 = prophet_png(model_items, forecast_items, 'Forecast - Items (con bandas)')
                        pdf.add_page()
                        pdf.set_font('Helvetica', 'B', 14)
                        pdf.cell(0, 8, 'Forecast - Pedidos (con bandas)', ln=1)
                        pdf.image(img3, x=10, y=None, w=190)
                        pdf.ln(4)
                        pdf.set_font('Helvetica', 'B', 14)
                        pdf.cell(0, 8, 'Forecast - Items (con bandas)', ln=1)
                        pdf.image(img4, x=10, y=None, w=190)

                        # Tablas compactas (últimos días predichos)
                        def tabla_compacta(pdf, df_tab, titulo):
                            pdf.ln(2)
                            pdf.set_font('Helvetica', 'B', 12)
                            pdf.cell(0, 8, titulo, ln=1)
                            pdf.set_font('Helvetica', size=10)
                            sub = df_tab.tail(min(10, len(df_tab)))
                            headers = list(sub.columns)
                            widths = [40, 45, 45, 45]
                            for h, w in zip(headers, widths):
                                pdf.cell(w, 8, str(h), border=1)
                            pdf.ln(8)
                            for _, row in sub.iterrows():
                                pdf.cell(widths[0], 8, str(row[headers[0]])[:20], border=1)
                                pdf.cell(widths[1], 8, f"{row[headers[1]]:.2f}", border=1)
                                pdf.cell(widths[2], 8, f"{row[headers[2]]:.2f}", border=1)
                                pdf.cell(widths[3], 8, f"{row[headers[3]]:.2f}", border=1)
                                pdf.ln(8)

                        pdf.add_page()
                        tabla_compacta(pdf, tabla_ped, 'Tabla - Forecast Pedidos (últimos días)')
                        tabla_compacta(pdf, tabla_items, 'Tabla - Forecast Items (últimos días)')

                    # Productividad
                    if inc_productividad:
                        pdf.add_page()
                        pdf.set_font('Helvetica', 'B', 14)
                        pdf.cell(0, 8, 'Productividad - Picking (Top)', ln=1)
                        if prod_picker is not None and len(prod_picker) > 0:
                            top_pick = prod_picker.sort_values('items_por_hora', ascending=False).head(15)
                            pdf.set_font('Helvetica', size=10)
                            headers = list(top_pick.columns)
                            widths = [35, 18, 25, 22, 30, 30, 30]
                            for h, w in zip(headers, widths):
                                pdf.cell(w, 8, str(h)[:15], border=1)
                            pdf.ln(8)
                            for _, row in top_pick.iterrows():
                                pdf.cell(widths[0], 8, str(row[headers[0]])[:15], border=1)
                                pdf.cell(widths[1], 8, str(int(row[headers[1]])), border=1)
                                pdf.cell(widths[2], 8, str(int(row[headers[2]])), border=1)
                                pdf.cell(widths[3], 8, f"{row[headers[3]]:.2f}", border=1)
                                pdf.cell(widths[4], 8, f"{row[headers[4]]:.2f}", border=1)
                                pdf.cell(widths[5], 8, f"{row[headers[5]]:.2f}", border=1)
                                pdf.cell(widths[6], 8, f"{row[headers[6]]:.1f}", border=1)
                                pdf.ln(8)
                        else:
                            pdf.set_font('Helvetica', size=11)
                            pdf.cell(0, 8, 'No hay datos de picking.', ln=1)

                        pdf.ln(4)
                        pdf.set_font('Helvetica', 'B', 14)
                        pdf.cell(0, 8, 'Productividad - Delivery (Top)', ln=1)
                        if prod_driver is not None and len(prod_driver) > 0:
                            top_drv = prod_driver.sort_values('ordenes_por_hora', ascending=False).head(15)
                            pdf.set_font('Helvetica', size=10)
                            headers = list(top_drv.columns)
                            widths = [45, 20, 25, 30, 30]
                            for h, w in zip(headers, widths):
                                pdf.cell(w, 8, str(h)[:18], border=1)
                            pdf.ln(8)
                            for _, row in top_drv.iterrows():
                                pdf.cell(widths[0], 8, str(row[headers[0]])[:18], border=1)
                                pdf.cell(widths[1], 8, str(int(row[headers[1]])), border=1)
                                pdf.cell(widths[2], 8, f"{row[headers[2]]:.2f}", border=1)
                                pdf.cell(widths[3], 8, f"{row[headers[3]]:.2f}", border=1)
                                pdf.cell(widths[4], 8, f"{row[headers[4]]:.1f}", border=1)
                                pdf.ln(8)
                        else:
                            pdf.set_font('Helvetica', size=11)
                            pdf.cell(0, 8, 'No hay datos de delivery.', ln=1)

                    # Entrega del archivo
                    out = pdf.output(dest='S')
                    pdf_bytes = bytes(out) if isinstance(out, bytearray) else out
                    file_name = f"reporte_{tienda_seleccionada}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                    st.success('Reporte PDF generado.')
                    st.download_button('Descargar reporte PDF', data=pdf_bytes, file_name=file_name, mime='application/pdf')

            # === BLOQUE 8: Descarga CSVs consolidados ===
            st.subheader("Descargas de datos (CSV)")
            if st.button("Descargar CSVs consolidados (ZIP)"):
                buffer = io.BytesIO()
                with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                    # Datos procesados
                    zf.writestr(f"datos_procesados_{tienda_seleccionada}.csv", df_tienda.to_csv(index=False))
                    # Productividad
                    if prod_picker is not None:
                        zf.writestr(f"productividad_picking_{tienda_seleccionada}.csv", prod_picker.to_csv(index=False))
                    if prod_driver is not None:
                        zf.writestr(f"productividad_delivery_{tienda_seleccionada}.csv", prod_driver.to_csv(index=False))
                    # Forecasts completos
                    zf.writestr(f"forecast_pedidos_{tienda_seleccionada}.csv", forecast_pedidos.to_csv(index=False))
                    zf.writestr(f"forecast_items_{tienda_seleccionada}.csv", forecast_items.to_csv(index=False))
                buffer.seek(0)
                st.download_button("Descargar ZIP", data=buffer, file_name=f"consolidados_{tienda_seleccionada}.zip", mime="application/zip")

    except Exception as e:
        st.error(f"Error al procesar el archivo: {e}")
else:
    st.info("⬅️ Por favor carga un archivo CSV para comenzar.")


