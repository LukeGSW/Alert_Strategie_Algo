# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
    page_title="Kriterion Quant - Dashboard Strategie SPX/VIX",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Dashboard di Analisi SPX e VIX")
st.caption(f"Dati aggiornati al: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

@st.cache_data(ttl=600)
def load_data(ticker, period='2y'):
    # --- MODIFICATO ---
    return yf.download(ticker, period=period, interval='1d', auto_adjust=True)

spx_data = load_data('^GSPC')
vix_data = load_data('^VIX')

if not spx_data.empty and not vix_data.empty:
    spx_data['SMA90'] = spx_data['Close'].rolling(window=90).mean()
    spx_data['SMA125'] = spx_data['Close'].rolling(window=125).mean()
    spx_data['SMA150'] = spx_data['Close'].rolling(window=150).mean()

    # --- SEZIONE MODIFICATA ---
    # Estrai i valori più recenti come scalari usando .values[-1]
    latest_spx = spx_data['Close'].values[-1]
    latest_vix = vix_data['Close'].values[-1]
    latest_sma90 = spx_data['SMA90'].values[-1]
    latest_sma125 = spx_data['SMA125'].values[-1]
    latest_sma150 = spx_data['SMA150'].values[-1]
    # --- FINE SEZIONE MODIFICATA ---

    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("Andamento Indice S&P 500 (SPX) e Medie Mobili")
        fig_spx = go.Figure()
        fig_spx.add_trace(go.Scatter(x=spx_data.index, y=spx_data['Close'], mode='lines', name='SPX Close', line=dict(color='blue', width=2)))
        fig_spx.add_trace(go.Scatter(x=spx_data.index, y=spx_data['SMA90'], mode='lines', name='SMA 90', line=dict(color='orange', dash='dot')))
        fig_spx.add_trace(go.Scatter(x=spx_data.index, y=spx_data['SMA125'], mode='lines', name='SMA 125', line=dict(color='purple', dash='dot')))
        fig_spx.add_trace(go.Scatter(x=spx_data.index, y=spx_data['SMA150'], mode='lines', name='SMA 150', line=dict(color='red', dash='dot')))
        fig_spx.update_layout(title_text='SPX con SMA 90, 125, 150', xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_spx, use_container_width=True)

        st.subheader("Andamento Indice di Volatilità (VIX) e Soglie Critiche")
        fig_vix = go.Figure()
        fig_vix.add_trace(go.Scatter(x=vix_data.index, y=vix_data['Close'], mode='lines', name='VIX Close', line=dict(color='green')))
        fig_vix.add_hline(y=15, line_dash="dash", line_color="orange", annotation_text="Soglia 15", annotation_position="bottom right")
        fig_vix.add_hline(y=20, line_dash="dash", line_color="red", annotation_text="Soglia 20", annotation_position="top right")
        fig_vix.update_layout(title_text='VIX con soglie 15 e 20', xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_vix, use_container_width=True)

    with col2:
        st.subheader("Valori di Mercato")
        st.metric(label="SPX", value=f"{latest_spx:,.2f}")
        st.metric(label="VIX", value=f"{latest_vix:,.2f}")
        st.metric(label="SMA 90", value=f"{latest_sma90:,.2f}")
        st.metric(label="SMA 125", value=f"{latest_sma125:,.2f}")
        st.metric(label="SMA 150", value=f"{latest_sma150:,.2f}")
        
        st.divider()

        st.subheader("Stato Attuale Strategie")

        def display_strategy_status(name, is_active):
            emoji = "✅" if is_active else "❌"
            status_text = "ATTIVA" if is_active else "NON ATTIVA"
            st.markdown(f"**{name}**: {status_text} {emoji}")

        display_strategy_status("M2K SHORT", latest_spx > latest_sma90 and latest_vix < 15)
        display_strategy_status("MES SHORT", latest_spx > latest_sma125 and latest_vix < 15)
        display_strategy_status("MNQ SHORT", latest_spx < latest_sma150 and latest_vix > 20)
        display_strategy_status("DVO LONG", latest_spx > latest_sma125 and latest_vix < 20)
        display_strategy_status("KeyCandle LONG", latest_spx > latest_sma125 and latest_vix < 20)
        display_strategy_status("Z-SCORE LONG", latest_spx > latest_sma125 and latest_vix < 20)
else:
    st.error("Errore nel caricamento dei dati da Yahoo Finance. Riprova più tardi.")
