# app.py
import os
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st

# ——————————————————————————————————————————————————————————
# CONFIG
# ——————————————————————————————————————————————————————————
st.set_page_config(
    page_title="Kriterion Quant — Dashboard SPX/VIX",
    page_icon="📈",
    layout="wide"
)

# Piccolo tema CSS per card & badge
st.markdown("""
<style>
/* metric cards */
.block-container {padding-top: 1rem; padding-bottom: 2rem;}
.kq-badge {display:inline-block; padding:.15rem .5rem; border-radius:999px; font-size:.8rem; font-weight:600;}
.kq-on {background:#DCFCE7; color:#14532D;}   /* green */
.kq-neutral {background:#FEF9C3; color:#713F12;} /* amber */
.kq-off {background:#FFE4E6; color:#881337;}  /* rose */

/* status chips in table */
.kq-chip {padding:.15rem .5rem; border-radius:999px; font-weight:600;}
.kq-chip-on {background:#22c55e26; color:#166534;}
.kq-chip-off {background:#ef444426; color:#7f1d1d;}
</style>
""", unsafe_allow_html=True)

st.title("📈 Dashboard di Analisi SPX/VIX — Kriterion Quant")

# ——————————————————————————————————————————————————————————
# SIDEBAR
# ——————————————————————————————————————————————————————————
with st.sidebar:
    st.header("⚙️ Controlli")
    period = st.selectbox(
        "Periodo storico",
        options=["6mo", "1y", "2y", "5y", "10y", "max"],
        index=2
    )
    show_ranges = st.checkbox("Mostra range slider e bottoni periodo", value=True)
    st.divider()
    st.caption("Dati da Yahoo Finance, auto-adjusted. Cache 10 minuti.")

# ——————————————————————————————————————————————————————————
# DATA
# ——————————————————————————————————————————————————————————
@st.cache_data(ttl=600, show_spinner=False)
def load_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=True)
    # Normalizza colonne attese
    if not df.empty:
        df = df[["Close"]].rename(columns={"Close": "Close"}).copy()
    return df

spx = load_data("^GSPC", period)
vix = load_data("^VIX", period)

if spx.empty or vix.empty:
    st.error("Errore nel caricamento dei dati da Yahoo Finance. Riprova più tardi.")
    st.stop()

# Indicatori
for w in (90, 125, 150):
    spx[f"SMA{w}"] = spx["Close"].rolling(window=w).mean()

# Valori latest (cast a float per robustezza)
latest_spx = float(spx["Close"].iloc[-1])
prev_spx = float(spx["Close"].iloc[-2]) if len(spx) > 1 else np.nan
latest_vix = float(vix["Close"].iloc[-1])
prev_vix = float(vix["Close"].iloc[-2]) if len(vix) > 1 else np.nan
latest_sma90  = float(spx["SMA90"].iloc[-1])
latest_sma125 = float(spx["SMA125"].iloc[-1])
latest_sma150 = float(spx["SMA150"].iloc[-1])

def fmt(x, nd=2):
    if x is None or np.isnan(x):
        return "—"
    return f"{x:,.{nd}f}"

def pct(a, b):
    if b is None or np.isnan(b) or b == 0:
        return np.nan
    return (a/b - 1.0) * 100.0

# Delta % day-over-day
spx_dod = pct(latest_spx, prev_spx)
vix_dod = pct(latest_vix, prev_vix)

# Distance to SMAs
d_spx_sma90  = pct(latest_spx, latest_sma90)
d_spx_sma125 = pct(latest_spx, latest_sma125)
d_spx_sma150 = pct(latest_spx, latest_sma150)

# Regime sintetico
# Risk-ON: SPX > SMA125 e VIX < 20
# Risk-OFF: SPX < SMA150 o VIX > 20
# altrimenti Neutral
if (latest_spx > latest_sma125) and (latest_vix < 20):
    regime_label = '<span class="kq-badge kq-on">RISK-ON</span>'
elif (latest_spx < latest_sma150) or (latest_vix > 20):
    regime_label = '<span class="kq-badge kq-off">RISK-OFF</span>'
else:
    regime_label = '<span class="kq-badge kq-neutral">NEUTRAL</span>'

st.caption(f"Aggiornato al: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.markdown(f"**Regime corrente:** {regime_label}", unsafe_allow_html=True)

# ——————————————————————————————————————————————————————————
# KPI
# ——————————————————————————————————————————————————————————
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.metric("SPX", fmt(latest_spx), delta=(f"{spx_dod:+.2f}%" if not np.isnan(spx_dod) else None))
with k2:
    st.metric("VIX", fmt(latest_vix), delta=(f"{vix_dod:+.2f}%" if not np.isnan(vix_dod) else None))
with k3:
    st.metric("SPX vs SMA125", f"{d_spx_sma125:.2f}%" if not np.isnan(d_spx_sma125) else "—")
with k4:
    st.metric("SPX vs SMA90",  f"{d_spx_sma90:.2f}%" if not np.isnan(d_spx_sma90) else "—")
with k5:
    st.metric("SPX vs SMA150", f"{d_spx_sma150:.2f}%" if not np.isnan(d_spx_sma150) else "—")

st.divider()

# ——————————————————————————————————————————————————————————
# CHARTS
# ——————————————————————————————————————————————————————————
c1, c2 = st.columns([3, 2], gap="large")

with c1:
    st.subheader("SPX con SMA (90/125/150)")
    fig_spx = go.Figure()
    fig_spx.add_trace(go.Scatter(x=spx.index, y=spx["Close"], name="SPX Close", mode="lines", line=dict(width=2)))
    fig_spx.add_trace(go.Scatter(x=spx.index, y=spx["SMA90"], name="SMA 90", mode="lines", line=dict(dash="dot")))
    fig_spx.add_trace(go.Scatter(x=spx.index, y=spx["SMA125"], name="SMA 125", mode="lines", line=dict(dash="dot")))
    fig_spx.add_trace(go.Scatter(x=spx.index, y=spx["SMA150"], name="SMA 150", mode="lines", line=dict(dash="dot")))
    fig_spx.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(rangeslider=dict(visible=show_ranges), rangebreaks=[dict(bounds=["sat", "mon"])]),
        yaxis_title="Prezzo",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    if show_ranges:
        fig_spx.update_xaxes(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )
    st.plotly_chart(fig_spx, use_container_width=True)

with c2:
    st.subheader("VIX con soglie 15/20")
    fig_vix = go.Figure()
    fig_vix.add_trace(go.Scatter(x=vix.index, y=vix["Close"], name="VIX Close", mode="lines"))
    # zone 15-20
    fig_vix.add_hrect(y0=15, y1=20, line_width=0, fillcolor="orange", opacity=0.08)
    fig_vix.add_hline(y=15, line_dash="dash", annotation_text="15")
    fig_vix.add_hline(y=20, line_dash="dash", annotation_text="20")
    fig_vix.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(rangeslider=dict(visible=show_ranges), rangebreaks=[dict(bounds=["sat", "mon"])]),
        yaxis_title="Indice",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    st.plotly_chart(fig_vix, use_container_width=True)

# ——————————————————————————————————————————————————————————
# STRATEGIE: stato + motivazione sintetica
# ——————————————————————————————————————————————————————————
def strategy_rows(spx_price, vix_price, sma90, sma125, sma150):
    rules = {
        "M2K SHORT":  ("SPX>SMA90 & VIX<15",  (spx_price > sma90)  and (vix_price < 15)),
        "MES SHORT":  ("SPX>SMA125 & VIX<15", (spx_price > sma125) and (vix_price < 15)),
        "MNQ SHORT":  ("SPX<SMA150 & VIX>20", (spx_price < sma150) and (vix_price > 20)),
        "DVO LONG":   ("SPX>SMA125 & VIX<20", (spx_price > sma125) and (vix_price < 20)),
        "KeyCandle LONG": ("SPX>SMA125 & VIX<20", (spx_price > sma125) and (vix_price < 20)),
        "Z-SCORE LONG":   ("SPX>SMA125 & VIX<20", (spx_price > sma125) and (vix_price < 20)),
    }
    rows = []
    for name, (cond, active) in rules.items():
        # margini (distanze utili al controllo condizione)
        margins = []
        if "SMA90" in cond:
            margins.append(f"Δ(SPX/SMA90) {pct(spx_price, sma90):+.2f}%")
        if "SMA125" in cond:
            margins.append(f"Δ(SPX/SMA125) {pct(spx_price, sma125):+.2f}%")
        if "SMA150" in cond:
            margins.append(f"Δ(SPX/SMA150) {pct(spx_price, sma150):+.2f}%")
        if "VIX<15" in cond:
            margins.append(f"VIX {vix_price:.2f} (<15)")
        if "VIX<20" in cond:
            margins.append(f"VIX {vix_price:.2f} (<20)")
        if "VIX>20" in cond:
            margins.append(f"VIX {vix_price:.2f} (>20)")
        rows.append({
            "Strategia": name,
            "Regola": cond,
            "Stato": "🟢 ATTIVA" if active else "🔴 NON ATTIVA",
            "Margini": " | ".join(margins)
        })
    return pd.DataFrame(rows)

st.subheader("📊 Stato Attuale Strategie")
df_strat = strategy_rows(latest_spx, latest_vix, latest_sma90, latest_sma125, latest_sma150)

# Piccola formattazione: colonne strette dove serve
st.dataframe(
    df_strat,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Strategia": st.column_config.TextColumn(width="small"),
        "Regola": st.column_config.TextColumn(width="medium"),
        "Stato": st.column_config.TextColumn(width="small"),
        "Margini": st.column_config.TextColumn(width="large"),
    }
)

st.caption("Nota: logiche dei segnali semplificate per il monitoraggio; l’esecuzione reale segue i sistemi proprietari.")
