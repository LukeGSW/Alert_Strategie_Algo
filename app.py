# app.py
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Kriterion Quant — Dashboard SPX/VIX",
    page_icon="📈",
    layout="wide",
)

# Stile badge/chips (compatibile dark mode)
st.markdown("""
<style>
.block-container {padding-top: 1rem; padding-bottom: 2rem;}
.kq-badge {display:inline-block; padding:.15rem .6rem; border-radius:999px; font-size:.8rem; font-weight:600;}
.kq-on {background:#DCFCE7; color:#14532D;}
.kq-neutral {background:#FEF9C3; color:#713F12;}
.kq-off {background:#FFE4E6; color:#881337;}
</style>
""", unsafe_allow_html=True)

st.title("📈 Dashboard di Analisi SPX/VIX — Kriterion Quant")

# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Controlli")
    period = st.selectbox(
        "Periodo storico",
        options=["6mo", "1y", "2y", "5y", "10y", "max"],
        index=2
    )
    show_ranges = st.checkbox("Mostra range slider e bottoni periodo", value=True)
    show_debug = st.checkbox("Mostra diagnostica (debug)", value=False)
    st.caption("Dati Yahoo Finance (auto-adjust). Cache 10 minuti.")

# ─────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def load_data(ticker: str, period: str) -> pd.DataFrame:
    """
    Scarica dati da yfinance e restituisce sempre un DataFrame con:
      - index: datetime tz-naive, ordinato, senza duplicati
      - colonna unica 'Close' (float)
    Gestisce sia colonne semplici sia MultiIndex (('Close', TICKER)).
    """
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=True)
    if df.empty:
        return df

    # Se MultiIndex: prova a prendere ('Close', ticker) o 'Close' alla 2° level
    if isinstance(df.columns, pd.MultiIndex):
        close = None
        if ("Close", ticker) in df.columns:
            close = df[("Close", ticker)]
        elif "Close" in df.columns.get_level_values(0):
            # se c'è una sola colonna al livello 1, squeeze
            sub = df.xs("Close", axis=1, level=0, drop_level=False)
            close = sub.iloc[:, 0] if sub.shape[1] == 1 else sub.mean(axis=1)
        elif ("Adj Close", ticker) in df.columns:
            close = df[("Adj Close", ticker)]
        else:
            # fallback: prima colonna numerica disponibile
            sub = df.droplevel(0, axis=1) if df.columns.nlevels > 1 else df
            close = sub.select_dtypes(include="number").iloc[:, 0]
    else:
        # Colonne semplici
        if "Close" in df.columns:
            close = df["Close"]
        elif "Adj Close" in df.columns:
            close = df["Adj Close"]
        else:
            close = df.select_dtypes(include="number").iloc[:, 0]

    # Assicura una Series 1-D
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    # Normalizza output
    close = pd.to_numeric(close, errors="coerce")
    out = pd.DataFrame({"Close": close}).dropna()
    out = out[~out.index.duplicated(keep="last")].sort_index()

    # Togli tz per Plotly
    try:
        out.index = out.index.tz_localize(None)
    except Exception:
        pass

    return out


spx = load_data("^GSPC", period)
vix = load_data("^VIX", period)

if spx.empty or vix.empty:
    st.error("Errore nel caricamento dei dati da Yahoo Finance. Riprova più tardi.")
    st.stop()

# SMA con min_periods=1 per avere curve sempre visibili
close_spx = spx["Close"].astype(float)
spx["SMA90"]  = close_spx.rolling(90,  min_periods=1).mean()
spx["SMA125"] = close_spx.rolling(125, min_periods=1).mean()
spx["SMA150"] = close_spx.rolling(150, min_periods=1).mean()

# Per evitare qualunque ambiguità di indicizzazione, uso numpy
spx_close_np = close_spx.to_numpy()
vix_close_np = vix["Close"].astype(float).to_numpy()

latest_spx   = float(spx_close_np[-1]) if spx_close_np.size else np.nan
prev_spx     = float(spx_close_np[-2]) if spx_close_np.size > 1 else np.nan
latest_vix   = float(vix_close_np[-1]) if vix_close_np.size else np.nan
prev_vix     = float(vix_close_np[-2]) if vix_close_np.size > 1 else np.nan

latest_sma90  = float(spx["SMA90"].iloc[-1])
latest_sma125 = float(spx["SMA125"].iloc[-1])
latest_sma150 = float(spx["SMA150"].iloc[-1])

def pct(a: float, b: float) -> float:
    if b is None or (isinstance(b, float) and (np.isnan(b) or b == 0.0)):
        return np.nan
    return (a / b - 1.0) * 100.0

def fmt(x, nd=2):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.{nd}f}"

# KPI & regime
spx_dod = pct(latest_spx, prev_spx)
vix_dod = pct(latest_vix, prev_vix)
d_spx_sma90  = pct(latest_spx, latest_sma90)
d_spx_sma125 = pct(latest_spx, latest_sma125)
d_spx_sma150 = pct(latest_spx, latest_sma150)

if (latest_spx > latest_sma125) and (latest_vix < 20):
    regime_label = '<span class="kq-badge kq-on">RISK-ON</span>'
elif (latest_spx < latest_sma150) or (latest_vix > 20):
    regime_label = '<span class="kq-badge kq-off">RISK-OFF</span>'
else:
    regime_label = '<span class="kq-badge kq-neutral">NEUTRAL</span>'

st.caption(f"Aggiornato al: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.markdown(f"**Regime corrente:** {regime_label}", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# KPI
# ─────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
with k1: st.metric("SPX", fmt(latest_spx), delta=(f"{spx_dod:+.2f}%" if not np.isnan(spx_dod) else None))
with k2: st.metric("VIX", fmt(latest_vix), delta=(f"{vix_dod:+.2f}%" if not np.isnan(vix_dod) else None))
with k3: st.metric("SPX vs SMA125", f"{d_spx_sma125:.2f}%" if not np.isnan(d_spx_sma125) else "—")
with k4: st.metric("SPX vs SMA90",  f"{d_spx_sma90:.2f}%"  if not np.isnan(d_spx_sma90)  else "—")
with k5: st.metric("SPX vs SMA150", f"{d_spx_sma150:.2f}%" if not np.isnan(d_spx_sma150) else "—")

st.divider()

# ─────────────────────────────────────────────────────────
# CHARTS — stacked (uno sopra l'altro, full width)
# ─────────────────────────────────────────────────────────

st.subheader("SPX con SMA (90 / 125 / 150)")
fig_spx = go.Figure()
fig_spx.add_trace(go.Scatter(
    x=spx.index, y=spx["Close"], name="SPX Close",
    mode="lines", line=dict(color="#2E86C1", width=2)
))
fig_spx.add_trace(go.Scatter(
    x=spx.index, y=spx["SMA90"], name="SMA 90",
    mode="lines", line=dict(color="#F39C12", dash="dot", width=1.5)
))
fig_spx.add_trace(go.Scatter(
    x=spx.index, y=spx["SMA125"], name="SMA 125",
    mode="lines", line=dict(color="#8E44AD", dash="dot", width=1.5)
))
fig_spx.add_trace(go.Scatter(
    x=spx.index, y=spx["SMA150"], name="SMA 150",
    mode="lines", line=dict(color="#E74C3C", dash="dot", width=1.5)
))
fig_spx.update_layout(
    height=420,
    margin=dict(l=10, r=10, t=30, b=10),
    hovermode="x unified",
    xaxis=dict(
        rangeslider=dict(visible=show_ranges),
        rangebreaks=[dict(bounds=["sat", "mon"])]
    ),
    yaxis_title="Prezzo",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)
if show_ranges:
    fig_spx.update_xaxes(
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ]
        )
    )
st.plotly_chart(fig_spx, use_container_width=True)

st.divider()

st.subheader("VIX con soglie 15 / 20")
fig_vix = go.Figure()
# fascia 15–20 sotto la linea
fig_vix.add_hrect(y0=15, y1=20, line_width=0, fillcolor="orange", opacity=0.08, layer="below")
fig_vix.add_trace(go.Scatter(
    x=vix.index, y=vix["Close"], name="VIX Close",
    mode="lines", line=dict(color="#16A085", width=2)
))
fig_vix.add_hline(y=15, line_dash="dash", line_color="#F39C12", annotation_text="15")
fig_vix.add_hline(y=20, line_dash="dash", line_color="#E74C3C", annotation_text="20")
fig_vix.update_layout(
    height=360,
    margin=dict(l=10, r=10, t=30, b=10),
    hovermode="x unified",
    xaxis=dict(
        rangeslider=dict(visible=show_ranges),
        rangebreaks=[dict(bounds=["sat", "mon"])]
    ),
    yaxis_title="Indice",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)
st.plotly_chart(fig_vix, use_container_width=True)

# ─────────────────────────────────────────────────────────
# STRATEGIE
# ─────────────────────────────────────────────────────────
def strategy_rows(spx_price, vix_price, sma90, sma125, sma150) -> pd.DataFrame:
    rules = {
        "M2K SHORT":      ("SPX>SMA90 & VIX<15",  (spx_price > sma90)  and (vix_price < 15)),
        "MES SHORT":      ("SPX>SMA125 & VIX<15", (spx_price > sma125) and (vix_price < 15)),
        "MNQ SHORT":      ("SPX<SMA150 & VIX>20", (spx_price < sma150) and (vix_price > 20)),
        "DVO LONG":       ("SPX>SMA125 & VIX<20", (spx_price > sma125) and (vix_price < 20)),
        "KeyCandle LONG": ("SPX>SMA125 & VIX<20", (spx_price > sma125) and (vix_price < 20)),
        "Z-SCORE LONG":   ("SPX>SMA125 & VIX<20", (spx_price > sma125) and (vix_price < 20)),
    }
    def _pct(a, b):
        return np.nan if b is None or (isinstance(b, float) and (np.isnan(b) or b == 0.0)) else (a/b - 1.0) * 100.0
    rows = []
    for name, (cond, active) in rules.items():
        margins = []
        if "SMA90"  in cond: margins.append(f"Δ(SPX/SMA90)  {_pct(spx_price, sma90):+.2f}%")
        if "SMA125" in cond: margins.append(f"Δ(SPX/SMA125) {_pct(spx_price, sma125):+.2f}%")
        if "SMA150" in cond: margins.append(f"Δ(SPX/SMA150) {_pct(spx_price, sma150):+.2f}%")
        if "VIX<15"  in cond: margins.append(f"VIX {vix_price:.2f} (<15)")
        if "VIX<20"  in cond: margins.append(f"VIX {vix_price:.2f} (<20)")
        if "VIX>20"  in cond: margins.append(f"VIX {vix_price:.2f} (>20)")
        rows.append({
            "Strategia": name,
            "Regola": cond,
            "Stato": "🟢 ATTIVA" if active else "🔴 NON ATTIVA",
            "Margini": " | ".join(margins)
        })
    return pd.DataFrame(rows)

st.subheader("📊 Stato Attuale Strategie")
df_strat = strategy_rows(latest_spx, latest_vix, latest_sma90, latest_sma125, latest_sma150)
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

# ─────────────────────────────────────────────────────────
# DEBUG (facoltativo)
# ─────────────────────────────────────────────────────────
if show_debug:
    st.divider()
    st.write("**Diagnostica**")
    st.write("SPX tail:", spx.tail(3))
    st.write("VIX tail:", vix.tail(3))
    st.write("Dtypes SPX:", spx.dtypes)
    st.write("Dtypes VIX:", vix.dtypes)
