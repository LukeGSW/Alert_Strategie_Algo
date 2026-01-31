# app.py
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
/* Riduci lo spazio tra i subplot di Plotly */
.js-plotly-plot .plotly .main-svg {
    margin-top: -20px !important;
    margin-bottom: -20px !important;
}
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
        index=2 # Default su "2y"
    )
    show_ranges = st.checkbox("Mostra range slider (Panoramica)", value=True)
    show_debug = st.checkbox("Mostra diagnostica (debug)", value=False)
    st.caption("Dati Yahoo Finance (auto-adjust). Cache 10 minuti.")

# ─────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner="Caricamento dati Yahoo Finance...")
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
            sub = df.xs("Close", axis=1, level=0, drop_level=False)
            close = sub.iloc[:, 0] if sub.shape[1] == 1 else sub.mean(axis=1)
        elif ("Adj Close", ticker) in df.columns:
            close = df[("Adj Close", ticker)]
        else:
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

    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    close = pd.to_numeric(close, errors="coerce")
    out = pd.DataFrame({"Close": close}).dropna()
    out = out[~out.index.duplicated(keep="last")].sort_index()

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
# TABS
# ─────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Panoramica", "📊 Stato Attuale", "📈 Analisi Storica"])


# === TAB 1: PANORAMICA (GRAFICI ESISTENTI) ===
with tab1:
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
    st.plotly_chart(fig_spx, width='stretch') 

    st.divider()

    st.subheader("VIX con soglie 15 / 20")
    fig_vix = go.Figure()
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
    st.plotly_chart(fig_vix, width='stretch')


# === TAB 2: STATO ATTUALE (TABELLA AGGIORNATA) ===
with tab2:
    def strategy_rows(spx_price, vix_price, sma90, sma125, sma150) -> pd.DataFrame:
        # Condizioni ausiliarie per MGC ZScore
        # Active se Combined Weight > 1 (SPX Bull/Neutral AND VIX Low/Mid)
        mgc_spx_ok = spx_price >= (sma125 * 0.98) # NON Bear
        mgc_vix_ok = vix_price < 20               # NON High
        mgc_active = mgc_spx_ok and mgc_vix_ok

        # Condizioni ausiliarie per MNQ TrendFollowing
        # Active se Combined Weight > 1 (SPX Bull AND VIX Mid/High)
        mnq_spx_ok = spx_price > (sma125 * 1.02)
        mnq_vix_ok = vix_price >= 15
        mnq_active = mnq_spx_ok and mnq_vix_ok

        # Condizioni ausiliarie per Bias Intraweek Correlation
        # Active se Combined Weight > 1 (SPX Bull AND VIX Low/Mid)
        bias_spx_ok = spx_price > (sma125 * 1.02)  # BULL
        bias_vix_ok = vix_price < 20               # LOW o MID
        bias_active = bias_spx_ok and bias_vix_ok

        # Condizioni ausiliarie per Donchian Correlation
        # Active se Combined Weight > 1 (SPX Bull AND VIX Low/Mid)
        donch_spx_ok = spx_price > (sma125 * 1.02)  # BULL
        donch_vix_ok = vix_price < 20               # LOW o MID
        donch_active = donch_spx_ok and donch_vix_ok

        # Condizioni ausiliarie per MYM Sushi
        # Active se Combined Weight > 1 (SPX Bull AND VIX Mid)
        mym_spx_ok = spx_price > (sma125 * 1.02)              # BULL
        mym_vix_ok = (vix_price >= 15) and (vix_price < 20)   # MID
        mym_active = mym_spx_ok and mym_vix_ok

        rules = {
            "M2K SHORT":           ("SPX>SMA90 & VIX<15",  (spx_price > sma90)  and (vix_price < 15)),
            "MES SHORT":           ("SPX>SMA125 & VIX<15", (spx_price > sma125) and (vix_price < 15)),
            "MNQ SHORT":           ("SPX<SMA150 & VIX>20", (spx_price < sma150) and (vix_price > 20)),
            "ZScoreCorr LONG":     ("SPX>SMA125 & VIX<20", (spx_price > sma125) and (vix_price < 20)),
            "MGC ZScore":          ("SPX>=125MA-2% & VIX<20", mgc_active),
            "MNQ TrendFoll":       ("SPX>125MA+2% & VIX>=15", mnq_active),
            "Bias Intraweek":      ("SPX>125MA+2% & VIX<20", bias_active),
            "Donchian Corr":       ("SPX>125MA+2% & VIX<20", donch_active),
            "MYM Sushi":           ("SPX>125MA+2% & 15<=VIX<20", mym_active),
        }
        
        def _pct(a, b):
            return np.nan if b is None or (isinstance(b, float) and (np.isnan(b) or b == 0.0)) else (a/b - 1.0) * 100.0
            
        rows = []
        for name, (cond, active) in rules.items():
            margins = []
            if "SMA90"  in cond: margins.append(f"ΔSPX90 {_pct(spx_price, sma90):+.2f}%")
            if "SMA125" in cond or "125MA" in cond: margins.append(f"ΔSPX125 {_pct(spx_price, sma125):+.2f}%")
            if "SMA150" in cond: margins.append(f"ΔSPX150 {_pct(spx_price, sma150):+.2f}%")
            if "VIX" in cond:
                margins.append(f"VIX {vix_price:.2f}")

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
        width='stretch',
        hide_index=True,
        column_config={
            "Strategia": st.column_config.TextColumn(width="small"),
            "Regola": st.column_config.TextColumn(width="medium"),
            "Stato": st.column_config.TextColumn(width="small"),
            "Margini": st.column_config.TextColumn(width="large"),
        }
    )
    st.caption("Nota: logiche dei segnali semplificate per il monitoraggio; l'esecuzione reale segue i sistemi proprietari.")


# === TAB 3: ANALISI STORICA (AGGIORNATA) ===
with tab3:
    st.subheader("Analisi Storica Regimi vs SPX")
    st.caption("Grafico SPX (in alto) con lo stato storico (Attivo=1 / Non Attivo=0) delle strategie nei subchart sottostanti. I grafici sono sincronizzati: zoomando o spostando l'asse temporale, tutti i grafici si adatteranno.")

    # 1. Unisci dati storici SPX (con SMA) e VIX
    vix_close_series = vix["Close"].rename("VIX_Close")
    
    # ffill per coprire i buchi
    df_storico = spx.join(vix_close_series, how='left').ffill().dropna()

    # 2. Calcola le regole storiche e converti True/False in 1/0
    df_storico["STATUS_M2K"] = ((df_storico["Close"] > df_storico["SMA90"]) & (df_storico["VIX_Close"] < 15)).astype(int)
    df_storico["STATUS_MES"] = ((df_storico["Close"] > df_storico["SMA125"]) & (df_storico["VIX_Close"] < 15)).astype(int)
    df_storico["STATUS_MNQ"] = ((df_storico["Close"] < df_storico["SMA150"]) & (df_storico["VIX_Close"] > 20)).astype(int)
    df_storico["STATUS_ZSCORE"] = ((df_storico["Close"] > df_storico["SMA125"]) & (df_storico["VIX_Close"] < 20)).astype(int)

    # MGC ZScore (SMA125 +/- 2%)
    cond_mgc_spx = df_storico["Close"] >= (df_storico["SMA125"] * 0.98)
    cond_mgc_vix = df_storico["VIX_Close"] < 20
    df_storico["STATUS_MGC_ZSCORE"] = (cond_mgc_spx & cond_mgc_vix).astype(int)

    # MNQ TrendFollowing (SMA125 + 2%)
    cond_mnq_spx = df_storico["Close"] > (df_storico["SMA125"] * 1.02)
    cond_mnq_vix = df_storico["VIX_Close"] >= 15
    df_storico["STATUS_MNQ_TREND"] = (cond_mnq_spx & cond_mnq_vix).astype(int)

    # Bias Intraweek Correlation (SMA125 + 2% AND VIX < 20)
    cond_bias_spx = df_storico["Close"] > (df_storico["SMA125"] * 1.02)
    cond_bias_vix = df_storico["VIX_Close"] < 20
    df_storico["STATUS_BIAS_INTRAWEEK"] = (cond_bias_spx & cond_bias_vix).astype(int)

    # Donchian Correlation (SMA125 + 2% AND VIX < 20)
    cond_donch_spx = df_storico["Close"] > (df_storico["SMA125"] * 1.02)
    cond_donch_vix = df_storico["VIX_Close"] < 20
    df_storico["STATUS_DONCHIAN"] = (cond_donch_spx & cond_donch_vix).astype(int)

    # MYM Sushi (SMA125 + 2% AND 15 <= VIX < 20)
    cond_mym_spx = df_storico["Close"] > (df_storico["SMA125"] * 1.02)
    cond_mym_vix = (df_storico["VIX_Close"] >= 15) & (df_storico["VIX_Close"] < 20)
    df_storico["STATUS_MYM_SUSHI"] = (cond_mym_spx & cond_mym_vix).astype(int)

    # 3. Crea la griglia di subplot (10 righe)
    fig_storico = make_subplots(
        rows=10,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.012,
        row_heights=[0.25, 0.083, 0.083, 0.083, 0.083, 0.083, 0.083, 0.083, 0.083, 0.083],
        subplot_titles=(
            "SPX Close", 
            "M2K SHORT (SPX>90 & VIX<15)", 
            "MES SHORT (SPX>125 & VIX<15)", 
            "MNQ SHORT (SPX<150 & VIX>20)", 
            "ZScoreCorr LONG (SPX>125 & VIX<20)",
            "MGC ZScore (SPX>=125MA-2% & VIX<20)",
            "MNQ TrendFoll (SPX>125MA+2% & VIX>=15)",
            "Bias Intraweek (SPX>125MA+2% & VIX<20)",
            "Donchian Corr (SPX>125MA+2% & VIX<20)",
            "MYM Sushi (SPX>125MA+2% & 15<=VIX<20)"
        )
    )

    # 4. Aggiungi traccia SPX (Riga 1)
    fig_storico.add_trace(go.Scatter(
        x=df_storico.index, 
        y=df_storico["Close"], 
        name="SPX Close",
        mode="lines", 
        line=dict(color="#2E86C1", width=2)
    ), row=1, col=1)

    # 5. Aggiungi tracce Strategie (Righe 2-10)
    
    # M2K SHORT (Rosso)
    fig_storico.add_trace(go.Scatter(
        x=df_storico.index, y=df_storico["STATUS_M2K"], name="M2K SHORT",
        fill='tozeroy', mode='lines', line=dict(color="#E74C3C", width=1)
    ), row=2, col=1)
    
    # MES SHORT (Arancio)
    fig_storico.add_trace(go.Scatter(
        x=df_storico.index, y=df_storico["STATUS_MES"], name="MES SHORT",
        fill='tozeroy', mode='lines', line=dict(color="#F39C12", width=1)
    ), row=3, col=1)
    
    # MNQ SHORT (Viola)
    fig_storico.add_trace(go.Scatter(
        x=df_storico.index, y=df_storico["STATUS_MNQ"], name="MNQ SHORT",
        fill='tozeroy', mode='lines', line=dict(color="#8E44AD", width=1)
    ), row=4, col=1)
    
    # ZScoreCorr LONG (Blu)
    fig_storico.add_trace(go.Scatter(
        x=df_storico.index, y=df_storico["STATUS_ZSCORE"], name="ZScoreCorr LONG",
        fill='tozeroy', mode='lines', line=dict(color="#2980B9", width=1)
    ), row=5, col=1)

    # MGC ZScore (Viola scuro) - RIGA 6
    fig_storico.add_trace(go.Scatter(
        x=df_storico.index, y=df_storico["STATUS_MGC_ZSCORE"], name="MGC ZScore",
        fill='tozeroy', mode='lines', line=dict(color="#7D3C98", width=1)
    ), row=6, col=1)

    # MNQ TrendFollowing (Teal) - RIGA 7
    fig_storico.add_trace(go.Scatter(
        x=df_storico.index, y=df_storico["STATUS_MNQ_TREND"], name="MNQ Trend",
        fill='tozeroy', mode='lines', line=dict(color="#117864", width=1)
    ), row=7, col=1)

    # Bias Intraweek (Verde lime) - RIGA 8
    fig_storico.add_trace(go.Scatter(
        x=df_storico.index, y=df_storico["STATUS_BIAS_INTRAWEEK"], name="Bias Intraweek",
        fill='tozeroy', mode='lines', line=dict(color="#27AE60", width=1)
    ), row=8, col=1)

    # Donchian Correlation (Ciano) - RIGA 9
    fig_storico.add_trace(go.Scatter(
        x=df_storico.index, y=df_storico["STATUS_DONCHIAN"], name="Donchian Corr",
        fill='tozeroy', mode='lines', line=dict(color="#17A2B8", width=1)
    ), row=9, col=1)

    # MYM Sushi (Rosa/Magenta) - RIGA 10
    fig_storico.add_trace(go.Scatter(
        x=df_storico.index, y=df_storico["STATUS_MYM_SUSHI"], name="MYM Sushi",
        fill='tozeroy', mode='lines', line=dict(color="#E91E63", width=1)
    ), row=10, col=1)

    # 6. Configura Layout
    fig_storico.update_layout(
        height=1300,
        hovermode="x unified",
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(rangeslider=dict(visible=False)),
        xaxis10=dict(rangeslider=dict(visible=True, thickness=0.08))
    )

    # Nascondi etichette assi X per i grafici 1-9
    for i in range(1, 10):
        fig_storico.update_xaxes(showticklabels=False, row=i, col=1)
    
    # Configura assi Y
    fig_storico.update_yaxes(title_text="Prezzo SPX", row=1, col=1)
    for i in range(2, 11):
        fig_storico.update_yaxes(
            title_text="Stato",
            range=[-0.1, 1.1], 
            showticklabels=False, 
            row=i, 
            col=1
        )
        
    # Migliora la leggibilità dei titoli dei subplot
    for annotation in fig_storico['layout']['annotations']:
        annotation['font'] = dict(size=11)
        annotation['yanchor'] = 'bottom'
        annotation['y'] = annotation['y'] + 0.01

    st.plotly_chart(fig_storico, width='stretch')


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
    if 'df_storico' in locals():
         st.write("df_storico tail (per Analisi Storica):", df_storico.tail(3))
