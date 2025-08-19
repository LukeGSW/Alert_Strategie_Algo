# run_daily_check.py
"""
Daily strategy check + Telegram alert (HTML).
Robusto a:
- Colonne MultiIndex di yfinance (('Close', '^GSPC'), ecc.)
- Dtype non numerici / NaN
- Indici timezone-aware (normalizzati a tz-naive)
Config via environment:
  TELEGRAM_BOT_TOKEN (obblig.)
  TELEGRAM_CHAT_ID   (obblig.)
  TIMEZONE           (opz., default 'Europe/Sofia')
  DASHBOARD_URL      (opz., link alla dashboard Streamlit)
"""

import os
from datetime import datetime
import numpy as np
import pandas as pd
import pytz
import requests
import yfinance as yf

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")
TIMEZONE           = os.getenv("TIMEZONE", "Europe/Sofia")
DASHBOARD_URL      = os.getenv("DASHBOARD_URL", "")

SPX_TICKER = "^GSPC"
VIX_TICKER = "^VIX"

# ─────────────────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────────────────
def send_telegram_message(message_html: str) -> None:
    """Invia messaggio Telegram in HTML."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("ERRORE: TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID non impostati.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message_html,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        r = requests.post(url, json=payload, timeout=20)
        r.raise_for_status()
        print("Messaggio Telegram inviato con successo.")
    except requests.RequestException as e:
        print(f"Errore invio Telegram: {e}")

def _tz_naive_index(df: pd.DataFrame) -> pd.DataFrame:
    """Rimuove timezone dall'indice DatetimeIndex se presente."""
    if isinstance(df.index, pd.DatetimeIndex):
        try:
            if df.index.tz is not None:
                # alcune versioni richiedono tz_convert(None), altre tz_localize(None)
                try:
                    df.index = df.index.tz_convert(None)
                except Exception:
                    df.index = df.index.tz_localize(None)
        except Exception:
            pass
    return df

def load_close_series(ticker: str, period: str) -> pd.Series:
    """
    Scarica da yfinance e ritorna SEMPRE una Series float con index tz-naive e ordinato.
    Gestisce colonne MultiIndex e fallback se 'Close' non è presente.
    """
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
    if df is None or len(df) == 0:
        return pd.Series(dtype=float)

    df = _tz_naive_index(df)
    # Se MultiIndex (es. ('Close','^GSPC'))
    if isinstance(df.columns, pd.MultiIndex):
        s = None
        if ("Close", ticker) in df.columns:
            s = df[("Close", ticker)]
        elif "Close" in df.columns.get_level_values(0):
            sub = df.xs("Close", axis=1, level=0, drop_level=False)
            # se più colonne, media (o prendi la prima disponibile)
            s = sub.iloc[:, 0]
        elif ("Adj Close", ticker) in df.columns:
            s = df[("Adj Close", ticker)]
        elif "Adj Close" in df.columns.get_level_values(0):
            sub = df.xs("Adj Close", axis=1, level=0, drop_level=False)
            s = sub.iloc[:, 0]
        else:
            # fallback: prima colonna numerica
            s = df.droplevel(list(range(df.columns.nlevels - 1)), axis=1).select_dtypes("number").iloc[:, 0]
    else:
        # Colonne semplici
        if "Close" in df.columns:
            s = df["Close"]
        elif "Adj Close" in df.columns:
            s = df["Adj Close"]
        else:
            s = df.select_dtypes("number").iloc[:, 0]

    # Assicura Series 1-D numerica, pulita e ordinata
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s = pd.to_numeric(s, errors="coerce").dropna()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.astype(float)

def _pct(a: float, b: float) -> float:
    """Ritorna variazione % (a/b - 1)*100 in modo sicuro."""
    if b is None or (isinstance(b, float) and (np.isnan(b) or b == 0.0)):
        return np.nan
    return (a / b - 1.0) * 100.0

def _fmt_num(x: float, nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:,.{nd}f}"

# ─────────────────────────────────────────────────────────
# CORE
# ─────────────────────────────────────────────────────────
def check_strategies_and_alert() -> None:
    print("Avvio controllo giornaliero…")

    spx_close = load_close_series(SPX_TICKER, "220d")
    vix_close = load_close_series(VIX_TICKER, "30d")

    if spx_close.empty or vix_close.empty:
        send_telegram_message("<b>⚠️ Errore Dati</b><br/>Impossibile scaricare SPX o VIX.")
        return

    # Scalar latest/prev (via .iat per evitare FutureWarning)
    spx_price = float(spx_close.iat[-1])
    vix_price = float(vix_close.iat[-1])
    prev_spx  = float(spx_close.iat[-2]) if len(spx_close) > 1 else np.nan
    prev_vix  = float(vix_close.iat[-2]) if len(vix_close) > 1 else np.nan

    # SMA con min_periods=1 per avere valori anche su storici corti
    sma90  = float(spx_close.rolling(90,  min_periods=1).mean().iat[-1])
    sma125 = float(spx_close.rolling(125, min_periods=1).mean().iat[-1])
    sma150 = float(spx_close.rolling(150, min_periods=1).mean().iat[-1])

    # Stato strategie
    strategies = [
        ("M2K SHORT",      "SPX>SMA90 & VIX<15",   (spx_price > sma90)  and (vix_price < 15)),
        ("MES SHORT",      "SPX>SMA125 & VIX<15",  (spx_price > sma125) and (vix_price < 15)),
        ("MNQ SHORT",      "SPX<SMA150 & VIX>20",  (spx_price < sma150) and (vix_price > 20)),
        ("DVO LONG",       "SPX>SMA125 & VIX<20",  (spx_price > sma125) and (vix_price < 20)),
        ("KeyCandle LONG", "SPX>SMA125 & VIX<20",  (spx_price > sma125) and (vix_price < 20)),
        ("Z-SCORE LONG",   "SPX>SMA125 & VIX<20",  (spx_price > sma125) and (vix_price < 20)),
    ]

    # Regime sintetico
    if (spx_price > sma125) and (vix_price < 20):
        regime = "🟢 RISK-ON"
    elif (spx_price < sma150) or (vix_price > 20):
        regime = "🔴 RISK-OFF"
    else:
        regime = "🟡 NEUTRAL"

    # Delta giornalieri
    spx_dod = _pct(spx_price, prev_spx)
    vix_dod = _pct(vix_price, prev_vix)

    # Timestamp locale
    try:
        tz = pytz.timezone(TIMEZONE)
    except Exception:
        tz = pytz.timezone("UTC")
    now_local = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")

    # Messaggio HTML (compatto e leggibile in monospace)
    lines = []
    lines.append("<b>🔔 Report Strategie — Kriterion Quant</b>")
    lines.append(f"<i>{now_local} ({TIMEZONE})</i>")
    lines.append("")

    lines.append("<b>Valori attuali</b>")
    lines.append(
        "<pre>"
        f"SPX     : {_fmt_num(spx_price)}  ({'+' if not np.isnan(spx_dod) and spx_dod>=0 else ''}{'' if np.isnan(spx_dod) else f'{spx_dod:.2f}%'} vs ieri)\n"
        f"VIX     : {_fmt_num(vix_price)}  ({'+' if not np.isnan(vix_dod) and vix_dod>=0 else ''}{'' if np.isnan(vix_dod) else f'{vix_dod:.2f}%'} vs ieri)\n"
        f"SMA90   : {_fmt_num(sma90)}\n"
        f"SMA125  : {_fmt_num(sma125)}\n"
        f"SMA150  : {_fmt_num(sma150)}"
        "</pre>"
    )

    lines.append(f"<b>Regime:</b> {regime}")
    lines.append("")
    lines.append("<b>Stato Strategie</b>")
    lines.append("<pre>")
    for name, rule, is_on in strategies:
        chip = "✅ ATTIVA " if is_on else "❌ NON ATTIVA"
        margins = []
        if "SMA90"  in rule: margins.append(f"ΔSPX/SMA90  {_pct(spx_price, sma90):+.2f}%")
        if "SMA125" in rule: margins.append(f"ΔSPX/SMA125 {_pct(spx_price, sma125):+.2f}%")
        if "SMA150" in rule: margins.append(f"ΔSPX/SMA150 {_pct(spx_price, sma150):+.2f}%")
        if "VIX<15"  in rule: margins.append(f"VIX {_fmt_num(vix_price)} (<15)")
        if "VIX<20"  in rule: margins.append(f"VIX {_fmt_num(vix_price)} (<20)")
        if "VIX>20"  in rule: margins.append(f"VIX {_fmt_num(vix_price)} (>20)")
        lines.append(f"{name:<14} {chip}  {' | '.join(margins)}")
    lines.append("</pre>")

    if DASHBOARD_URL:
        lines.append(f'🔗 <a href="{DASHBOARD_URL}">Apri dashboard</a>')

    send_telegram_message("\n".join(lines))

# ─────────────────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    check_strategies_and_alert()
