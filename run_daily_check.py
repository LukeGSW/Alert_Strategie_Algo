# run_daily_check.py
"""
Daily strategy check + Telegram alert (HTML), robusto per:
- Colonne MultiIndex di yfinance
- Dtype non numerici / NaN
- Indici timezone-aware (normalizzati a tz-naive)
- Escaping HTML per evitare 400 Bad Request (es. "<15" / ">20")
Env vars richieste:
  TELEGRAM_BOT_TOKEN (obbl.)
  TELEGRAM_CHAT_ID   (obbl.)
Facoltative:
  TIMEZONE      (default 'Europe/Sofia')
  DASHBOARD_URL (link alla dashboard)
"""

import os
from datetime import datetime
from html import escape as html_escape

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
    """Invia messaggio Telegram (HTML). In caso di errore stampa anche la description."""
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
        if r.status_code != 200:
            # stampa la risposta di Telegram, utile per capire il motivo del 400
            print(f"Telegram API error {r.status_code}: {r.text}")
        r.raise_for_status()
        print("Messaggio Telegram inviato con successo.")
    except requests.RequestException as e:
        print(f"Errore invio Telegram: {e}")

def _tz_naive_index(df: pd.DataFrame) -> pd.DataFrame:
    """Rimuove timezone dall'indice DatetimeIndex se presente."""
    if isinstance(df.index, pd.DatetimeIndex):
        try:
            if df.index.tz is not None:
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

    if isinstance(df.columns, pd.MultiIndex):
        # Priorità: ('Close', ticker) -> 'Close' (xs) -> ('Adj Close', ticker) -> 'Adj Close'
        if ("Close", ticker) in df.columns:
            s = df[("Close", ticker)]
        elif "Close" in df.columns.get_level_values(0):
            sub = df.xs("Close", axis=1, level=0, drop_level=False)
            s = sub.iloc[:, 0]
        elif ("Adj Close", ticker) in df.columns:
            s = df[("Adj Close", ticker)]
        elif "Adj Close" in df.columns.get_level_values(0):
            sub = df.xs("Adj Close", axis=1, level=0, drop_level=False)
            s = sub.iloc[:, 0]
        else:
            s = df.droplevel(list(range(df.columns.nlevels - 1)), axis=1).select_dtypes("number").iloc[:, 0]
    else:
        if "Close" in df.columns:
            s = df["Close"]
        elif "Adj Close" in df.columns:
            s = df["Adj Close"]
        else:
            s = df.select_dtypes("number").iloc[:, 0]

    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]

    s = pd.to_numeric(s, errors="coerce").dropna()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.astype(float)

def _pct(a: float, b: float) -> float:
    """Variazione % (a/b - 1)*100 in modo sicuro."""
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

    # Blocchi testuali in plain text → escape HTML → incapsulati in <pre>
    values_lines = [
        f"SPX     : {_fmt_num(spx_price)}  ({'' if np.isnan(spx_dod) else f'{spx_dod:+.2f}%'} vs ieri)",
        f"VIX     : {_fmt_num(vix_price)}  ({'' if np.isnan(vix_dod) else f'{vix_dod:+.2f}%'} vs ieri)",
        f"SMA90   : {_fmt_num(sma90)}",
        f"SMA125  : {_fmt_num(sma125)}",
        f"SMA150  : {_fmt_num(sma150)}",
    ]
    values_block = "<pre>" + html_escape("\n".join(values_lines)) + "</pre>"

    strat_lines = []
    for name, rule, is_on in strategies:
        chip = "✅ ATTIVA " if is_on else "❌ NON ATTIVA"
        margins = []
        # NB: qui usiamo simboli "<" ">" ma ESCAPPIAMO l'intero blocco <pre/>
        if "SMA90"  in rule: margins.append(f"ΔSPX/SMA90  {_pct(spx_price, sma90):+.2f}%")
        if "SMA125" in rule: margins.append(f"ΔSPX/SMA125 {_pct(spx_price, sma125):+.2f}%")
        if "SMA150" in rule: margins.append(f"ΔSPX/SMA150 {_pct(spx_price, sma150):+.2f}%")
        if "VIX<15"  in rule: margins.append(f"VIX {_fmt_num(vix_price)} (<15)")
        if "VIX<20"  in rule: margins.append(f"VIX {_fmt_num(vix_price)} (<20)")
        if "VIX>20"  in rule: margins.append(f"VIX {_fmt_num(vix_price)} (>20)")
        strat_lines.append(f"{name:<14} {chip}  {' | '.join(margins)}")
    strategies_block = "<pre>" + html_escape("\n".join(strat_lines)) + "</pre>"

    # Messaggio finale
    parts = [
        "<b>🔔 Report Strategie — Kriterion Quant</b>",
        f"<i>{now_local} ({TIMEZONE})</i>",
        "",
        "<b>Valori attuali</b>",
        values_block,
        f"<b>Regime:</b> {regime}",
        "",
        "<b>Stato Strategie</b>",
        strategies_block,
    ]
    if DASHBOARD_URL:
        parts.append(f'🔗 <a href="{html_escape(DASHBOARD_URL)}">Apri dashboard</a>')

    send_telegram_message("\n".join(parts))

# ─────────────────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    check_strategies_and_alert()
