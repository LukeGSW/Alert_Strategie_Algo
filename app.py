# run_daily_check.py
import os
from datetime import datetime
import numpy as np
import pandas as pd
import pytz
import requests
import yfinance as yf

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")
DASHBOARD_URL      = os.getenv("DASHBOARD_URL", "")      # opzionale
TIMEZONE           = os.getenv("TIMEZONE", "Europe/Berlin")

SPX_TICKER = "^GSPC"
VIX_TICKER = "^VIX"

def send_telegram_message(message_html: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("ERRORE: TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID non impostati.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message_html,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    try:
        r = requests.post(url, json=payload, timeout=20)
        r.raise_for_status()
        print("Messaggio Telegram inviato con successo.")
    except requests.RequestException as e:
        print(f"Errore invio Telegram: {e}")

def _pct(a, b):
    if b is None or (isinstance(b, float) and (np.isnan(b) or b == 0.0)):
        return np.nan
    return (a/b - 1.0) * 100.0

def check_strategies_and_alert():
    spx = yf.download(SPX_TICKER, period="220d", interval="1d", auto_adjust=True)[["Close"]].dropna()
    vix = yf.download(VIX_TICKER, period="30d",  interval="1d", auto_adjust=True)[["Close"]].dropna()

    if spx.empty or vix.empty:
        send_telegram_message("<b>⚠️ Errore Dati</b><br/>Impossibile scaricare SPX o VIX.")
        return

    spx_close = spx["Close"].astype(float)
    vix_close = vix["Close"].astype(float)

    spx_price = float(spx_close.iat[-1])
    vix_price = float(vix_close.iat[-1])
    prev_spx  = float(spx_close.iat[-2]) if len(spx_close) > 1 else np.nan
    prev_vix  = float(vix_close.iat[-2]) if len(vix_close) > 1 else np.nan

    sma90  = float(spx_close.rolling(90,  min_periods=1).mean().iat[-1])
    sma125 = float(spx_close.rolling(125, min_periods=1).mean().iat[-1])
    sma150 = float(spx_close.rolling(150, min_periods=1).mean().iat[-1])

    strategies = [
        ("M2K SHORT",      "SPX>SMA90 & VIX<15",   (spx_price > sma90)  and (vix_price < 15)),
        ("MES SHORT",      "SPX>SMA125 & VIX<15",  (spx_price > sma125) and (vix_price < 15)),
        ("MNQ SHORT",      "SPX<SMA150 & VIX>20",  (spx_price < sma150) and (vix_price > 20)),
        ("DVO LONG",       "SPX>SMA125 & VIX<20",  (spx_price > sma125) and (vix_price < 20)),
        ("KeyCandle LONG", "SPX>SMA125 & VIX<20",  (spx_price > sma125) and (vix_price < 20)),
        ("Z-SCORE LONG",   "SPX>SMA125 & VIX<20",  (spx_price > sma125) and (vix_price < 20)),
    ]

    if (spx_price > sma125) and (vix_price < 20):
        regime = "🟢 RISK-ON"
    elif (spx_price < sma150) or (vix_price > 20):
        regime = "🔴 RISK-OFF"
    else:
        regime = "🟡 NEUTRAL"

    spx_dod = _pct(spx_price, prev_spx)
    vix_dod = _pct(vix_price, prev_vix)

    tz = pytz.timezone(TIMEZONE)
    now_local = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append("<b>🔔 Report Strategie — Kriterion Quant</b>")
    lines.append(f"<i>{now_local} ({TIMEZONE})</i>")
    lines.append("")
    lines.append("<b>Valori attuali</b>")
    lines.append("<pre>"
                 f"SPX     : {spx_price:,.2f}  ({'+' if not np.isnan(spx_dod) and spx_dod>=0 else ''}{'' if np.isnan(spx_dod) else f'{spx_dod:.2f}%'} vs ieri)\n"
                 f"VIX     : {vix_price:,.2f}  ({'+' if not np.isnan(vix_dod) and vix_dod>=0 else ''}{'' if np.isnan(vix_dod) else f'{vix_dod:.2f}%'} vs ieri)\n"
                 f"SMA90   : {sma90:,.2f}\n"
                 f"SMA125  : {sma125:,.2f}\n"
                 f"SMA150  : {sma150:,.2f}"
                 "</pre>")
    lines.append(f"<b>Regime:</b> {regime}")
    lines.append("")
    lines.append("<b>Stato Strategie</b>")
    lines.append("<pre>")
    for name, rule, is_on in strategies:
        chip = "✅ ATTIVA " if is_on else "❌ NON ATTIVA"
        margins = []
        if "SMA90"  in rule: margins.append(f"ΔSPX/SMA90 { _pct(spx_price, sma90):+.2f}%")
        if "SMA125" in rule: margins.append(f"ΔSPX/SMA125 { _pct(spx_price, sma125):+.2f}%")
        if "SMA150" in rule: margins.append(f"ΔSPX/SMA150 { _pct(spx_price, sma150):+.2f}%")
        if "VIX<15"  in rule: margins.append(f"VIX {vix_price:,.2f} (<15)")
        if "VIX<20"  in rule: margins.append(f"VIX {vix_price:,.2f} (<20)")
        if "VIX>20"  in rule: margins.append(f"VIX {vix_price:,.2f} (>20)")
        lines.append(f"{name:<14} {chip}  {(' | '.join(margins))}")
    lines.append("</pre>")
    if DASHBOARD_URL:
        lines.append(f'🔗 <a href="{DASHBOARD_URL}">Apri dashboard</a>')

    send_telegram_message("\n".join(lines))

if __name__ == "__main__":
    check_strategies_and_alert()
