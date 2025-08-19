# run_daily_check.py
import os
import yfinance as yf
import pandas as pd
import requests
from datetime import datetime
import pytz

# --- CONFIGURAZIONE ---
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
SPX_TICKER = '^GSPC'
VIX_TICKER = '^VIX'

# --- FUNZIONE PER INVIO MESSAGGIO TELEGRAM (invariata) ---
def send_telegram_message(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("ERRORE: Le variabili d'ambiente TELEGRAM_BOT_TOKEN e TELEGRAM_CHAT_ID non sono state impostate.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'Markdown'}
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print("Messaggio Telegram inviato con successo.")
    except requests.exceptions.RequestException as e:
        print(f"Errore durante l'invio del messaggio Telegram: {e}")

# --- FUNZIONE PRINCIPALE DI ANALISI E ALERT ---
def check_strategies_and_alert():
    print("Avvio del controllo giornaliero delle strategie...")

    # --- SEZIONE MODIFICATA ---
    # Scarica i dati specificando auto_adjust=True per rimuovere i warning
    spx_data = yf.download(SPX_TICKER, period='200d', interval='1d', auto_adjust=True)
    vix_data = yf.download(VIX_TICKER, period='10d', interval='1d', auto_adjust=True)

    if spx_data.empty or vix_data.empty:
        send_telegram_message("⚠️ *Errore Dati*: Impossibile scaricare i dati per SPX o VIX.")
        return

    # Estrai i valori più recenti come scalari usando .values[-1]
    spx_price = spx_data['Close'].values[-1]
    vix_price = vix_data['Close'].values[-1]

    # Calcola le SMA e estrai i valori come scalari
    sma90 = spx_data['Close'].rolling(window=90).mean().values[-1]
    sma125 = spx_data['Close'].rolling(window=125).mean().values[-1]
    sma150 = spx_data['Close'].rolling(window=150).mean().values[-1]
    # --- FINE SEZIONE MODIFICATA ---

    # Valutazione strategie (ora funziona correttamente)
    strategies = {
        "M2K SHORT": (spx_price > sma90 and vix_price < 15),
        "MES SHORT": (spx_price > sma125 and vix_price < 15),
        "MNQ SHORT": (spx_price < sma150 and vix_price > 20),
        "DVO LONG": (spx_price > sma125 and vix_price < 20),
        "KeyCandle LONG": (spx_price > sma125 and vix_price < 20),
        "Z-SCORE LONG": (spx_price > sma125 and vix_price < 20)
    }

    # Composizione del messaggio (invariata)
    sofia_tz = pytz.timezone('Europe/Sofia')
    now_sofia = datetime.now(sofia_tz).strftime('%Y-%m-%d %H:%M:%S')

    message = f"🔔 *Report Strategie Kriterion Quant*\n"
    message += f"_{now_sofia} (Ora di Sofia)_\n\n"
    message += f"*Valori Attuali:*\n"
    message += f"  • *SPX*: `{spx_price:,.2f}`\n"
    message += f"  • *VIX*: `{vix_price:,.2f}`\n"
    message += f"  • *SMA90*: `{sma90:,.2f}`\n"
    message += f"  • *SMA125*: `{sma125:,.2f}`\n"
    message += f"  • *SMA150*: `{sma150:,.2f}`\n\n"
    message += "*Stato Strategie:*\n"
    message += "---------------------\n"

    for name, is_active in strategies.items():
        status_icon = "✅ ATTIVA" if is_active else "❌ NON ATTIVA"
        message += f"*{name}*: {status_icon}\n"

    send_telegram_message(message)

if __name__ == "__main__":
    check_strategies_and_alert()
