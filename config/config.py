"""
Configuration module for the Trading Signal Bot
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Telegram Configuration
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# Exchange Configuration
EXCHANGE_NAME = os.getenv('EXCHANGE_NAME', 'kraken')
EXCHANGE_API_KEY = os.getenv('EXCHANGE_API_KEY', '')
EXCHANGE_API_SECRET = os.getenv('EXCHANGE_API_SECRET', '')

# Trading Pairs to Monitor (Top cryptocurrencies by liquidity)
# Kraken uses USDT for most pairs (alta liquidez similar a Binance)
TRADING_PAIRS = [
    # Top cryptos más líquidas (formato Kraken - mezcla USDT/USD según disponibilidad)
    'BTC/USDT',    # Bitcoin - Mayor liquidez en Kraken
    'ETH/USDT',    # Ethereum - Segunda mayor liquidez
    'XRP/USDT',    # Ripple
    'SOL/USDT',    # Solana
    'ADA/USDT',    # Cardano
    'DOGE/USDT',   # Dogecoin
    'AVAX/USDT',   # Avalanche
    'MATIC/USDT',  # Polygon
    'DOT/USDT',    # Polkadot
    'LTC/USDT',    # Litecoin
    'LINK/USDT',   # Chainlink (alta liquidez)
]

# Analysis Configuration
CHECK_INTERVAL = int(os.getenv('CHECK_INTERVAL', 180))  # 3 minutes default (más frecuente)
TIMEFRAME = '1h'  # Candlestick timeframe for conservative signals
FLASH_TIMEFRAME = '15m'  # Timeframe for flash signals (risky but faster) - Kraken soporta: 1m, 5m, 15m, 30m, 1h, 4h, 12h, 1d

# Signal Types Configuration
ENABLE_FLASH_SIGNALS = os.getenv('ENABLE_FLASH_SIGNALS', 'true').lower() == 'true'
CONSERVATIVE_THRESHOLD = 7.0  # Score threshold for conservative signals (1h/4h/1d)
FLASH_THRESHOLD = 4.0  # Lower threshold for flash signals (10m) - more risky

# Daily Report Configuration
DAILY_REPORT_HOUR = 21  # 9 PM
DAILY_REPORT_MINUTE = 0
TIMEZONE = 'America/Mexico_City'  # Hora CDMX

# Signal Tracking (for accuracy calculation)
TRACK_SIGNALS = True
TRACKING_FILE = 'logs/signal_tracking.json'
PROFIT_THRESHOLD = 2.0  # 2% profit to consider signal successful

# Technical Indicators Thresholds
RSI_OVERSOLD = int(os.getenv('RSI_OVERSOLD', 30))
RSI_OVERBOUGHT = int(os.getenv('RSI_OVERBOUGHT', 70))
RSI_PERIOD = 14

# MACD Configuration
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Moving Averages
EMA_SHORT = 9
EMA_MEDIUM = 21
EMA_LONG = 50

# Bollinger Bands
BB_PERIOD = 20
BB_STD = 2

# Logging
LOG_LEVEL = 'INFO'
LOG_FILE = 'logs/trading_bot.log'
