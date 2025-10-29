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
EXCHANGE_NAME = os.getenv('EXCHANGE_NAME', 'binance')
EXCHANGE_API_KEY = os.getenv('EXCHANGE_API_KEY', '')
EXCHANGE_API_SECRET = os.getenv('EXCHANGE_API_SECRET', '')

# Trading Pairs to Monitor
TRADING_PAIRS = [
    'BTC/USDT',
    'ETH/USDT',
    'XRP/USDT',
    'MXN/USD',  # Note: Availability depends on exchange
]

# Analysis Configuration
CHECK_INTERVAL = int(os.getenv('CHECK_INTERVAL', 300))  # 5 minutes default
TIMEFRAME = '1h'  # Candlestick timeframe

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
