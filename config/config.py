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

# Trading Pairs to Monitor - HIGH VOLATILITY CONFIGURATION
# Maximizar señales flash con pares de alta volatilidad
TRADING_PAIRS = [
    # Anchors (referencia de mercado)
    'BTC/USDT',    # Bitcoin - Referencia
    'ETH/USDT',    # Ethereum - Referencia

    # High Volatility Layer 1s (Movimientos grandes)
    'SOL/USDT',    # Solana - Muy volátil, swings grandes
    'AVAX/USDT',   # Avalanche - Alta volatilidad
    'ATOM/USDT',   # Cosmos - Movimientos fuertes
    'NEAR/USDT',   # NEAR Protocol - Swings significativos
    'FTM/USDT',    # Fantom - DeFi volatility
    'ALGO/USDT',   # Algorand - Movimientos rápidos

    # Meme Coins (MÁXIMA volatilidad - MUCHAS señales)
    'DOGE/USDT',   # Dogecoin - Meme king
    'SHIB/USDT',   # Shiba Inu - Volatilidad extrema

    # DeFi Tokens (Alta volatilidad intraday)
    'AAVE/USDT',   # Lending protocol - Swings grandes
    'UNI/USDT',    # Uniswap DEX - Muy activo
    'CRV/USDT',    # Curve Finance - Alta volatilidad
    'LINK/USDT',   # Chainlink - Movimientos significativos

    # Gaming/Metaverse (Volatilidad por noticias)
    'SAND/USDT',   # The Sandbox - Gaming volatility
    'MANA/USDT',   # Decentraland - Metaverse swings

    # Additional High-Volatility
    'DOT/USDT',    # Polkadot - Swings considerables
    'ADA/USDT',    # Cardano - Movimientos frecuentes
]

# Analysis Configuration
CHECK_INTERVAL = int(os.getenv('CHECK_INTERVAL', 120))  # 2 minutes (más frecuente para experimentar)
TIMEFRAME = '1h'  # Candlestick timeframe for conservative signals
FLASH_TIMEFRAME = '15m'  # Timeframe for flash signals (risky but faster) - Kraken soporta: 1m, 5m, 15m, 30m, 1h, 4h, 12h, 1d

# Signal Types Configuration
ENABLE_FLASH_SIGNALS = os.getenv('ENABLE_FLASH_SIGNALS', 'true').lower() == 'true'
CONSERVATIVE_THRESHOLD = 7.0  # Score threshold for conservative signals (1h/4h/1d)
FLASH_THRESHOLD = 3.5  # Lower threshold for flash signals (15m) - EXPERIMENTAL: más señales

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
