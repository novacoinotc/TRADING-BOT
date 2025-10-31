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

# Proxy Configuration (Required for Binance on Railway/datacenter IPs)
# Get proxy from: Webshare.io, Smartproxy, Bright Data, etc.
USE_PROXY = os.getenv('USE_PROXY', 'true').lower() == 'true'
PROXY_HOST = os.getenv('PROXY_HOST', '')  # e.g., proxy.webshare.io
PROXY_PORT = os.getenv('PROXY_PORT', '')  # e.g., 80 or 443
PROXY_USERNAME = os.getenv('PROXY_USERNAME', '')
PROXY_PASSWORD = os.getenv('PROXY_PASSWORD', '')

# Trading Pairs to Monitor - HIGH VOLATILITY BINANCE CONFIGURATION
# Pares con MAYOR LIQUIDEZ y volatilidad de Binance
TRADING_PAIRS = [
    # Top Tier - Máxima Liquidez (Anchors)
    'BTC/USDT',    # Bitcoin - $30B+ volumen 24h
    'ETH/USDT',    # Ethereum - $15B+ volumen 24h

    # High Volatility Layer 1s (Liquidez Alta)
    'SOL/USDT',    # Solana - $2B+ volumen, muy volátil
    'BNB/USDT',    # Binance Coin - $1B+ volumen
    'TON/USDT',    # The Open Network (Telegram) - $500M+ volumen, NUEVO TOP
    'SUI/USDT',    # Sui - $500M+ volumen, Layer 1 emergente
    'AVAX/USDT',   # Avalanche - $500M+ volumen
    'APT/USDT',    # Aptos - $300M+ volumen, Layer 1
    'DOT/USDT',    # Polkadot - $300M+ volumen
    'MATIC/USDT',  # Polygon - $500M+ volumen
    'ATOM/USDT',   # Cosmos - $200M+ volumen
    'NEAR/USDT',   # NEAR Protocol - $300M+ volumen
    'ADA/USDT',    # Cardano - $500M+ volumen

    # Layer 2s - Alta Liquidez
    'ARB/USDT',    # Arbitrum - $400M+ volumen, Layer 2 de Ethereum
    'OP/USDT',     # Optimism - $200M+ volumen, Layer 2 de Ethereum

    # Meme Coins - MÁXIMA Volatilidad
    'DOGE/USDT',   # Dogecoin - $1B+ volumen
    'SHIB/USDT',   # Shiba Inu - $500M+ volumen
    'PEPE/USDT',   # Pepe - $400M+ volumen
    'NOT/USDT',    # Notcoin - $300M+ volumen, MUY nuevo y volátil
    'WIF/USDT',    # Dogwifhat - $200M+ volumen, MUY volátil
    'BONK/USDT',   # Bonk - $150M+ volumen

    # DeFi Tokens - Alta Volatilidad
    'UNI/USDT',    # Uniswap - $300M+ volumen
    'LINK/USDT',   # Chainlink - $400M+ volumen
    'AAVE/USDT',   # AAVE - $200M+ volumen
    'INJ/USDT',    # Injective - $200M+ volumen, DeFi
    'CRV/USDT',    # Curve - $100M+ volumen

    # Payment/Transfer - Volátiles
    'XRP/USDT',    # Ripple - $2B+ volumen
    'LTC/USDT',    # Litecoin - $500M+ volumen
    'TRX/USDT',    # Tron - $400M+ volumen
    'XLM/USDT',    # Stellar - $200M+ volumen

    # Gaming/Metaverse - Alta Volatilidad
    'SAND/USDT',   # The Sandbox - $100M+ volumen
    'MANA/USDT',   # Decentraland - $80M+ volumen
    'AXS/USDT',    # Axie Infinity - $100M+ volumen

    # Altcoins Volátiles
    'FTM/USDT',    # Fantom - $150M+ volumen
    'TIA/USDT',    # Celestia - $200M+ volumen, modular blockchain
    'SEI/USDT',    # Sei - $150M+ volumen, Layer 1
    'ALGO/USDT',   # Algorand - $100M+ volumen
]

# Analysis Configuration
CHECK_INTERVAL = int(os.getenv('CHECK_INTERVAL', 120))  # 2 minutes (más frecuente para experimentar)
TIMEFRAME = '1h'  # Candlestick timeframe for conservative signals
FLASH_TIMEFRAME = '15m'  # Timeframe for flash signals (risky but faster) - Binance soporta: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M

# Signal Types Configuration
ENABLE_FLASH_SIGNALS = os.getenv('ENABLE_FLASH_SIGNALS', 'true').lower() == 'true'
CONSERVATIVE_THRESHOLD = 7.0  # Score threshold for conservative signals (1h/4h/1d)
FLASH_THRESHOLD = 5.0  # Medium threshold for flash signals (15m) - Más selectivo
FLASH_MIN_CONFIDENCE = 50  # Minimum confidence % for flash signals

# Daily Report Configuration
DAILY_REPORT_HOUR = 21  # 9 PM
DAILY_REPORT_MINUTE = 0
TIMEZONE = 'America/Mexico_City'  # Hora CDMX

# Signal Tracking (for accuracy calculation)
TRACK_SIGNALS = True
TRACKING_FILE = 'logs/signal_tracking.json'
PROFIT_THRESHOLD = 2.0  # 2% profit to consider signal successful

# Paper Trading + ML Configuration
ENABLE_PAPER_TRADING = os.getenv('ENABLE_PAPER_TRADING', 'true').lower() == 'true'
PAPER_TRADING_INITIAL_BALANCE = float(os.getenv('PAPER_TRADING_INITIAL_BALANCE', '50000.0'))  # $50,000 USDT

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
