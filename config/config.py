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
CHECK_INTERVAL = int(os.getenv('CHECK_INTERVAL', 120))  # 120 segundos - OPTIMIZADO para evitar rate limits con 37 pares
TIMEFRAME = '1h'  # Candlestick timeframe for conservative signals
FLASH_TIMEFRAME = '15m'  # Timeframe for flash signals (risky but faster) - Binance soporta: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M

# Signal Types Configuration
ENABLE_FLASH_SIGNALS = os.getenv('ENABLE_FLASH_SIGNALS', 'true').lower() == 'true'
CONSERVATIVE_THRESHOLD = 5.5  # Score threshold for conservative signals - OPTIMIZADO (antes 7.0, muy estricto)
FLASH_THRESHOLD = 6.0  # Threshold para flash signals - BALANCEADO (selectivo pero genera señales)
FLASH_MIN_CONFIDENCE = 55  # Minimum confidence % for flash signals - BALANCEADO

# Daily Report Configuration
DAILY_REPORT_HOUR = 21  # 9 PM
DAILY_REPORT_MINUTE = 0
TIMEZONE = 'America/Mexico_City'  # Hora CDMX

# Signal Tracking (for accuracy calculation)
TRACK_SIGNALS = True
TRACKING_FILE = 'logs/signal_tracking.json'
PROFIT_THRESHOLD = 1.5  # 1.5% profit to consider signal successful - OPTIMIZADO (antes 2.0%, más realista)

# Paper Trading + ML Configuration
ENABLE_PAPER_TRADING = os.getenv('ENABLE_PAPER_TRADING', 'true').lower() == 'true'
PAPER_TRADING_INITIAL_BALANCE = float(os.getenv('PAPER_TRADING_INITIAL_BALANCE', '50000.0'))  # $50,000 USDT

# Historical Training Configuration (Pre-entrenar modelo con datos pasados)
ENABLE_HISTORICAL_TRAINING = os.getenv('ENABLE_HISTORICAL_TRAINING', 'true').lower() == 'true'
HISTORICAL_START_DATE = os.getenv('HISTORICAL_START_DATE', '2025-01-01')  # Solo 2025 (10 meses = más rápido y datos recientes)
HISTORICAL_END_DATE = os.getenv('HISTORICAL_END_DATE', '2025-11-01')  # Hasta hoy
HISTORICAL_TIMEFRAMES = os.getenv('HISTORICAL_TIMEFRAMES', '1h,4h,1d,15m').split(',')  # Timeframes a descargar
MIN_HISTORICAL_SAMPLES = int(os.getenv('MIN_HISTORICAL_SAMPLES', '200'))  # Mínimo de señales históricas requeridas
# IMPORTANTE: Si tienes errores de rate limit en Binance por múltiples redeploys,
# temporalmente deshabilita el backtest con ENABLE_HISTORICAL_TRAINING=false en Railway
# y espera 30-60 minutos antes de reactivarlo

# Sentiment Analysis Configuration
ENABLE_SENTIMENT_ANALYSIS = os.getenv('ENABLE_SENTIMENT_ANALYSIS', 'true').lower() == 'true'
CRYPTOPANIC_API_KEY = os.getenv('CRYPTOPANIC_API_KEY', '')  # Get free key from cryptopanic.com
SENTIMENT_UPDATE_INTERVAL = int(os.getenv('SENTIMENT_UPDATE_INTERVAL', '15'))  # Minutos entre updates (optimizado GROWTH: 2,880 req/mes)
SENTIMENT_BLOCK_ON_EXTREME_FEAR = os.getenv('SENTIMENT_BLOCK_ON_EXTREME_FEAR', 'true').lower() == 'true'
SENTIMENT_BOOST_ON_POSITIVE = os.getenv('SENTIMENT_BOOST_ON_POSITIVE', 'true').lower() == 'true'
FORCE_HISTORICAL_DOWNLOAD = os.getenv('FORCE_HISTORICAL_DOWNLOAD', 'false').lower() == 'true'  # Forzar re-descarga
SKIP_HISTORICAL_IF_MODEL_EXISTS = os.getenv('SKIP_HISTORICAL_IF_MODEL_EXISTS', 'true').lower() == 'true'  # Skip si ya hay modelo

# Technical Indicators Thresholds
RSI_OVERSOLD = int(os.getenv('RSI_OVERSOLD', 35))  # OPTIMIZADO: 35 en lugar de 30 (captura rebotes más temprano)
RSI_OVERBOUGHT = int(os.getenv('RSI_OVERBOUGHT', 65))  # OPTIMIZADO: 65 en lugar de 70 (captura reversiones más temprano)
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

# Autonomous AI System Configuration - CONTROL ABSOLUTO
# La IA tiene poder total para modificar TODOS los parámetros sin limitaciones
ENABLE_AUTONOMOUS_MODE = os.getenv('ENABLE_AUTONOMOUS_MODE', 'true').lower() == 'true'
AUTONOMOUS_AUTO_SAVE_INTERVAL = int(os.getenv('AUTONOMOUS_AUTO_SAVE_INTERVAL', '30'))  # Minutos entre auto-saves
AUTONOMOUS_OPTIMIZATION_INTERVAL = float(os.getenv('AUTONOMOUS_OPTIMIZATION_INTERVAL', '2.0'))  # Horas entre optimizaciones
AUTONOMOUS_MIN_TRADES_BEFORE_OPT = int(os.getenv('AUTONOMOUS_MIN_TRADES_BEFORE_OPT', '20'))  # Mínimo trades antes de optimizar

# ============================================================================
# NUEVOS PARÁMETROS DE AUTONOMÍA 100% - CONTROL TOTAL DE LA IA
# ============================================================================

# Smart Order Routing (Spot vs Futures dinámico)
# La IA decide automáticamente si usar SPOT o FUTURES según condiciones de mercado
SMART_ROUTING_ENABLED = os.getenv('SMART_ROUTING_ENABLED', 'true').lower() == 'true'
MIN_CONFIDENCE_FOR_FUTURES = float(os.getenv('MIN_CONFIDENCE_FOR_FUTURES', '70.0'))  # 60-85% (optimizable)
MIN_WINRATE_FOR_FUTURES = float(os.getenv('MIN_WINRATE_FOR_FUTURES', '55.0'))  # 45-65% (optimizable)
MAX_DRAWDOWN_FOR_FUTURES = float(os.getenv('MAX_DRAWDOWN_FOR_FUTURES', '10.0'))  # 5-15% (optimizable)
VOLATILITY_THRESHOLD_FUTURES = float(os.getenv('VOLATILITY_THRESHOLD_FUTURES', '0.02'))  # 0.015-0.03 (optimizable)
CONSERVATIVE_LEVERAGE = int(os.getenv('CONSERVATIVE_LEVERAGE', '3'))  # 2-5x (optimizable)
BALANCED_LEVERAGE = int(os.getenv('BALANCED_LEVERAGE', '8'))  # 5-10x (optimizable)
AGGRESSIVE_LEVERAGE = int(os.getenv('AGGRESSIVE_LEVERAGE', '15'))  # 10-20x (optimizable)

# Trailing Stops Automáticos
# Stop loss dinámico que sigue el precio y protege ganancias automáticamente
TRAILING_STOP_ENABLED = os.getenv('TRAILING_STOP_ENABLED', 'true').lower() == 'true'
TRAILING_DISTANCE_PCT = float(os.getenv('TRAILING_DISTANCE_PCT', '0.4'))  # 0.3-0.7% (optimizable)
BREAKEVEN_AFTER_PCT = float(os.getenv('BREAKEVEN_AFTER_PCT', '0.5'))  # 0.3-1.0% (optimizable)
LOCK_PROFIT_STEP_PCT = float(os.getenv('LOCK_PROFIT_STEP_PCT', '0.5'))  # 0.3-0.8% (optimizable)
MIN_PROFIT_TO_LOCK_PCT = float(os.getenv('MIN_PROFIT_TO_LOCK_PCT', '0.3'))  # 0.2-0.5% (optimizable)

# Position Sizing Agresivo
# Ampliado de 8% a 12% para permitir mayor agresividad en señales excelentes
MAX_POSITION_SIZE_PCT = float(os.getenv('MAX_POSITION_SIZE_PCT', '12.0'))  # Máximo 12% (antes 8%)
BASE_POSITION_SIZE_PCT = float(os.getenv('BASE_POSITION_SIZE_PCT', '4.0'))  # 2-12% (optimizable)

# Anomaly Detection System
# Detecta comportamiento anómalo y revierte parámetros automáticamente
ANOMALY_DETECTION_ENABLED = os.getenv('ANOMALY_DETECTION_ENABLED', 'true').lower() == 'true'
PERFORMANCE_DEGRADATION_THRESHOLD = float(os.getenv('PERFORMANCE_DEGRADATION_THRESHOLD', '10.0'))  # 5-20% (optimizable)
OUTLIER_STD_THRESHOLD = float(os.getenv('OUTLIER_STD_THRESHOLD', '3.0'))  # 2.0-4.0 (optimizable)
MIN_TRADES_FOR_DETECTION = int(os.getenv('MIN_TRADES_FOR_DETECTION', '20'))  # 10-50 (optimizable)
ANOMALY_LOOKBACK_WINDOW = int(os.getenv('ANOMALY_LOOKBACK_WINDOW', '50'))  # 30-100 (optimizable)
AUTO_REVERT_ENABLED = os.getenv('AUTO_REVERT_ENABLED', 'true').lower() == 'true'

# A/B Testing de Estrategias
# Prueba dos estrategias en paralelo y switch automático a la ganadora (EXPERIMENTAL)
AB_TESTING_ENABLED = os.getenv('AB_TESTING_ENABLED', 'false').lower() == 'true'  # Deshabilitado por defecto
AB_TEST_DURATION_TRADES = int(os.getenv('AB_TEST_DURATION_TRADES', '50'))  # 30-100 (optimizable)
AB_TEST_DURATION_DAYS = int(os.getenv('AB_TEST_DURATION_DAYS', '7'))  # 3-14 (optimizable)
AB_TEST_CAPITAL_SPLIT = float(os.getenv('AB_TEST_CAPITAL_SPLIT', '0.5'))  # 0.3-0.7 (optimizable)
AB_TEST_MIN_CONFIDENCE = float(os.getenv('AB_TEST_MIN_CONFIDENCE', '0.8'))  # 0.7-0.95 (optimizable)
AB_TEST_METRIC = os.getenv('AB_TEST_METRIC', 'win_rate')  # 'win_rate', 'profit_factor', 'sharpe_ratio'

# ============================================================================
# RESUMEN DE PARÁMETROS OPTIMIZABLES POR LA IA
# ============================================================================
# TOTAL: 62 parámetros optimizables (antes 41, +21 nuevos)
#
# Categorías:
# 1. Trading Thresholds (5): CONSERVATIVE_THRESHOLD, FLASH_THRESHOLD, etc.
# 2. Technical Indicators (11): RSI_PERIOD, MACD_FAST, EMA_SHORT, etc.
# 3. Risk Management (4): BASE_POSITION_SIZE_PCT, MAX_DRAWDOWN_LIMIT, etc.
# 4. ML Hyperparameters (9): N_ESTIMATORS, MAX_DEPTH, LEARNING_RATE, etc.
# 5. Sentiment Analysis (8): NEWS_IMPORTANCE_THRESHOLD, etc.
# 6. Dynamic Take Profits (5): TP1_BASE_PCT, TP2_BASE_PCT, etc.
# 7. Smart Routing (7): MIN_CONFIDENCE_FOR_FUTURES, CONSERVATIVE_LEVERAGE, etc.
# 8. Trailing Stops (4): TRAILING_DISTANCE_PCT, BREAKEVEN_AFTER_PCT, etc.
# 9. Anomaly Detection (4): PERFORMANCE_DEGRADATION_THRESHOLD, etc.
# 10. A/B Testing (5): AB_TEST_DURATION_TRADES, AB_TEST_CAPITAL_SPLIT, etc.
#
# La IA puede modificar CUALQUIERA de estos parámetros sin intervención humana
# basándose en performance histórica y aprendizaje continuo
# ============================================================================
