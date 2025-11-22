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

# Binance Futures API Configuration (v2.0 - Real Trading)
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET', '')
BINANCE_TESTNET = os.getenv('BINANCE_TESTNET', 'true').lower() == 'true'
BINANCE_BASE_URL = os.getenv('BINANCE_BASE_URL', 'https://testnet.binancefuture.com' if os.getenv('BINANCE_TESTNET', 'true').lower() == 'true' else 'https://fapi.binance.com')
BINANCE_WS_URL = os.getenv('BINANCE_WS_URL', 'wss://stream.binancefuture.com' if os.getenv('BINANCE_TESTNET', 'true').lower() == 'true' else 'wss://fstream.binance.com')

# Trading Mode Configuration
TRADING_MODE = os.getenv('TRADING_MODE', 'testnet')  # 'testnet' or 'live'
AUTO_TRADE = os.getenv('AUTO_TRADE', 'true').lower() == 'true'  # Enable/disable automatic trading
DEFAULT_LEVERAGE = int(os.getenv('DEFAULT_LEVERAGE', '3'))  # Default leverage for futures (1-125x)
USE_ISOLATED_MARGIN = os.getenv('USE_ISOLATED_MARGIN', 'true').lower() == 'true'  # Use isolated margin (recommended)

# Trade Amounts (USDT)
TRADE_AMOUNT_USDT = float(os.getenv('TRADE_AMOUNT_USDT', '100.0'))  # Amount in USDT per regular trade
FLASH_TRADE_AMOUNT_USDT = float(os.getenv('FLASH_TRADE_AMOUNT_USDT', '50.0'))  # Amount for flash trades

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
    # 'FTM/USDT',    # Fantom - DESHABILITADO: Order book vacío en Binance
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
CONSERVATIVE_THRESHOLD = 4.5  # MODO EXPLORACIÓN - más accesible (antes 5.5, ahora permite más trades)
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

# ML Configuration + Historical Training (Pre-entrenar modelo con datos pasados)
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
ENABLE_ML_SYSTEM = os.getenv('ENABLE_ML_SYSTEM', 'true').lower() == 'true'  # Machine Learning predictions (v2.0)
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

# ============================================================================
# ARSENAL AVANZADO - TIER 1, 2 Y 3 (Análisis Profesional Institucional)
# ============================================================================

# Correlation Matrix (evitar overexposure)
CORRELATION_ANALYSIS_ENABLED = os.getenv('CORRELATION_ANALYSIS_ENABLED', 'true').lower() == 'true'
HIGH_CORRELATION_THRESHOLD = float(os.getenv('HIGH_CORRELATION_THRESHOLD', '0.7'))  # 0.6-0.85 (optimizable)
CORRELATION_LOOKBACK_PERIODS = int(os.getenv('CORRELATION_LOOKBACK_PERIODS', '100'))  # 50-200 (optimizable)
CORRELATION_MIN_DATA_POINTS = int(os.getenv('CORRELATION_MIN_DATA_POINTS', '30'))  # 20-50 (optimizable)
MAX_CORRELATED_POSITIONS = int(os.getenv('MAX_CORRELATED_POSITIONS', '2'))  # 1-3 (optimizable)

# Liquidation Heatmap Analysis
LIQUIDATION_ANALYSIS_ENABLED = os.getenv('LIQUIDATION_ANALYSIS_ENABLED', 'true').lower() == 'true'
MIN_LIQUIDATION_VOLUME_USD = float(os.getenv('MIN_LIQUIDATION_VOLUME_USD', '1000000'))  # $1M (0.5M-5M optimizable)
LIQUIDATION_PROXIMITY_THRESHOLD_PCT = float(os.getenv('LIQUIDATION_PROXIMITY_THRESHOLD_PCT', '2.0'))  # 2% (1-5% optimizable)
LIQUIDATION_BOOST_FACTOR = float(os.getenv('LIQUIDATION_BOOST_FACTOR', '1.3'))  # 1.1-1.5x (optimizable)
LIQUIDATION_LOOKBACK_HOURS = int(os.getenv('LIQUIDATION_LOOKBACK_HOURS', '24'))  # 12-48h (optimizable)
USE_MOCK_LIQUIDATION_DATA = os.getenv('USE_MOCK_LIQUIDATION_DATA', 'false').lower() == 'true'  # True para testing

# Funding Rate Analysis
FUNDING_RATE_ANALYSIS_ENABLED = os.getenv('FUNDING_RATE_ANALYSIS_ENABLED', 'true').lower() == 'true'
FUNDING_EXTREME_POSITIVE = float(os.getenv('FUNDING_EXTREME_POSITIVE', '0.10'))  # 0.08-0.15% (optimizable)
FUNDING_EXTREME_NEGATIVE = float(os.getenv('FUNDING_EXTREME_NEGATIVE', '-0.10'))  # -0.15 a -0.08% (optimizable)
FUNDING_HIGH_POSITIVE = float(os.getenv('FUNDING_HIGH_POSITIVE', '0.05'))  # 0.03-0.08% (optimizable)
FUNDING_HIGH_NEGATIVE = float(os.getenv('FUNDING_HIGH_NEGATIVE', '-0.05'))  # -0.08 a -0.03% (optimizable)
FUNDING_BOOST_EXTREME = float(os.getenv('FUNDING_BOOST_EXTREME', '1.5'))  # 1.3-1.8x (optimizable)
FUNDING_BOOST_HIGH = float(os.getenv('FUNDING_BOOST_HIGH', '1.2'))  # 1.1-1.4x (optimizable)

# Volume Profile & POC
VOLUME_PROFILE_ENABLED = os.getenv('VOLUME_PROFILE_ENABLED', 'true').lower() == 'true'
VOLUME_PROFILE_LOOKBACK = int(os.getenv('VOLUME_PROFILE_LOOKBACK', '100'))  # 50-200 (optimizable)
VOLUME_PROFILE_BINS = int(os.getenv('VOLUME_PROFILE_BINS', '50'))  # 30-100 (optimizable)
VOLUME_PROFILE_VALUE_AREA = float(os.getenv('VOLUME_PROFILE_VALUE_AREA', '70.0'))  # 65-75% (optimizable)
POC_PROXIMITY_PCT = float(os.getenv('POC_PROXIMITY_PCT', '1.0'))  # 0.5-2% (optimizable)
POC_BOOST_FACTOR = float(os.getenv('POC_BOOST_FACTOR', '1.3'))  # 1.2-1.5x (optimizable)
VALUE_AREA_BOOST_FACTOR = float(os.getenv('VALUE_AREA_BOOST_FACTOR', '1.15'))  # 1.1-1.3x (optimizable)

# Technical Pattern Recognition
PATTERN_RECOGNITION_ENABLED = os.getenv('PATTERN_RECOGNITION_ENABLED', 'true').lower() == 'true'
MIN_PATTERN_CONFIDENCE = float(os.getenv('MIN_PATTERN_CONFIDENCE', '0.7'))  # 0.6-0.85 (optimizable)
PATTERN_LOOKBACK_CANDLES = int(os.getenv('PATTERN_LOOKBACK_CANDLES', '50'))  # 30-100 (optimizable)
PATTERN_BOOST_FACTOR = float(os.getenv('PATTERN_BOOST_FACTOR', '1.4'))  # 1.2-1.6x (optimizable)

# Session-Based Trading
SESSION_TRADING_ENABLED = os.getenv('SESSION_TRADING_ENABLED', 'true').lower() == 'true'
US_OPEN_BOOST = float(os.getenv('US_OPEN_BOOST', '1.3'))  # 1.2-1.5x (optimizable)
SESSION_OVERLAP_BOOST = float(os.getenv('SESSION_OVERLAP_BOOST', '1.2'))  # 1.1-1.4x (optimizable)
ASIAN_SESSION_PENALTY = float(os.getenv('ASIAN_SESSION_PENALTY', '0.9'))  # 0.85-0.95x (optimizable)

# Order Flow Imbalance
ORDER_FLOW_ENABLED = os.getenv('ORDER_FLOW_ENABLED', 'true').lower() == 'true'
STRONG_IMBALANCE_RATIO = float(os.getenv('STRONG_IMBALANCE_RATIO', '2.5'))  # 2.0-3.5 (optimizable)
MODERATE_IMBALANCE_RATIO = float(os.getenv('MODERATE_IMBALANCE_RATIO', '1.5'))  # 1.3-2.0 (optimizable)
ORDER_FLOW_BOOST_STRONG = float(os.getenv('ORDER_FLOW_BOOST_STRONG', '1.3'))  # 1.2-1.5x (optimizable)
ORDER_FLOW_BOOST_MODERATE = float(os.getenv('ORDER_FLOW_BOOST_MODERATE', '1.15'))  # 1.1-1.3x (optimizable)

# ============================================================================
# RESUMEN FINAL DE PARÁMETROS OPTIMIZABLES
# ============================================================================
# TOTAL: 93 parámetros optimizables (antes 62, +31 nuevos del arsenal avanzado)
#
# Categorías expandidas:
# 1. Trading Thresholds (5)
# 2. Technical Indicators (11)
# 3. Risk Management (4)
# 4. ML Hyperparameters (9)
# 5. Sentiment Analysis (8)
# 6. Dynamic Take Profits (5)
# 7. Smart Routing (7)
# 8. Trailing Stops (4)
# 9. Anomaly Detection (4)
# 10. A/B Testing (5)
# 11. Correlation Matrix (4) ← NUEVO
# 12. Liquidation Heatmap (4) ← NUEVO
# 13. Funding Rate (6) ← NUEVO
# 14. Volume Profile (7) ← NUEVO
# 15. Pattern Recognition (3) ← NUEVO
# 16. Session Trading (3) ← NUEVO
# 17. Order Flow (4) ← NUEVO
#
# El bot ahora es un TRADER INSTITUCIONAL con acceso a:
# ✅ Análisis de correlación (evitar overexposure)
# ✅ Liquidation hunting (trade hacia liquidaciones)
# ✅ Funding rate extremes (sentiment overleveraged)
# ✅ Volume Profile & POC (zonas de valor real)
# ✅ Pattern recognition (Head & Shoulders, Double Top/Bottom)
# ✅ Session-based trading (adaptación a volatilidad horaria)
# ✅ Order flow imbalance (momentum antes que precio)
#
# = EL MEJOR TRADER DEL MUNDO, 100% AUTÓNOMO, SIN LÍMITES =
# ============================================================================
