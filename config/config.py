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
FLASH_THRESHOLD = 4.5  # Threshold para flash signals - AGRESIVO para scalping (prueba 24h)
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
# NOTA: ENABLE_PAPER_TRADING habilita el sistema de trading (Paper o Live)
# El modo actual se determina por TRADING_MODE ('PAPER' o 'LIVE')
# Si TRADING_MODE=LIVE, usara Binance Futures real en lugar de simulacion
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
BASE_POSITION_SIZE_PCT = float(os.getenv('BASE_POSITION_SIZE_PCT', '10.0'))  # 10% para scalping (antes 4%)

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

# ============================================================================
# BINANCE FUTURES LIVE TRADING CONFIGURATION
# ============================================================================
# ADVERTENCIA: Esta configuracion habilita trading REAL con dinero real.
# Asegurate de configurar correctamente antes de activar.
# ============================================================================

# Trading Mode: 'PAPER' (simulado) o 'LIVE' (real)
# PAPER: Usa el paper_trader.py (sin riesgo, simulacion)
# LIVE: Usa live_trader.py con Binance Futures API (dinero real)
TRADING_MODE = os.getenv('TRADING_MODE', 'PAPER')  # 'PAPER' o 'LIVE'

# Binance Futures API Credentials
# IMPORTANTE: Estas credenciales deben tener permisos de Futures trading
# Crealas en: https://www.binance.com/en/my/settings/api-management
BINANCE_FUTURES_API_KEY = os.getenv('BINANCE_FUTURES_API_KEY', '')
BINANCE_FUTURES_API_SECRET = os.getenv('BINANCE_FUTURES_API_SECRET', '')

# Testnet Mode (recomendado para pruebas iniciales)
# Testnet URL: https://testnet.binancefuture.com
# Testnet API Keys: https://testnet.binancefuture.com/en/futures/BTCUSDT (Settings)
BINANCE_FUTURES_TESTNET = os.getenv('BINANCE_FUTURES_TESTNET', 'false').lower() == 'true'

# Live Trading Configuration
LIVE_TRADING_INITIAL_BALANCE = float(os.getenv('LIVE_TRADING_INITIAL_BALANCE', '0'))  # 0 = auto-detect from Binance

# Leverage and Margin Settings
DEFAULT_LEVERAGE = int(os.getenv('DEFAULT_LEVERAGE', '10'))  # 1-125x (recomendado 5-15x)
MARGIN_TYPE = os.getenv('MARGIN_TYPE', 'ISOLATED')  # 'ISOLATED' o 'CROSSED' (ISOLATED recomendado)
USE_HEDGE_MODE = os.getenv('USE_HEDGE_MODE', 'false').lower() == 'true'  # False = One-way mode (recomendado)

# Position Limits for Live Trading (mas conservadores que paper)
MAX_POSITIONS = int(os.getenv('MAX_POSITIONS', '5'))  # Maximo posiciones simultaneas (reducido para live)
MAX_DRAWDOWN_LIMIT = float(os.getenv('MAX_DRAWDOWN_LIMIT', '10.0'))  # % drawdown maximo antes de reducir riesgo
MAX_RISK_PER_TRADE_PCT = float(os.getenv('MAX_RISK_PER_TRADE_PCT', '1.0'))  # % riesgo maximo por trade

# User Data Stream (WebSocket para actualizaciones en tiempo real)
ENABLE_USER_DATA_STREAM = os.getenv('ENABLE_USER_DATA_STREAM', 'true').lower() == 'true'

# Emergency Settings
EMERGENCY_STOP_LOSS_PCT = float(os.getenv('EMERGENCY_STOP_LOSS_PCT', '5.0'))  # % de perdida total para cerrar todo
ENABLE_AUTO_STOP = os.getenv('ENABLE_AUTO_STOP', 'true').lower() == 'true'  # Auto-stop si perdidas excesivas

# Minimum Trade Values (Binance requirements)
MIN_NOTIONAL_VALUE = float(os.getenv('MIN_NOTIONAL_VALUE', '5.0'))  # Minimo valor de orden en USDT

# ============================================================================
# GPT BRAIN CONFIGURATION - ADVANCED AI REASONING
# ============================================================================
# GPT Brain provides advanced reasoning capabilities for the trading bot:
# - Meta-Reasoner: Analyzes performance and suggests improvements
# - Decision Explainer: Explains trading decisions in natural language
# - Risk Assessor: Evaluates risk before executing trades
# - Strategy Advisor: Autonomously optimizes parameters
# ============================================================================

# Enable/Disable GPT Brain
ENABLE_GPT_BRAIN = os.getenv('ENABLE_GPT_BRAIN', 'true').lower() == 'true'

# OpenAI API Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')

# ============================================================================
# GPT MODEL SELECTION - Cost Optimization Strategy (GPT-5)
# ============================================================================
# Modelo FRECUENTE (95% de llamadas) - Para decisiones rápidas de trading
GPT_MODEL_FREQUENT = os.getenv('GPT_MODEL_FREQUENT', 'gpt-5-mini')

# Modelo PREMIUM (5% de llamadas) - Para análisis profundo y crítico
GPT_MODEL_PREMIUM = os.getenv('GPT_MODEL_PREMIUM', 'gpt-5.1')

# Modelo por defecto
GPT_MODEL = os.getenv('GPT_MODEL', 'gpt-5-mini')

# Cuándo usar modelo PREMIUM automáticamente:
GPT_USE_PREMIUM_FOR_LOSING_STREAK = os.getenv('GPT_USE_PREMIUM_FOR_LOSING_STREAK', 'true').lower() == 'true'
GPT_USE_PREMIUM_FOR_HIGH_VALUE = os.getenv('GPT_USE_PREMIUM_FOR_HIGH_VALUE', 'true').lower() == 'true'
GPT_HIGH_VALUE_THRESHOLD_USD = float(os.getenv('GPT_HIGH_VALUE_THRESHOLD_USD', '1000'))  # Trades > $1000
GPT_USE_PREMIUM_FOR_STRATEGY_REVIEW = os.getenv('GPT_USE_PREMIUM_FOR_STRATEGY_REVIEW', 'true').lower() == 'true'

# ============================================================================
# GPT RISK TOLERANCE - Aggressive Learning Mode
# ============================================================================
# Permitir que GPT tome señales de menor confianza si ve oportunidad
GPT_ALLOW_RISKY_TRADES = os.getenv('GPT_ALLOW_RISKY_TRADES', 'true').lower() == 'true'
GPT_MIN_CONFIDENCE_OVERRIDE = int(os.getenv('GPT_MIN_CONFIDENCE_OVERRIDE', '40'))  # Puede bajar hasta 40%
GPT_RISKY_TRADE_SIZE_REDUCTION = float(os.getenv('GPT_RISKY_TRADE_SIZE_REDUCTION', '0.5'))  # 50% del tamaño normal

# Aprendizaje agresivo de errores
GPT_AGGRESSIVE_LEARNING = os.getenv('GPT_AGGRESSIVE_LEARNING', 'true').lower() == 'true'
GPT_LEARN_FROM_EVERY_TRADE = os.getenv('GPT_LEARN_FROM_EVERY_TRADE', 'true').lower() == 'true'

# ============================================================================
# GPT DYNAMIC TP/SL - Sin mínimos fijos
# ============================================================================
# GPT decide TP/SL según condiciones (puede ser menor que 1.5%)
GPT_DYNAMIC_TP_ENABLED = os.getenv('GPT_DYNAMIC_TP_ENABLED', 'true').lower() == 'true'
GPT_MIN_TP_PCT = float(os.getenv('GPT_MIN_TP_PCT', '0.5'))  # Mínimo absoluto 0.5%
GPT_MAX_TP_PCT = float(os.getenv('GPT_MAX_TP_PCT', '10.0'))  # Máximo 10%
GPT_MIN_SL_PCT = float(os.getenv('GPT_MIN_SL_PCT', '0.3'))  # Mínimo absoluto 0.3%
GPT_MAX_SL_PCT = float(os.getenv('GPT_MAX_SL_PCT', '5.0'))  # Máximo 5%

# GPT Behavior Settings
GPT_TEMPERATURE = float(os.getenv('GPT_TEMPERATURE', '0.7'))  # 0.0-1.0 (creativity)
GPT_MAX_TOKENS = int(os.getenv('GPT_MAX_TOKENS', '2000'))  # Max response length

# GPT Features Toggle
GPT_RISK_ASSESSMENT = os.getenv('GPT_RISK_ASSESSMENT', 'true').lower() == 'true'
GPT_DECISION_EXPLANATION = os.getenv('GPT_DECISION_EXPLANATION', 'true').lower() == 'true'
GPT_STRATEGY_OPTIMIZATION = os.getenv('GPT_STRATEGY_OPTIMIZATION', 'true').lower() == 'true'
GPT_AUTO_PARAMETER_ADJUSTMENT = os.getenv('GPT_AUTO_PARAMETER_ADJUSTMENT', 'true').lower() == 'true'

# Optimization Intervals
GPT_OPTIMIZATION_INTERVAL_HOURS = float(os.getenv('GPT_OPTIMIZATION_INTERVAL_HOURS', '2.0'))
GPT_MIN_TRADES_FOR_OPTIMIZATION = int(os.getenv('GPT_MIN_TRADES_FOR_OPTIMIZATION', '20'))

# Risk Assessment Settings
GPT_BLOCK_HIGH_RISK_TRADES = os.getenv('GPT_BLOCK_HIGH_RISK_TRADES', 'true').lower() == 'true'
GPT_MAX_RISK_SCORE_TO_APPROVE = int(os.getenv('GPT_MAX_RISK_SCORE_TO_APPROVE', '70'))  # 0-100

# Reaction Settings
GPT_REACT_TO_LOSING_STREAK = os.getenv('GPT_REACT_TO_LOSING_STREAK', 'true').lower() == 'true'
GPT_LOSING_STREAK_THRESHOLD = int(os.getenv('GPT_LOSING_STREAK_THRESHOLD', '3'))
GPT_REACT_TO_WINNING_STREAK = os.getenv('GPT_REACT_TO_WINNING_STREAK', 'true').lower() == 'true'
GPT_WINNING_STREAK_THRESHOLD = int(os.getenv('GPT_WINNING_STREAK_THRESHOLD', '5'))

# Cost Control
GPT_MAX_DAILY_COST_USD = float(os.getenv('GPT_MAX_DAILY_COST_USD', '10.0'))  # Max daily API cost
GPT_CACHE_RESPONSES = os.getenv('GPT_CACHE_RESPONSES', 'true').lower() == 'true'

# Notification Settings
GPT_NOTIFY_BLOCKED_TRADES = os.getenv('GPT_NOTIFY_BLOCKED_TRADES', 'true').lower() == 'true'
GPT_NOTIFY_PARAMETER_CHANGES = os.getenv('GPT_NOTIFY_PARAMETER_CHANGES', 'true').lower() == 'true'
GPT_NOTIFY_ANALYSIS_RESULTS = os.getenv('GPT_NOTIFY_ANALYSIS_RESULTS', 'true').lower() == 'true'

# ============================================================================
# SCALPING STRATEGY CONFIGURATION
# ============================================================================
# Strategy: Many trades with quick entries/exits and tight risk management
# Philosophy: Small consistent wins > few large wins
# ============================================================================

# Scalping Mode (enables aggressive trading)
SCALPING_MODE = os.getenv('SCALPING_MODE', 'true').lower() == 'true'

# Scalping Risk Parameters (TIGHT)
SCALPING_STOP_LOSS_PCT = float(os.getenv('SCALPING_STOP_LOSS_PCT', '1.5'))  # 1-2% max
SCALPING_TAKE_PROFIT_PCT = float(os.getenv('SCALPING_TAKE_PROFIT_PCT', '2.5'))  # 1.5-3% typical
SCALPING_MIN_RR_RATIO = float(os.getenv('SCALPING_MIN_RR_RATIO', '1.5'))  # Minimum 1:1.5 R:R

# Scalping Entry Requirements
SCALPING_MIN_VOLUME_RATIO = float(os.getenv('SCALPING_MIN_VOLUME_RATIO', '1.0'))  # Volume > average
SCALPING_MAX_SPREAD_PCT = float(os.getenv('SCALPING_MAX_SPREAD_PCT', '0.1'))  # Max 0.1% spread
SCALPING_RSI_OVERSOLD = int(os.getenv('SCALPING_RSI_OVERSOLD', '25'))  # Buy zone
SCALPING_RSI_OVERBOUGHT = int(os.getenv('SCALPING_RSI_OVERBOUGHT', '75'))  # Sell zone

# Scalping Position Sizing
SCALPING_POSITION_SIZE_PCT = float(os.getenv('SCALPING_POSITION_SIZE_PCT', '10.0'))  # 10% per trade
SCALPING_MAX_POSITIONS = int(os.getenv('SCALPING_MAX_POSITIONS', '5'))  # Max concurrent positions

# Scalping Session Preferences
SCALPING_PREFER_US_SESSION = os.getenv('SCALPING_PREFER_US_SESSION', 'true').lower() == 'true'
SCALPING_AVOID_LOW_VOLUME = os.getenv('SCALPING_AVOID_LOW_VOLUME', 'true').lower() == 'true'

# Scalping GPT Requirements
SCALPING_MIN_CONFIDENCE = int(os.getenv('SCALPING_MIN_CONFIDENCE', '60'))  # Min GPT confidence
SCALPING_MIN_FACTORS = int(os.getenv('SCALPING_MIN_FACTORS', '3'))  # Min confluent factors

# ============================================================================
# LIVE TRADING SAFETY CHECKLIST
# ============================================================================
# Antes de cambiar TRADING_MODE a 'LIVE', verifica:
#
# 1. [ ] API Keys configuradas correctamente
# 2. [ ] API Keys tienen permisos de Futures trading
# 3. [ ] Has probado en TESTNET primero (BINANCE_FUTURES_TESTNET=true)
# 4. [ ] Entiendes los riesgos del trading con apalancamiento
# 5. [ ] Has configurado limites de riesgo apropiados
# 6. [ ] Tienes capital que puedes permitirte perder
# 7. [ ] Has revisado el historial de paper trading (87%+ win rate)
#
# Para activar live trading:
# 1. TRADING_MODE=LIVE
# 2. BINANCE_FUTURES_API_KEY=tu_api_key
# 3. BINANCE_FUTURES_API_SECRET=tu_api_secret
# 4. BINANCE_FUTURES_TESTNET=false (para produccion real)
#
# RECOMENDACION: Empieza con BINANCE_FUTURES_TESTNET=true para validar
# que todo funciona correctamente antes de usar dinero real.
# ============================================================================
