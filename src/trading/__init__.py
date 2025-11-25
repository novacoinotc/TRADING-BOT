"""
Trading Module
Contiene componentes de paper trading y live trading con Binance Futures

Componentes principales:
- PaperTrader: Trading simulado (sin riesgo)
- LiveTrader: Trading real con Binance Futures API
- TradingSystem: Sistema unificado que soporta ambos modos
- BinanceFuturesClient: Cliente para la API de Binance Futures

Modo de uso:
    # Paper Trading (default)
    from src.trading.paper_trader import PaperTrader
    trader = PaperTrader(initial_balance=50000.0)

    # Live Trading
    from src.trading.live_trader import LiveTrader
    trader = LiveTrader(api_key='...', api_secret='...')

    # Sistema Unificado (automatico segun config)
    from src.trading.trading_system import TradingSystem
    system = TradingSystem.from_config()
"""

# Paper Trading (siempre disponible)
from src.trading.paper_trader import PaperTrader
from src.trading.portfolio import Portfolio
from src.trading.position_manager import PositionManager
from src.trading.risk_manager import RiskManager

# Live Trading (disponible si hay dependencias)
try:
    from src.trading.binance_futures_client import (
        BinanceFuturesClient,
        OrderSide,
        OrderType,
        PositionSide,
        MarginType,
        TimeInForce,
        BinanceAPIError,
        BinanceConnectionError
    )
    from src.trading.live_trader import LiveTrader
    from src.trading.live_portfolio import LivePortfolio
    from src.trading.live_position_manager import LivePositionManager
    from src.trading.user_data_stream import UserDataStream, UserDataStreamManager
    from src.trading.trading_system import TradingSystem, TradingMode, get_trader_for_mode

    LIVE_TRADING_AVAILABLE = True
except ImportError as e:
    LIVE_TRADING_AVAILABLE = False
    # Define placeholders
    BinanceFuturesClient = None
    LiveTrader = None
    TradingSystem = None

# Smart Order Router
try:
    from src.trading.smart_order_router import SmartOrderRouter
except ImportError:
    SmartOrderRouter = None

# Dynamic TP Manager
try:
    from src.trading.dynamic_tp_manager import DynamicTPManager
except ImportError:
    DynamicTPManager = None

# Trailing Stop Manager
try:
    from src.trading.trailing_stop_manager import TrailingStopManager
except ImportError:
    TrailingStopManager = None

__all__ = [
    # Paper Trading
    'PaperTrader',
    'Portfolio',
    'PositionManager',
    'RiskManager',

    # Live Trading
    'LiveTrader',
    'LivePortfolio',
    'LivePositionManager',
    'BinanceFuturesClient',
    'UserDataStream',
    'UserDataStreamManager',

    # Sistema Unificado
    'TradingSystem',
    'TradingMode',
    'get_trader_for_mode',

    # Enums y Excepciones
    'OrderSide',
    'OrderType',
    'PositionSide',
    'MarginType',
    'TimeInForce',
    'BinanceAPIError',
    'BinanceConnectionError',

    # Otros
    'SmartOrderRouter',
    'DynamicTPManager',
    'TrailingStopManager',

    # Flag
    'LIVE_TRADING_AVAILABLE',
]
