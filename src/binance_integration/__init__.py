"""
Binance Futures Integration v2.0
Integración completa con Binance USD-M Futures API
"""

from .binance_client import BinanceClient
from .futures_trader import FuturesTrader
from .position_monitor import PositionMonitor
from .utils import sign_request, validate_price, validate_quantity

__all__ = [
    'BinanceClient',
    'FuturesTrader',
    'PositionMonitor',
    'sign_request',
    'validate_price',
    'validate_quantity',
]

__version__ = '2.0.0'
