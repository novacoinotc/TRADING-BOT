"""
Configuración para usar Trading Real con Binance
"""
# CAMBIOS PRINCIPALES v1.0 -> v2.0:
# 1. Paper Trading -> Trading Real con Binance Futures
# 2. IA v1.0 (87% win rate) mantenida
# 3. Integración completa con Binance API v2.0

TRADING_MODE = "REAL"  # Cambiado de "PAPER" a "REAL"
USE_FUTURES_TRADER = True  # Usar futures_trader.py de v2.0

# IMPORTANTE: La IA v1.0 funciona igual, solo cambia la ejecución
# - Señales generadas por RL Agent v1.0
# - Ejecución real a través de futures_trader.py v2.0
