"""
Trading System - Sistema unificado que soporta Paper Trading y Live Trading
Permite cambiar entre modos sin modificar el resto del codigo

Este modulo funciona como un wrapper/facade que expone la misma interfaz
independientemente del modo de trading seleccionado.
"""

import logging
import asyncio
from typing import Dict, Optional, Union, TYPE_CHECKING

from config import config
from src.trading.paper_trader import PaperTrader

logger = logging.getLogger(__name__)

# Imports condicionales para evitar errores si no se van a usar
try:
    from src.trading.live_trader import LiveTrader, create_live_trader_from_config
    from src.trading.user_data_stream import UserDataStreamManager
    LIVE_TRADING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Live trading modules not available: {e}")
    LIVE_TRADING_AVAILABLE = False


class TradingMode:
    """Modos de trading disponibles"""
    PAPER = 'PAPER'
    LIVE = 'LIVE'


class TradingSystem:
    """
    Sistema de Trading Unificado

    Soporta dos modos:
    - PAPER: Trading simulado (sin riesgo)
    - LIVE: Trading real con Binance Futures

    La interfaz es identica en ambos modos, permitiendo
    intercambiar entre ellos cambiando solo la configuracion.

    Uso:
        # Modo automatico (segun config)
        system = TradingSystem.from_config()

        # Modo explicito
        system = TradingSystem(mode='PAPER')
        system = TradingSystem(mode='LIVE', api_key='...', api_secret='...')

        # Operaciones (misma interfaz en ambos modos)
        result = system.process_signal(pair, signal, price)
        stats = system.get_statistics()
    """

    def __init__(
        self,
        mode: str = TradingMode.PAPER,
        initial_balance: float = 50000.0,
        api_key: str = None,
        api_secret: str = None,
        testnet: bool = False,
        **kwargs
    ):
        """
        Inicializa el sistema de trading

        Args:
            mode: 'PAPER' o 'LIVE'
            initial_balance: Balance inicial (solo para PAPER)
            api_key: API Key de Binance (solo para LIVE)
            api_secret: API Secret de Binance (solo para LIVE)
            testnet: Usar testnet de Binance (solo para LIVE)
            **kwargs: Argumentos adicionales para el trader
        """
        self.mode = mode
        self._trader = None
        self._user_data_stream = None

        if mode == TradingMode.PAPER:
            self._init_paper_trading(initial_balance)
        elif mode == TradingMode.LIVE:
            self._init_live_trading(api_key, api_secret, testnet, initial_balance, **kwargs)
        else:
            raise ValueError(f"Invalid trading mode: {mode}. Use 'PAPER' or 'LIVE'")

        logger.info(f"TradingSystem initialized in {mode} mode")

    def _init_paper_trading(self, initial_balance: float):
        """Inicializa paper trading"""
        self._trader = PaperTrader(initial_balance=initial_balance)
        logger.info(f"Paper Trading initialized with ${initial_balance:,.2f} USDT")

    def _init_live_trading(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool,
        initial_balance: float,
        **kwargs
    ):
        """Inicializa live trading"""
        if not LIVE_TRADING_AVAILABLE:
            raise ImportError(
                "Live trading modules not available. "
                "Please install required dependencies: pip install websockets"
            )

        if not api_key or not api_secret:
            raise ValueError("API key and secret required for LIVE trading")

        self._trader = LiveTrader(
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet,
            initial_reference_balance=initial_balance if initial_balance > 0 else None,
            base_position_size_pct=kwargs.get('base_position_size_pct', 4.0),
            max_drawdown_limit=kwargs.get('max_drawdown_limit', 15.0),
            max_positions=kwargs.get('max_positions', 8),
            max_risk_per_trade_pct=kwargs.get('max_risk_per_trade_pct', 1.5),
            default_leverage=kwargs.get('default_leverage', 10),
            margin_type=kwargs.get('margin_type', 'ISOLATED'),
            use_hedge_mode=kwargs.get('use_hedge_mode', False),
            proxy=kwargs.get('proxy')
        )

        mode_str = "TESTNET" if testnet else "PRODUCTION"
        logger.info(f"Live Trading initialized - {mode_str}")

    @classmethod
    def from_config(cls) -> 'TradingSystem':
        """
        Crea TradingSystem desde la configuracion

        Lee TRADING_MODE de config y crea el sistema apropiado.

        Returns:
            TradingSystem configurado segun config.py
        """
        mode = getattr(config, 'TRADING_MODE', 'PAPER')

        if mode == TradingMode.LIVE:
            # Validar que tenemos las credenciales
            api_key = getattr(config, 'BINANCE_FUTURES_API_KEY', '')
            api_secret = getattr(config, 'BINANCE_FUTURES_API_SECRET', '')

            if not api_key or not api_secret:
                logger.warning(
                    "TRADING_MODE is LIVE but API credentials not configured. "
                    "Falling back to PAPER mode."
                )
                mode = TradingMode.PAPER

        if mode == TradingMode.PAPER:
            return cls(
                mode=TradingMode.PAPER,
                initial_balance=getattr(config, 'PAPER_TRADING_INITIAL_BALANCE', 50000.0)
            )
        else:
            # Build proxy config
            proxy = None
            if getattr(config, 'USE_PROXY', False):
                if config.PROXY_HOST and config.PROXY_PORT:
                    if config.PROXY_USERNAME and config.PROXY_PASSWORD:
                        proxy_url = f"http://{config.PROXY_USERNAME}:{config.PROXY_PASSWORD}@{config.PROXY_HOST}:{config.PROXY_PORT}"
                    else:
                        proxy_url = f"http://{config.PROXY_HOST}:{config.PROXY_PORT}"
                    proxy = {'http': proxy_url, 'https': proxy_url}

            return cls(
                mode=TradingMode.LIVE,
                initial_balance=getattr(config, 'LIVE_TRADING_INITIAL_BALANCE', 0),
                api_key=config.BINANCE_FUTURES_API_KEY,
                api_secret=config.BINANCE_FUTURES_API_SECRET,
                testnet=getattr(config, 'BINANCE_FUTURES_TESTNET', False),
                base_position_size_pct=getattr(config, 'BASE_POSITION_SIZE_PCT', 4.0),
                max_drawdown_limit=getattr(config, 'MAX_DRAWDOWN_LIMIT', 15.0),
                max_positions=getattr(config, 'MAX_POSITIONS', 8),
                max_risk_per_trade_pct=getattr(config, 'MAX_RISK_PER_TRADE_PCT', 1.5),
                default_leverage=getattr(config, 'DEFAULT_LEVERAGE', 10),
                margin_type=getattr(config, 'MARGIN_TYPE', 'ISOLATED'),
                use_hedge_mode=getattr(config, 'USE_HEDGE_MODE', False),
                proxy=proxy
            )

    # ==================== INTERFAZ UNIFICADA ====================

    def process_signal(
        self,
        pair: str,
        signal: Dict,
        current_price: float
    ) -> Optional[Dict]:
        """
        Procesa una senal de trading

        Args:
            pair: Par de trading (ej. BTC/USDT)
            signal: Senal de trading
            current_price: Precio actual

        Returns:
            Resultado del trade o None
        """
        return self._trader.process_signal(pair, signal, current_price)

    def update_position(self, pair: str, current_price: float) -> Optional[Dict]:
        """
        Actualiza posicion existente

        Args:
            pair: Par de trading
            current_price: Precio actual

        Returns:
            Trade cerrado si alcanzo SL/TP, None otherwise
        """
        return self._trader.update_position(pair, current_price)

    def get_balance(self) -> float:
        """Retorna balance USDT disponible"""
        return self._trader.get_balance()

    def get_equity(self) -> float:
        """Retorna equity total (balance + posiciones)"""
        return self._trader.get_equity()

    def get_statistics(self) -> Dict:
        """Retorna estadisticas completas"""
        stats = self._trader.get_statistics()
        stats['trading_mode'] = self.mode
        return stats

    def get_open_positions(self) -> Dict:
        """Retorna posiciones abiertas"""
        return self._trader.get_open_positions()

    def get_closed_trades(self, limit: int = 50) -> list:
        """Retorna historial de trades cerrados"""
        return self._trader.get_closed_trades(limit)

    def force_close_all(self, current_prices: Dict[str, float], reason: str = 'MANUAL'):
        """Cierra todas las posiciones"""
        self._trader.force_close_all(current_prices, reason)

    def enable(self):
        """Habilita el trading"""
        self._trader.enable()

    def disable(self):
        """Deshabilita el trading"""
        self._trader.disable()

    def get_performance_summary(self) -> str:
        """Genera resumen de performance"""
        return self._trader.get_performance_summary()

    # ==================== PROPIEDADES PARA COMPATIBILIDAD ====================

    @property
    def portfolio(self):
        """Portfolio (para compatibilidad con PaperTrader)"""
        return self._trader.portfolio

    @property
    def risk_manager(self):
        """Risk Manager (para compatibilidad)"""
        return self._trader.risk_manager

    @property
    def position_manager(self):
        """Position Manager (para compatibilidad)"""
        return self._trader.position_manager

    @property
    def enabled(self) -> bool:
        """Estado de habilitacion"""
        return self._trader.enabled

    @enabled.setter
    def enabled(self, value: bool):
        """Setter para estado de habilitacion"""
        self._trader.enabled = value

    # ==================== METODOS ESPECIFICOS DE LIVE TRADING ====================

    def is_live(self) -> bool:
        """Retorna True si estamos en modo LIVE"""
        return self.mode == TradingMode.LIVE

    def is_paper(self) -> bool:
        """Retorna True si estamos en modo PAPER"""
        return self.mode == TradingMode.PAPER

    async def start_user_data_stream(self, telegram_notifier=None):
        """
        Inicia el User Data Stream para actualizaciones en tiempo real

        Solo disponible en modo LIVE.

        Args:
            telegram_notifier: Notificador de Telegram (opcional)
        """
        if not self.is_live():
            logger.warning("User Data Stream only available in LIVE mode")
            return

        if not LIVE_TRADING_AVAILABLE:
            logger.warning("User Data Stream modules not available")
            return

        self._user_data_stream = UserDataStreamManager(
            binance_client=self._trader.client,
            portfolio=self._trader.portfolio,
            telegram_notifier=telegram_notifier,
            testnet=self._trader.testnet
        )

        # Start in background
        asyncio.create_task(self._user_data_stream.start())
        logger.info("User Data Stream started")

    async def stop_user_data_stream(self):
        """Detiene el User Data Stream"""
        if self._user_data_stream:
            await self._user_data_stream.stop()
            self._user_data_stream = None
            logger.info("User Data Stream stopped")

    def sync_with_exchange(self):
        """
        Sincroniza estado local con el exchange

        Solo disponible en modo LIVE.
        """
        if self.is_live():
            self._trader.sync_with_binance()
        else:
            logger.debug("Sync not needed in PAPER mode")

    def emergency_close_all(self) -> int:
        """
        Cierre de emergencia de todas las posiciones

        Solo disponible en modo LIVE.

        Returns:
            Numero de posiciones cerradas
        """
        if self.is_live():
            return self._trader.emergency_close_all()
        else:
            logger.warning("Emergency close not available in PAPER mode")
            return 0

    def get_account_info(self) -> Optional[Dict]:
        """
        Obtiene informacion de la cuenta de Binance

        Solo disponible en modo LIVE.
        """
        if self.is_live():
            return self._trader.get_account_info()
        return None

    # ==================== UTILIDADES ====================

    def validate_mode(self) -> bool:
        """
        Valida que el sistema esta funcionando correctamente

        Returns:
            True si todo esta OK
        """
        try:
            # Test basico
            balance = self.get_balance()
            equity = self.get_equity()

            logger.info(f"Mode: {self.mode}")
            logger.info(f"Balance: ${balance:,.2f}")
            logger.info(f"Equity: ${equity:,.2f}")

            # Test especifico de modo
            if self.is_live():
                # Validar conexion con Binance
                return self._trader.client.validate_connection()

            return True

        except Exception as e:
            logger.error(f"Mode validation failed: {e}")
            return False


def get_trader_for_mode(mode: str = None, **kwargs) -> Union[PaperTrader, 'LiveTrader']:
    """
    Factory function para obtener el trader apropiado

    Args:
        mode: 'PAPER' o 'LIVE' (si es None, usa config)
        **kwargs: Argumentos para el trader

    Returns:
        PaperTrader o LiveTrader segun el modo
    """
    if mode is None:
        mode = getattr(config, 'TRADING_MODE', 'PAPER')

    system = TradingSystem.from_config() if mode is None else TradingSystem(mode=mode, **kwargs)
    return system._trader


# Exportar
__all__ = [
    'TradingSystem',
    'TradingMode',
    'get_trader_for_mode'
]
