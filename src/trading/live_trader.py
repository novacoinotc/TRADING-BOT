"""
Live Trading Engine - Motor principal de trading REAL con Binance Futures
Reemplaza PaperTrader para operaciones de produccion

ADVERTENCIA: Este modulo ejecuta operaciones REALES con dinero real.
Asegurate de configurar correctamente los parametros de riesgo.
"""

import logging
from typing import Dict, Optional

from src.trading.binance_futures_client import (
    BinanceFuturesClient, MarginType, BinanceAPIError
)
from src.trading.live_portfolio import LivePortfolio
from src.trading.live_position_manager import LivePositionManager
from src.trading.risk_manager import RiskManager

logger = logging.getLogger(__name__)


class LiveTrader:
    """
    Motor principal de Trading REAL con Binance Futures

    Equivalente a PaperTrader pero ejecuta operaciones reales.
    Mantiene compatibilidad con la interfaz de PaperTrader para
    integracion transparente con el resto del sistema.

    ADVERTENCIA: Todas las operaciones son reales y tienen impacto financiero.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = False,
        initial_reference_balance: Optional[float] = None,
        base_position_size_pct: float = 4.0,
        max_drawdown_limit: float = 15.0,
        max_positions: int = 8,
        max_risk_per_trade_pct: float = 1.5,
        default_leverage: int = 10,
        margin_type: str = 'ISOLATED',
        use_hedge_mode: bool = False,
        proxy: Optional[Dict] = None
    ):
        """
        Inicializa el motor de trading real

        Args:
            api_key: API Key de Binance
            api_secret: API Secret de Binance
            testnet: True para usar testnet (recomendado para pruebas)
            initial_reference_balance: Balance de referencia para ROI (opcional, se obtiene de Binance)
            base_position_size_pct: Tamano base por posicion (% del equity)
            max_drawdown_limit: Drawdown maximo permitido
            max_positions: Maximo numero de posiciones simultaneas
            max_risk_per_trade_pct: Maximo riesgo por trade
            default_leverage: Leverage por defecto
            margin_type: 'ISOLATED' o 'CROSSED'
            use_hedge_mode: True para Hedge Mode, False para One-way Mode
            proxy: Configuracion de proxy (opcional)
        """
        # Initialize Binance client
        self.client = BinanceFuturesClient(
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet,
            proxy=proxy
        )

        # Validate connection
        if not self.client.validate_connection():
            raise ConnectionError("Failed to connect to Binance Futures API")

        # Get actual balance if not provided
        if initial_reference_balance is None:
            try:
                usdt_balance = self.client.get_usdt_balance()
                initial_reference_balance = usdt_balance.balance
                logger.info(f"Initial balance from Binance: ${initial_reference_balance:,.2f}")
            except Exception as e:
                logger.error(f"Error getting initial balance: {e}")
                initial_reference_balance = 50000.0  # Fallback

        # Initialize portfolio
        self.portfolio = LivePortfolio(
            binance_client=self.client,
            initial_reference_balance=initial_reference_balance
        )

        # Initialize risk manager (uses same logic as paper trading)
        self.risk_manager = RiskManager(
            portfolio=self.portfolio,
            base_position_size_pct=base_position_size_pct,
            max_drawdown_limit=max_drawdown_limit,
            max_positions=max_positions,
            max_risk_per_trade_pct=max_risk_per_trade_pct
        )

        # Initialize position manager
        margin_type_enum = MarginType.ISOLATED if margin_type == 'ISOLATED' else MarginType.CROSSED
        self.position_manager = LivePositionManager(
            binance_client=self.client,
            portfolio=self.portfolio,
            max_position_size_pct=base_position_size_pct,
            default_leverage=default_leverage,
            default_margin_type=margin_type_enum,
            use_hedge_mode=use_hedge_mode
        )

        self.enabled = True
        self.testnet = testnet
        self.stats_cache = {}

        mode_str = "TESTNET" if testnet else "PRODUCTION"
        logger.info(f"LiveTrader initialized - {mode_str}")
        logger.info(f"Balance: ${initial_reference_balance:,.2f} USDT")
        logger.info(f"Leverage: {default_leverage}x | Margin: {margin_type} | Mode: {'Hedge' if use_hedge_mode else 'One-way'}")

    def process_signal(
        self,
        pair: str,
        signal: Dict,
        current_price: float
    ) -> Optional[Dict]:
        """
        Procesa senal de trading y ejecuta trade REAL si cumple criterios

        Args:
            pair: Par de trading (ej. BTC/USDT)
            signal: Dict con senal {action, score, confidence, stop_loss, take_profit}
            current_price: Precio actual del mercado

        Returns:
            Dict con resultado del trade o None
        """
        if not self.enabled:
            return None

        action = signal.get('action', 'HOLD')

        # Si es HOLD, solo actualizar posiciones existentes
        if action == 'HOLD':
            return self._update_existing_positions(pair, current_price)

        # Verificar si se puede abrir posicion
        can_open, reason = self.risk_manager.can_open_position(pair)

        if not can_open:
            logger.debug(f"Cannot open position in {pair}: {reason}")
            return None

        # Calcular tamano de posicion dinamicamente
        position_size_pct = self.risk_manager.calculate_position_size(signal, current_price)

        # Actualizar position manager con nuevo tamano
        self.position_manager.max_position_size_pct = position_size_pct

        # Ejecutar senal (OPERACION REAL)
        result = self.position_manager.process_signal(pair, signal, current_price)

        if result:
            # Log del trade
            if 'status' in result and result['status'] == 'OPEN':
                leverage = signal.get('leverage', 10)
                logger.info(
                    f"LIVE Trade opened: {pair} {action} | "
                    f"Size: {position_size_pct:.1f}% | "
                    f"Leverage: {leverage}x | "
                    f"Score: {signal.get('score', 0):.1f}/10 | "
                    f"Confidence: {signal.get('confidence', 0)}%"
                )
            elif 'status' in result and result['status'] == 'CLOSED':
                pnl = result.get('pnl', 0)
                emoji = "+" if pnl > 0 else ""

                logger.info(
                    f"LIVE Trade closed: {pair} | "
                    f"P&L: {emoji}${pnl:.2f} | "
                    f"Reason: {result.get('reason', 'UNKNOWN')}"
                )

        return result

    def update_position(self, pair: str, current_price: float) -> Optional[Dict]:
        """
        Actualiza posicion existente y verifica SL/TP

        Args:
            pair: Par de trading
            current_price: Precio actual

        Returns:
            Trade cerrado si se alcanzo SL/TP, None otherwise
        """
        return self.position_manager.update_positions(pair, current_price)

    def _update_existing_positions(self, pair: str, current_price: float) -> Optional[Dict]:
        """Actualiza posiciones existentes con precio actual"""
        if self.portfolio.has_position(pair):
            return self.update_position(pair, current_price)
        return None

    def get_balance(self) -> float:
        """Retorna balance USDT disponible (REAL de Binance)"""
        return self.portfolio.get_available_balance()

    def get_equity(self) -> float:
        """Retorna equity total (balance + posiciones) (REAL)"""
        return self.portfolio.get_equity()

    def get_statistics(self) -> Dict:
        """
        Retorna estadisticas completas del trading

        Returns:
            Dict con todas las metricas
        """
        portfolio_stats = self.portfolio.get_statistics()
        risk_report = self.risk_manager.get_risk_report()

        return {
            **portfolio_stats,
            'risk': risk_report,
            'trading_mode': 'LIVE',
            'testnet': self.testnet
        }

    def get_open_positions(self) -> Dict:
        """Retorna todas las posiciones abiertas"""
        return self.portfolio.positions.copy()

    def get_closed_trades(self, limit: int = 50) -> list:
        """
        Retorna historial de trades cerrados

        Args:
            limit: Numero maximo de trades a retornar

        Returns:
            Lista de trades cerrados
        """
        return self.portfolio.closed_trades[-limit:]

    def force_close_all(self, current_prices: Dict[str, float], reason: str = 'MANUAL'):
        """
        Cierra todas las posiciones forzadamente

        Args:
            current_prices: Dict con precios actuales {pair: price}
            reason: Razon del cierre
        """
        self.position_manager.close_all_positions(current_prices, reason)

    def enable(self):
        """Habilita el trading"""
        self.enabled = True
        logger.info("LIVE Trading enabled")

    def disable(self):
        """Deshabilita el trading (no cierra posiciones existentes)"""
        self.enabled = False
        logger.info("LIVE Trading disabled")

    def sync_with_binance(self):
        """Sincroniza estado local con Binance"""
        self.position_manager.sync_positions_with_binance()

    def get_performance_summary(self) -> str:
        """
        Genera resumen de performance para reportes

        Returns:
            String con resumen formateado
        """
        stats = self.get_statistics()
        mode = "TESTNET" if self.testnet else "PRODUCTION"

        summary = f"""
**LIVE TRADING SUMMARY** ({mode})

**Balance**
Initial: ${stats['initial_balance']:,.2f} USDT
Current: ${stats['current_balance']:,.2f} USDT
Equity: ${stats['equity']:,.2f} USDT

**Performance**
P&L Neto: ${stats['net_pnl']:,.2f}
ROI: {stats['roi']:+.2f}%
Drawdown Maximo: {stats['max_drawdown']:.2f}%

**Trading**
Total Trades: {stats['total_trades']}
Posiciones Abiertas: {stats['open_positions']}
Ganadores: {stats['winning_trades']} ({stats['win_rate']:.1f}%)
Perdedores: {stats['losing_trades']}

**P&L**
Ganancias Totales: ${stats['total_profit']:,.2f}
Perdidas Totales: ${stats['total_loss']:,.2f}
Ganancia Promedio: ${stats['avg_win']:,.2f}
Perdida Promedio: ${stats['avg_loss']:,.2f}
Profit Factor: {stats['profit_factor']:.2f}

**Risk Metrics**
Sharpe Ratio: {stats['sharpe_ratio']:.2f}
Risk Level: {stats['risk']['risk_level']}
Should Reduce Risk: {'Yes' if stats['risk']['should_reduce_risk'] else 'No'}
"""
        return summary.strip()

    def get_account_info(self) -> Dict:
        """Obtiene informacion completa de la cuenta de Binance"""
        return self.client.get_account_info()

    def get_commission_rate(self, symbol: str) -> Dict:
        """Obtiene tasa de comision para un simbolo"""
        return self.client.get_commission_rate(symbol)

    def emergency_close_all(self) -> int:
        """
        Cierre de emergencia de TODAS las posiciones

        Usa la API de Binance directamente para asegurar el cierre.

        Returns:
            Numero de posiciones cerradas
        """
        logger.warning("EMERGENCY CLOSE ALL POSITIONS INITIATED")

        try:
            results = self.client.close_all_positions()
            closed_count = len(results)

            logger.warning(f"Emergency close completed: {closed_count} positions closed")

            # Sync local state
            self.sync_with_binance()

            return closed_count

        except Exception as e:
            logger.error(f"Error in emergency close: {e}")
            return 0


def create_live_trader_from_config(config) -> LiveTrader:
    """
    Crea un LiveTrader desde la configuracion

    Args:
        config: Objeto de configuracion con API keys y parametros

    Returns:
        Instancia de LiveTrader configurada
    """
    # Build proxy config if enabled
    proxy = None
    if getattr(config, 'USE_PROXY', False):
        if config.PROXY_HOST and config.PROXY_PORT:
            if config.PROXY_USERNAME and config.PROXY_PASSWORD:
                proxy_url = f"http://{config.PROXY_USERNAME}:{config.PROXY_PASSWORD}@{config.PROXY_HOST}:{config.PROXY_PORT}"
            else:
                proxy_url = f"http://{config.PROXY_HOST}:{config.PROXY_PORT}"

            proxy = {
                'http': proxy_url,
                'https': proxy_url
            }

    return LiveTrader(
        api_key=config.BINANCE_FUTURES_API_KEY,
        api_secret=config.BINANCE_FUTURES_API_SECRET,
        testnet=getattr(config, 'BINANCE_FUTURES_TESTNET', False),
        initial_reference_balance=getattr(config, 'LIVE_TRADING_INITIAL_BALANCE', None),
        base_position_size_pct=getattr(config, 'BASE_POSITION_SIZE_PCT', 4.0),
        max_drawdown_limit=getattr(config, 'MAX_DRAWDOWN_LIMIT', 15.0),
        max_positions=getattr(config, 'MAX_POSITIONS', 8),
        max_risk_per_trade_pct=getattr(config, 'MAX_RISK_PER_TRADE_PCT', 1.5),
        default_leverage=getattr(config, 'DEFAULT_LEVERAGE', 10),
        margin_type=getattr(config, 'MARGIN_TYPE', 'ISOLATED'),
        use_hedge_mode=getattr(config, 'USE_HEDGE_MODE', False),
        proxy=proxy
    )
