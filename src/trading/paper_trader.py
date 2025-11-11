"""
Paper Trading Engine - Motor principal de trading simulado con $50,000 USDT
Integra Portfolio, Position Manager y Risk Manager
"""
import logging
from typing import Dict, Optional
from src.trading.portfolio import Portfolio
from src.trading.position_manager import PositionManager
from src.trading.risk_manager import RiskManager

logger = logging.getLogger(__name__)


class PaperTrader:
    """
    Motor principal de Paper Trading
    - Gestiona portfolio de $50,000 USDT
    - Ejecuta trades basados en señales
    - Gestiona riesgo automáticamente
    - Optimiza parámetros según performance
    """

    def __init__(self, initial_balance: float = 50000.0):
        """
        Args:
            initial_balance: Balance inicial en USDT (default $50,000)
        """
        # Inicializar componentes
        self.portfolio = Portfolio(initial_balance=initial_balance)
        self.risk_manager = RiskManager(
            portfolio=self.portfolio,
            base_position_size_pct=5.0,  # 5% por posición inicialmente
            max_drawdown_limit=20.0,     # Máximo 20% drawdown
            max_positions=10,             # Máximo 10 posiciones simultáneas
            max_risk_per_trade_pct=2.0   # Máximo 2% riesgo por trade
        )
        self.position_manager = PositionManager(
            portfolio=self.portfolio,
            max_position_size_pct=5.0
        )

        self.enabled = True
        self.stats_cache = {}

        logger.info("🤖 Paper Trading Engine iniciado")
        logger.info(f"💰 Balance inicial: ${initial_balance:,.2f} USDT")

    def process_signal(self, pair: str, signal: Dict, current_price: float) -> Optional[Dict]:
        """
        Procesa señal de trading y ejecuta trade si cumple criterios

        Args:
            pair: Par de trading (ej. BTC/USDT)
            signal: Dict con señal {action, score, confidence, stop_loss, take_profit}
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

        # Verificar si se puede abrir posición
        can_open, reason = self.risk_manager.can_open_position(pair)

        if not can_open:
            logger.debug(f"No se puede abrir posición en {pair}: {reason}")
            return None

        # Calcular tamaño de posición dinámicamente
        position_size_pct = self.risk_manager.calculate_position_size(signal, current_price)

        # Actualizar position manager con nuevo tamaño
        self.position_manager.max_position_size_pct = position_size_pct

        # Ejecutar señal
        result = self.position_manager.process_signal(pair, signal, current_price)

        if result:
            # Log del trade
            if 'status' in result and result['status'] == 'OPEN':
                logger.info(
                    f"📊 Trade abierto: {pair} {action} | "
                    f"Size: {position_size_pct:.1f}% | "
                    f"Score: {signal.get('score', 0):.1f}/10 | "
                    f"Confidence: {signal.get('confidence', 0)}%"
                )
            elif 'status' in result and result['status'] == 'CLOSED':
                pnl = result.get('pnl', 0)
                emoji = "✅" if pnl > 0 else "❌"

                # Verificar que el trade se guardó
                total_closed = len(self.portfolio.closed_trades)
                total_historic = self.portfolio.total_trades

                logger.info(
                    f"{emoji} Trade cerrado: {pair} | "
                    f"P&L: ${pnl:.2f} | "
                    f"Razón: {result.get('reason', 'UNKNOWN')}"
                )
                logger.info(
                    f"✅ Trade cerrado y guardado | "
                    f"Closed trades: {total_closed} | "
                    f"Total histórico: {total_historic}"
                )
                logger.debug(f"📊 Portfolio: {len(self.portfolio.positions)} posiciones abiertas")

        return result

    def update_position(self, pair: str, current_price: float) -> Optional[Dict]:
        """
        Actualiza posición existente y verifica SL/TP

        Args:
            pair: Par de trading
            current_price: Precio actual

        Returns:
            Trade cerrado si se alcanzó SL/TP, None otherwise
        """
        return self.position_manager.update_positions(pair, current_price)

    def _update_existing_positions(self, pair: str, current_price: float) -> Optional[Dict]:
        """Actualiza posiciones existentes con precio actual"""
        if self.portfolio.has_position(pair):
            return self.update_position(pair, current_price)
        return None

    def get_balance(self) -> float:
        """Retorna balance USDT disponible"""
        return self.portfolio.get_available_balance()

    def get_equity(self) -> float:
        """Retorna equity total (balance + posiciones)"""
        return self.portfolio.get_equity()

    def get_statistics(self) -> Dict:
        """
        Retorna estadísticas completas del paper trading

        Returns:
            Dict con todas las métricas
        """
        portfolio_stats = self.portfolio.get_statistics()
        risk_report = self.risk_manager.get_risk_report()

        return {
            **portfolio_stats,
            'risk': risk_report
        }

    def get_open_positions(self) -> Dict:
        """Retorna todas las posiciones abiertas"""
        return self.portfolio.positions.copy()

    def get_closed_trades(self, limit: int = 50) -> list:
        """
        Retorna historial de trades cerrados

        Args:
            limit: Número máximo de trades a retornar

        Returns:
            Lista de trades cerrados
        """
        return self.portfolio.closed_trades[-limit:]

    def force_close_all(self, current_prices: Dict[str, float], reason: str = 'MANUAL'):
        """
        Cierra todas las posiciones forzadamente

        Args:
            current_prices: Dict con precios actuales {pair: price}
            reason: Razón del cierre
        """
        self.position_manager.close_all_positions(current_prices, reason)

    def enable(self):
        """Habilita el paper trading"""
        self.enabled = True
        logger.info("✅ Paper Trading habilitado")

    def disable(self):
        """Deshabilita el paper trading"""
        self.enabled = False
        logger.info("❌ Paper Trading deshabilitado")

    def get_performance_summary(self) -> str:
        """
        Genera resumen de performance para reportes

        Returns:
            String con resumen formateado
        """
        stats = self.get_statistics()

        summary = f"""
📊 **PAPER TRADING SUMMARY**

💰 **Balance**
Inicial: ${stats['initial_balance']:,.2f} USDT
Actual: ${stats['current_balance']:,.2f} USDT
Equity: ${stats['equity']:,.2f} USDT

📈 **Performance**
P&L Neto: ${stats['net_pnl']:,.2f}
ROI: {stats['roi']:+.2f}%
Drawdown Máximo: {stats['max_drawdown']:.2f}%

📊 **Trading**
Total Trades: {stats['total_trades']}
Posiciones Abiertas: {stats['open_positions']}
Ganadores: {stats['winning_trades']} ({stats['win_rate']:.1f}%)
Perdedores: {stats['losing_trades']}

💵 **P&L**
Ganancias Totales: ${stats['total_profit']:,.2f}
Pérdidas Totales: ${stats['total_loss']:,.2f}
Ganancia Promedio: ${stats['avg_win']:,.2f}
Pérdida Promedio: ${stats['avg_loss']:,.2f}
Profit Factor: {stats['profit_factor']:.2f}

📐 **Risk Metrics**
Sharpe Ratio: {stats['sharpe_ratio']:.2f}
Risk Level: {stats['risk']['risk_level']}
Should Reduce Risk: {'Yes' if stats['risk']['should_reduce_risk'] else 'No'}
"""
        return summary.strip()
