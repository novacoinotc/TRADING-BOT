"""
Trade Manager - Gestión inteligente de trades abiertos en tiempo real
Permite a la IA modificar SL/TP, cerrar posiciones anticipadamente, y ajustar estrategia
"""
import logging
import asyncio
from typing import Dict, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


class TradeManager:
    """
    Gestor inteligente de trades abiertos.

    Capacidades:
    - Modificar SL/TP dinámicamente según condiciones de mercado
    - Cerrar posiciones anticipadamente si detecta reversión
    - Trailing stop loss automático
    - Partial take profit (cerrar parte de la posición)
    - Breakeven protection (mover SL a entrada cuando hay +X% ganancia)
    """

    def __init__(
        self,
        position_monitor,
        futures_trader,
        rl_agent,
        ml_system,
        market_analyzer
    ):
        """
        Args:
            position_monitor: Monitor de posiciones
            futures_trader: Trader de Binance Futures
            rl_agent: Agente RL para decisiones
            ml_system: Sistema ML para análisis
            market_analyzer: Analizador de mercado
        """
        self.position_monitor = position_monitor
        self.futures_trader = futures_trader
        self.rl_agent = rl_agent
        self.ml_system = ml_system
        self.market_analyzer = market_analyzer

        self._running = False
        self._check_interval = 30  # Revisar cada 30 segundos

        # Configuración de gestión
        self.config = {
            'breakeven_trigger_pct': 1.5,  # Mover SL a breakeven cuando +1.5%
            'trailing_stop_trigger_pct': 3.0,  # Activar trailing stop cuando +3%
            'trailing_stop_distance_pct': 1.0,  # Distancia del trailing stop
            'partial_tp_trigger_pct': 4.0,  # Cerrar 50% cuando +4%
            'max_adverse_move_pct': -1.5,  # Cerrar si cae más de -1.5% desde máximo
            'reversal_close_confidence': 0.75,  # Cerrar si IA detecta reversión >75%
        }

        # Tracking de máximos/mínimos por posición
        self._position_highs = {}  # {symbol: highest_pnl_pct}
        self._position_lows = {}  # {symbol: lowest_pnl_pct}
        self._partial_closed = set()  # Símbolos donde ya se hizo partial TP

        logger.info("✅ Trade Manager inicializado")

    async def start_monitoring(self):
        """Inicia monitoreo activo de trades"""
        if self._running:
            logger.warning("⚠️ Trade Manager ya está corriendo")
            return

        self._running = True
        logger.info("🟢 Trade Manager: Iniciando monitoreo activo...")

        while self._running:
            try:
                await self._check_all_positions()
                await asyncio.sleep(self._check_interval)

            except asyncio.CancelledError:
                logger.info("⏸️ Trade Manager cancelled")
                break
            except Exception as e:
                logger.error(f"❌ Error en Trade Manager loop: {e}", exc_info=True)
                await asyncio.sleep(self._check_interval)

        self._running = False
        logger.info("🔴 Trade Manager detenido")

    def stop_monitoring(self):
        """Detiene monitoreo"""
        logger.info("🛑 Deteniendo Trade Manager...")
        self._running = False

    async def _check_all_positions(self):
        """Revisa todas las posiciones abiertas y aplica gestión inteligente"""
        positions = self.position_monitor.get_open_positions()

        if not positions:
            return

        logger.debug(f"🔍 Trade Manager: Revisando {len(positions)} posición(es) abierta(s)")

        for symbol, position in positions.items():
            try:
                await self._manage_position(symbol, position)
            except Exception as e:
                logger.error(f"❌ Error gestionando {symbol}: {e}", exc_info=True)

    async def _manage_position(self, symbol: str, position: Dict):
        """
        Aplica gestión inteligente a una posición específica

        Args:
            symbol: Símbolo del trade
            position: Datos de la posición
        """
        # Extraer datos de la posición
        pnl_pct = position.get('unrealized_pnl_pct', 0)
        pnl_usdt = position.get('unrealized_pnl', 0)
        entry_price = position.get('entry_price', 0)
        current_price = position.get('mark_price', 0)
        side = position.get('side', 'UNKNOWN')

        # Actualizar tracking de máximos/mínimos
        if symbol not in self._position_highs:
            self._position_highs[symbol] = pnl_pct
        if symbol not in self._position_lows:
            self._position_lows[symbol] = pnl_pct

        self._position_highs[symbol] = max(self._position_highs[symbol], pnl_pct)
        self._position_lows[symbol] = min(self._position_lows[symbol], pnl_pct)

        highest_pnl = self._position_highs[symbol]

        logger.debug(
            f"📊 {symbol}: P&L={pnl_pct:+.2f}% (Max={highest_pnl:+.2f}%), "
            f"Price=${current_price:.4f}, Side={side}"
        )

        # 1️⃣ Protección de Breakeven
        if pnl_pct >= self.config['breakeven_trigger_pct']:
            await self._set_breakeven(symbol, position)

        # 2️⃣ Trailing Stop
        if highest_pnl >= self.config['trailing_stop_trigger_pct']:
            await self._apply_trailing_stop(symbol, position, highest_pnl)

        # 3️⃣ Partial Take Profit
        if pnl_pct >= self.config['partial_tp_trigger_pct'] and symbol not in self._partial_closed:
            await self._partial_take_profit(symbol, position)

        # 4️⃣ Protección contra movimiento adverso
        drawdown_from_high = highest_pnl - pnl_pct
        if drawdown_from_high >= abs(self.config['max_adverse_move_pct']):
            await self._close_on_adverse_move(symbol, position, drawdown_from_high)

        # 5️⃣ Detección de reversión por IA
        await self._check_reversal_signals(symbol, position)

    async def _set_breakeven(self, symbol: str, position: Dict):
        """Mueve stop loss a precio de entrada (breakeven)"""
        try:
            entry_price = position.get('entry_price', 0)
            side = position.get('side', 'UNKNOWN')

            logger.info(f"🛡️ {symbol}: Moviendo SL a breakeven (${entry_price:.4f})")

            # Calcular nuevo SL en breakeven
            if side == 'LONG':
                new_sl = entry_price * 0.999  # -0.1% para evitar ejecución prematura
            else:  # SHORT
                new_sl = entry_price * 1.001  # +0.1%

            # Modificar SL en Binance
            await self.futures_trader.modify_stop_loss(symbol, new_sl)

            logger.info(f"✅ {symbol}: SL movido a breakeven ${new_sl:.4f}")

        except Exception as e:
            logger.error(f"❌ Error en breakeven para {symbol}: {e}")

    async def _apply_trailing_stop(self, symbol: str, position: Dict, highest_pnl: float):
        """Aplica trailing stop loss dinámico"""
        try:
            current_price = position.get('mark_price', 0)
            side = position.get('side', 'UNKNOWN')
            distance_pct = self.config['trailing_stop_distance_pct']

            # Calcular precio del trailing stop
            if side == 'LONG':
                new_sl = current_price * (1 - distance_pct / 100)
            else:  # SHORT
                new_sl = current_price * (1 + distance_pct / 100)

            logger.info(
                f"📈 {symbol}: Trailing stop activado (Max P&L: {highest_pnl:+.2f}%), "
                f"nuevo SL=${new_sl:.4f}"
            )

            await self.futures_trader.modify_stop_loss(symbol, new_sl)
            logger.info(f"✅ {symbol}: Trailing stop aplicado")

        except Exception as e:
            logger.error(f"❌ Error en trailing stop para {symbol}: {e}")

    async def _partial_take_profit(self, symbol: str, position: Dict):
        """Cierra 50% de la posición para asegurar ganancias"""
        try:
            quantity = abs(position.get('position_amt', 0))
            partial_qty = quantity * 0.5

            logger.info(f"💰 {symbol}: Ejecutando partial TP - Cerrando 50% ({partial_qty:.4f} qty)")

            # Cerrar 50% de la posición
            await self.futures_trader.close_partial_position(symbol, partial_qty)

            self._partial_closed.add(symbol)
            logger.info(f"✅ {symbol}: Partial TP ejecutado exitosamente")

        except Exception as e:
            logger.error(f"❌ Error en partial TP para {symbol}: {e}")

    async def _close_on_adverse_move(self, symbol: str, position: Dict, drawdown: float):
        """Cierra posición por movimiento adverso significativo"""
        try:
            logger.warning(
                f"⚠️ {symbol}: Movimiento adverso detectado desde máximo: {drawdown:.2f}%, "
                f"cerrando posición por protección"
            )

            await self.futures_trader.close_position(symbol, reason='ADVERSE_MOVE')

            # Limpiar tracking
            self._position_highs.pop(symbol, None)
            self._position_lows.pop(symbol, None)
            self._partial_closed.discard(symbol)

            logger.info(f"✅ {symbol}: Posición cerrada por movimiento adverso")

        except Exception as e:
            logger.error(f"❌ Error cerrando {symbol} por movimiento adverso: {e}")

    async def _check_reversal_signals(self, symbol: str, position: Dict):
        """Verifica si la IA detecta señales de reversión fuertes"""
        try:
            # TODO: Implementar detección de reversión cuando el ML system esté listo
            # Por ahora, esta funcionalidad está deshabilitada
            #
            # Para habilitar, necesitarás:
            # 1. Un método rápido de análisis de señales (ej: analyze_pair_fast)
            # 2. Que retorne {'action': 'BUY/SELL/HOLD', 'confidence': 0-100}
            #
            # Ejemplo de implementación futura:
            # pair = symbol.replace('USDT', '/USDT')
            # market_data = await self.market_analyzer.get_quick_signal(pair)
            # if market_data:
            #     signal_action = market_data.get('action', 'HOLD')
            #     confidence = market_data.get('confidence', 0) / 100
            #     side = position.get('side', 'UNKNOWN')
            #     is_reversal = (
            #         (side == 'LONG' and signal_action == 'SELL' and confidence >= 0.75) or
            #         (side == 'SHORT' and signal_action == 'BUY' and confidence >= 0.75)
            #     )
            #     if is_reversal:
            #         await self.futures_trader.close_position(symbol, reason='AI_REVERSAL')

            pass  # Funcionalidad deshabilitada por ahora

        except Exception as e:
            logger.error(f"❌ Error verificando reversión para {symbol}: {e}")

    def get_management_stats(self) -> Dict:
        """Obtiene estadísticas de gestión de trades"""
        return {
            'positions_tracked': len(self._position_highs),
            'partial_tps_executed': len(self._partial_closed),
            'position_highs': self._position_highs.copy(),
            'config': self.config.copy()
        }
