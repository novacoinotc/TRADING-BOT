"""
Position Monitor - Monitorea posiciones abiertas en tiempo real
Detecta cierres automáticos (SL/TP) y actualiza estado
"""

import logging
import time
import asyncio
from typing import Dict, List, Optional, Callable
from datetime import datetime

from .binance_client import BinanceClient, BinanceAPIError

logger = logging.getLogger(__name__)


class PositionMonitor:
    """
    Monitor de posiciones en tiempo real

    Responsabilidades:
    - Consultar posiciones abiertas periódicamente
    - Calcular P&L no realizado en tiempo real
    - Detectar cierres automáticos (SL/TP)
    - Notificar cambios a callbacks
    - Tracking de posiciones cerradas
    """

    def __init__(
        self,
        client: BinanceClient,
        update_interval: int = 5,
        on_position_closed: Optional[Callable] = None
    ):
        """
        Args:
            client: Cliente de Binance
            update_interval: Intervalo de actualización en segundos
            on_position_closed: Callback cuando se cierra una posición
        """
        self.client = client
        self.update_interval = update_interval
        self.on_position_closed = on_position_closed

        # Estado interno
        self._positions = {}  # {symbol: position_data}
        self._last_update = 0
        self._running = False
        self._task = None

        # Tracking de posiciones cerradas
        self._closed_positions_ids = set()  # Para no reprocesar

        logger.info(f"✅ Position Monitor inicializado (update interval: {update_interval}s)")

    def get_open_positions(self) -> Dict[str, Dict]:
        """
        Obtiene todas las posiciones abiertas

        Returns:
            Dict: {symbol: position_data}
        """
        return self._positions.copy()

    def get_position(self, symbol: str) -> Optional[Dict]:
        """
        Obtiene información de una posición específica

        Args:
            symbol: Símbolo

        Returns:
            Dict: Datos de la posición o None
        """
        return self._positions.get(symbol)

    def has_position(self, symbol: str) -> bool:
        """
        Verifica si hay posición abierta para un símbolo

        Args:
            symbol: Símbolo

        Returns:
            bool: True si hay posición abierta
        """
        return symbol in self._positions

    def _fetch_positions_from_binance(self) -> List[Dict]:
        """
        Consulta posiciones desde Binance API

        Returns:
            List: Lista de posiciones abiertas (con positionAmt != 0)
        """
        try:
            all_positions = self.client.get_position_risk()

            # Filtrar solo posiciones abiertas
            open_positions = []
            for pos in all_positions:
                # Debug: Log raw data para ver qué campos vienen de Binance
                logger.debug(f"Raw position data: {pos}")
                logger.debug(f"Position fields available: {list(pos.keys())}")

                # Usar .get() para todos los campos (algunos pueden no existir)
                position_amt = float(pos.get('positionAmt', 0))
                if position_amt != 0:  # Tiene posición abierta
                    # Enriquecer con datos calculados (todos con defaults seguros)
                    entry_price = float(pos.get('entryPrice', 0))
                    mark_price = float(pos.get('markPrice', 0))
                    unrealized_pnl = float(pos.get('unRealizedProfit', 0))

                    # Intentar obtener leverage de diferentes campos posibles
                    leverage = int(pos.get('leverage', pos.get('leverageBracket', 1)))

                    # Calcular P&L%
                    if entry_price > 0:
                        if position_amt > 0:  # LONG
                            pnl_pct = ((mark_price - entry_price) / entry_price) * 100 * leverage
                        else:  # SHORT
                            pnl_pct = ((entry_price - mark_price) / entry_price) * 100 * leverage
                    else:
                        pnl_pct = 0

                    # Determinar margin type (puede ser boolean 'isolated' o string 'marginType')
                    margin_type = pos.get('marginType', 'ISOLATED')
                    if 'isolated' in pos:
                        margin_type = 'ISOLATED' if pos['isolated'] else 'CROSS'

                    enriched = {
                        'symbol': pos.get('symbol', 'UNKNOWN'),
                        'position_amt': position_amt,
                        'entry_price': entry_price,
                        'mark_price': mark_price,
                        'unrealized_pnl': unrealized_pnl,
                        'unrealized_pnl_pct': pnl_pct,
                        'leverage': leverage,
                        'liquidation_price': float(pos.get('liquidationPrice', 0)),
                        'margin_type': margin_type,
                        'position_side': pos.get('positionSide', 'BOTH'),
                        'side': 'LONG' if position_amt > 0 else 'SHORT',
                        'update_time': datetime.now().isoformat(),
                        'raw_data': pos
                    }

                    open_positions.append(enriched)

            return open_positions

        except BinanceAPIError as e:
            logger.error(f"❌ Error fetching positions: [{e.code}] {e.message}")
            return []

        except Exception as e:
            logger.error(f"❌ Unexpected error fetching positions: {e}")
            return []

    def update_positions(self) -> Dict[str, Dict]:
        """
        Actualiza estado de posiciones desde Binance

        Detecta:
        - Nuevas posiciones
        - Posiciones cerradas
        - Cambios en P&L

        Returns:
            Dict: Posiciones actuales
        """
        # Guardar símbolos anteriores
        previous_symbols = set(self._positions.keys())

        # Obtener posiciones actuales de Binance
        current_positions = self._fetch_positions_from_binance()

        # Actualizar diccionario de posiciones
        new_positions = {}
        current_symbols = set()

        for pos in current_positions:
            symbol = pos.get('symbol', 'UNKNOWN')
            current_symbols.add(symbol)
            new_positions[symbol] = pos

            # Log si es nueva posición
            if symbol not in previous_symbols:
                logger.info(
                    f"📈 New position detected: {symbol} {pos.get('side', 'UNKNOWN')}\n"
                    f"   Entry: ${pos.get('entry_price', 0):,.2f}\n"
                    f"   Quantity: {pos.get('position_amt', 0)}\n"
                    f"   Leverage: {pos.get('leverage', 1)}x"
                )

        # Detectar posiciones cerradas
        closed_symbols = previous_symbols - current_symbols

        for symbol in closed_symbols:
            prev_pos = self._positions[symbol]
            logger.info(
                f"🔒 Position closed: {symbol} {prev_pos['side']}\n"
                f"   Entry: ${prev_pos['entry_price']:,.2f}\n"
                f"   Last P&L: ${prev_pos['unrealized_pnl']:+.2f} ({prev_pos['unrealized_pnl_pct']:+.2f}%)"
            )

            # Obtener detalles del cierre desde historial de trades
            closed_info = self._get_close_details(symbol, prev_pos)

            # Llamar callback si existe
            if self.on_position_closed and closed_info:
                try:
                    self.on_position_closed(closed_info)
                except Exception as e:
                    logger.error(f"❌ Error in on_position_closed callback: {e}")

        # Actualizar estado
        self._positions = new_positions
        self._last_update = time.time()

        return self._positions

    def _get_close_details(self, symbol: str, position: Dict) -> Optional[Dict]:
        """
        Obtiene detalles del cierre de una posición desde historial

        Args:
            symbol: Símbolo
            position: Datos de la posición cerrada

        Returns:
            Dict: Información del cierre
        """
        try:
            # Obtener últimos trades para este símbolo
            recent_trades = self.client.get_user_trades(symbol, limit=10)

            if not recent_trades:
                logger.warning(f"⚠️ No recent trades found for {symbol}")
                return None

            # El trade más reciente debería ser el cierre
            last_trade = recent_trades[-1]
            trade_time = last_trade.get('time', int(time.time() * 1000))

            # Verificar que es reciente (últimos 10 segundos)
            current_time = int(time.time() * 1000)
            if (current_time - trade_time) > 10000:  # Más de 10 segundos
                logger.warning(f"⚠️ Last trade for {symbol} is old ({(current_time - trade_time)/1000}s ago)")

            # Obtener P&L realizado desde income history
            try:
                income = self.client.get_income_history(
                    symbol=symbol,
                    income_type='REALIZED_PNL',
                    limit=1
                )

                realized_pnl = float(income[0]['income']) if income else 0
            except:
                realized_pnl = position.get('unrealized_pnl', 0)

            # Determinar razón del cierre (SL, TP, MANUAL, etc.)
            # Esto se puede mejorar consultando las órdenes canceladas/ejecutadas
            reason = 'AUTO_CLOSE'  # Por defecto

            # Construir info del cierre
            close_info = {
                'symbol': symbol,
                'side': position.get('side', 'UNKNOWN'),
                'quantity': abs(position.get('position_amt', 0)),
                'entry_price': position.get('entry_price', 0),
                'exit_price': float(last_trade.get('price', 0)),
                'realized_pnl': realized_pnl,
                'realized_pnl_pct': (realized_pnl / (position.get('entry_price', 1) * abs(position.get('position_amt', 1)))) * 100,
                'leverage': position.get('leverage', 1),
                'reason': reason,
                'trade_id': last_trade.get('id', 0),
                'timestamp': trade_time,
                'commission': float(last_trade.get('commission', 0)),
                'commission_asset': last_trade.get('commissionAsset', 'USDT')
            }

            return close_info

        except Exception as e:
            logger.error(f"❌ Error getting close details for {symbol}: {e}")
            return None

    def get_total_unrealized_pnl(self) -> float:
        """
        Calcula P&L no realizado total de todas las posiciones

        Returns:
            float: P&L total en USDT
        """
        return sum(pos['unrealized_pnl'] for pos in self._positions.values())

    def get_positions_summary(self) -> Dict:
        """
        Obtiene resumen de posiciones

        Returns:
            Dict: Resumen con estadísticas
        """
        positions = list(self._positions.values())

        if not positions:
            return {
                'total_positions': 0,
                'long_positions': 0,
                'short_positions': 0,
                'total_unrealized_pnl': 0,
                'total_unrealized_pnl_pct': 0
            }

        long_positions = [p for p in positions if p.get('side') == 'LONG']
        short_positions = [p for p in positions if p.get('side') == 'SHORT']

        total_pnl = sum(p.get('unrealized_pnl', 0) for p in positions)

        return {
            'total_positions': len(positions),
            'long_positions': len(long_positions),
            'short_positions': len(short_positions),
            'total_unrealized_pnl': total_pnl,
            'positions': positions
        }

    def log_positions_status(self):
        """Loggea estado actual de todas las posiciones"""
        if not self._positions:
            logger.info("📊 No open positions")
            return

        logger.info(f"\n{'='*80}\n📊 OPEN POSITIONS STATUS\n{'='*80}")

        for symbol, pos in self._positions.items():
            emoji = "📈" if pos.get('unrealized_pnl', 0) >= 0 else "📉"
            logger.info(
                f"{emoji} {symbol} - {pos.get('side', 'UNKNOWN')}\n"
                f"   Entry: ${pos.get('entry_price', 0):,.2f} | "
                f"Mark: ${pos.get('mark_price', 0):,.2f} | "
                f"Leverage: {pos.get('leverage', 1)}x\n"
                f"   P&L: ${pos.get('unrealized_pnl', 0):+.2f} ({pos.get('unrealized_pnl_pct', 0):+.2f}%)\n"
                f"   Liquidation: ${pos.get('liquidation_price', 0):,.2f}"
            )

        summary = self.get_positions_summary()
        logger.info(
            f"\n{'='*80}\n"
            f"Total: {summary['total_positions']} positions | "
            f"LONG: {summary['long_positions']} | "
            f"SHORT: {summary['short_positions']}\n"
            f"Total Unrealized P&L: ${summary['total_unrealized_pnl']:+.2f}\n"
            f"{'='*80}\n"
        )

    async def start_monitoring(self):
        """
        Inicia monitoreo continuo de posiciones

        Runs in background loop
        """
        if self._running:
            logger.warning("⚠️ Position monitor already running")
            return

        self._running = True
        logger.info("🟢 Starting position monitor...")

        # Cargar posiciones existentes ANTES del loop (crítico si reinició)
        try:
            initial_positions = self.update_positions()
            num_open = len([p for p in initial_positions.values() if float(p.get('positionAmt', 0)) != 0])
            if num_open > 0:
                logger.info(f"🔄 Position monitor: Encontradas {num_open} posiciones abiertas al iniciar")
            else:
                logger.info("📊 Position monitor: Sin posiciones abiertas")
        except Exception as init_error:
            logger.error(f"⚠️ Error cargando posiciones iniciales: {init_error}")

        while self._running:
            try:
                # Actualizar posiciones
                self.update_positions()

                # Log status periódicamente (cada 60 segundos)
                if int(time.time()) % 60 == 0:
                    self.log_positions_status()

                # Esperar intervalo
                await asyncio.sleep(self.update_interval)

            except asyncio.CancelledError:
                logger.info("⏸️ Position monitor cancelled")
                break

            except Exception as e:
                logger.error(f"❌ Error in position monitor loop: {e}", exc_info=True)
                await asyncio.sleep(self.update_interval)

        self._running = False
        logger.info("🔴 Position monitor stopped")

    def stop_monitoring(self):
        """Detiene monitoreo"""
        logger.info("🛑 Stopping position monitor...")
        self._running = False

        if self._task:
            self._task.cancel()

    def start_background_monitoring(self):
        """Inicia monitoreo en background task"""
        if self._task and not self._task.done():
            logger.warning("⚠️ Background monitoring already running")
            return

        loop = asyncio.get_event_loop()
        self._task = loop.create_task(self.start_monitoring())
        logger.info("🚀 Background position monitoring started")

        return self._task
