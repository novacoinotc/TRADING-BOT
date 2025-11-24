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
        update_interval: int = 10,  # Update interval aumentado de 5s a 10s para evitar rate limiting de Binance
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

        # Historial de trades cerrados (para el dashboard y Trade Manager learning)
        self.closed_trades: Dict[str, Dict] = {}  # {symbol: {pnl_pct, reason, timestamp}}
        self.max_closed_trades = 100  # Mantener últimos 100

        # NUEVO: Lista completa de trades cerrados para ML retraining
        self.closed_trades_history: List[Dict] = []  # Lista completa de todos los trades cerrados
        self.max_history_size = 500  # Mantener últimos 500 trades para ML

        # Referencia opcional a autonomy_controller (para verificar test_mode_active)
        self.autonomy_controller = None

        # Cache para retry resiliente
        self._last_positions_cache = []

        logger.info(f"✅ Position Monitor inicializado (update interval: {update_interval}s)")

    def get_open_positions(self) -> Dict[str, Dict]:
        """
        Obtiene todas las posiciones abiertas

        Returns:
            Dict: {symbol: position_data} - SIEMPRE devuelve dict, nunca None o string
        """
        try:
            if self._positions is None:
                logger.warning("⚠️ _positions es None, inicializando dict vacío")
                self._positions = {}

            if not isinstance(self._positions, dict):
                logger.error(f"❌ _positions es tipo inesperado: {type(self._positions)}, reiniciando")
                self._positions = {}

            return self._positions.copy()

        except Exception as e:
            logger.error(f"❌ Error en get_open_positions(): {e}", exc_info=True)
            return {}  # SIEMPRE devolver dict vacío en caso de error

    def _fetch_positions_with_retry(self, max_retries=3):
        """
        Fetch positions con retry y exponential backoff

        Args:
            max_retries: Número máximo de reintentos

        Returns:
            Lista de positions o cache previo si todo falla
        """
        import time

        for attempt in range(max_retries):
            try:
                positions = self.client.get_position_risk()

                # Validar que no está vacío si teníamos posiciones antes
                if not positions and hasattr(self, '_last_positions_cache') and self._last_positions_cache:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2  # 2s, 4s, 6s
                        logger.warning(
                            f"⚠️ Fetch returned empty positions, retry {attempt + 1}/{max_retries} "
                            f"in {wait_time}s"
                        )
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(
                            f"❌ Fetch failed after {max_retries} retries, "
                            f"usando cache previo"
                        )
                        return self._last_positions_cache

                # Guardar cache exitoso
                if positions:
                    self._last_positions_cache = positions

                return positions

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    logger.warning(
                        f"⚠️ Error fetching positions: {e}, "
                        f"retry {attempt + 1}/{max_retries} in {wait_time}s"
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"❌ Failed to fetch positions after {max_retries} retries: {e}")
                    # Retornar cache previo si existe
                    if hasattr(self, '_last_positions_cache'):
                        logger.warning("⚠️ Usando cache previo de positions")
                        return self._last_positions_cache
                    return []

        return []

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

    def record_closed_trade(self, symbol: str, pnl_pct: float, reason: str):
        """
        Guarda información de un trade cerrado para que Trade Manager pueda evaluarlo

        Args:
            symbol: Símbolo del trade
            pnl_pct: P&L en porcentaje
            reason: Razón del cierre (TP_HIT, SL_HIT, MANUAL, etc.)
        """
        try:
            from datetime import datetime

            self.closed_trades[symbol] = {
                'pnl_pct': pnl_pct,
                'reason': reason,
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol
            }

            # Mantener solo últimos N trades
            if len(self.closed_trades) > self.max_closed_trades:
                # Eliminar el más antiguo
                oldest = min(self.closed_trades.items(),
                            key=lambda x: x[1]['timestamp'])
                del self.closed_trades[oldest[0]]

            logger.debug(f"📝 Trade cerrado registrado: {symbol} ({reason}, {pnl_pct:+.2f}%)")

        except Exception as e:
            logger.error(f"Error recording closed trade: {e}", exc_info=True)

    def _get_real_leverage(self, symbol: str, position_data: Dict) -> int:
        """
        Obtiene el leverage real para un símbolo.
        Si position_data tiene leverage=1, consulta account info para verificar.

        Args:
            symbol: Símbolo
            position_data: Datos de posición desde positionRisk

        Returns:
            int: Leverage real
        """
        # Intentar obtener leverage del position data primero
        leverage_from_pos = int(position_data.get('leverage', 1))

        # 🔍 Logging de debugging
        logger.debug(f"🔍 Leverage detection for {symbol}: position_data={leverage_from_pos}x")

        # Si es 1, podría ser incorrecto - verificar con account info
        if leverage_from_pos == 1:
            try:
                # Consultar account info que tiene leverage por símbolo
                account_info = self.client.get_account_info()
                if 'positions' in account_info:
                    for pos in account_info['positions']:
                        if pos.get('symbol') == symbol:
                            leverage_from_account = int(pos.get('leverage', 1))
                            if leverage_from_account > 1:
                                logger.info(f"✅ Leverage corregido para {symbol}: {leverage_from_account}x (positionRisk mostraba 1x)")
                                return leverage_from_account
                            else:
                                logger.debug(f"  → account_info también muestra 1x para {symbol}")

            except Exception as e:
                logger.warning(f"⚠️ No se pudo verificar leverage real para {symbol}: {e}")

        return leverage_from_pos

    def _fetch_positions_from_binance(self) -> List[Dict]:
        """
        Consulta posiciones desde Binance API

        Returns:
            List: Lista de posiciones abiertas (con positionAmt != 0)
        """
        try:
            all_positions = self._fetch_positions_with_retry()

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

                    # Obtener leverage real (verifica con account info si es necesario)
                    leverage = self._get_real_leverage(pos.get('symbol', 'UNKNOWN'), pos)

                    # Calcular ROE% (Return on Equity) - NO multiplicar P&L por leverage
                    # El unrealized_pnl ya viene correcto de Binance en USDT
                    # ROE% = (P&L / Margen Inicial) * 100
                    initial_margin = (entry_price * abs(position_amt)) / leverage if leverage > 0 else entry_price * abs(position_amt)
                    roe_pct = (unrealized_pnl / initial_margin) * 100 if initial_margin > 0 else 0

                    # Determinar margin type (puede ser boolean 'isolated' o string 'marginType')
                    margin_type = pos.get('marginType', 'ISOLATED')
                    if 'isolated' in pos:
                        margin_type = 'ISOLATED' if pos['isolated'] else 'CROSS'

                    enriched = {
                        'symbol': pos.get('symbol', 'UNKNOWN'),
                        'position_amt': position_amt,
                        'entry_price': entry_price,
                        'mark_price': mark_price,
                        'unrealized_pnl': unrealized_pnl,  # P&L en USDT (absoluto, de Binance)
                        'unrealized_pnl_pct': roe_pct,  # ROE% (retorno sobre margen inicial)
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

        # 🔧 FIX CRÍTICO: Si hay error en fetch (retorna lista vacía pero había posiciones antes),
        # NO actualizar self._positions para evitar borrar posiciones válidas
        if len(current_positions) == 0 and len(previous_symbols) > 0:
            logger.warning(
                f"⚠️ Fetch returned 0 positions but had {len(previous_symbols)} before. "
                f"Possible API error. Keeping previous positions to avoid data loss."
            )
            return self._positions  # Retornar posiciones anteriores sin modificar

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

            # CRÍTICO: Guardar en historial para el dashboard
            # NO guardar si test_mode está activo (test_mode ya guarda con los datos correctos)
            test_mode_active = (self.autonomy_controller and
                               hasattr(self.autonomy_controller, 'test_mode_active') and
                               self.autonomy_controller.test_mode_active)

            if closed_info and not test_mode_active:
                # Registrar trade cerrado para Trade Manager learning
                realized_pnl_pct = closed_info.get('realized_pnl_pct', 0)
                reason = closed_info.get('reason', 'UNKNOWN')
                self.record_closed_trade(symbol, realized_pnl_pct, reason)
                logger.info(f"💾 Trade cerrado registrado para learning: {symbol} ({reason}, {realized_pnl_pct:+.2f}%)")

                # NUEVO: Agregar a historial completo para ML retraining
                trade_data = {
                    'symbol': symbol,
                    'pnl_pct': realized_pnl_pct,
                    'pnl_usdt': closed_info.get('realized_pnl', 0),
                    'entry_price': closed_info.get('entry_price', 0),
                    'exit_price': closed_info.get('exit_price', 0),
                    'leverage': closed_info.get('leverage', 1),
                    'side': closed_info.get('side', 'UNKNOWN'),
                    'reason': reason,
                    'timestamp': datetime.now().isoformat(),
                    'quantity': closed_info.get('quantity', 0)
                }
                self.closed_trades_history.append(trade_data)

                # Mantener solo últimos N trades
                if len(self.closed_trades_history) > self.max_history_size:
                    self.closed_trades_history = self.closed_trades_history[-self.max_history_size:]

                logger.info(f"📊 Trade agregado a historial ML: {len(self.closed_trades_history)} trades totales")

                # NUEVO: Llamar a process_trade_closure si autonomy_controller existe
                if self.autonomy_controller and hasattr(self.autonomy_controller, 'process_trade_closure'):
                    try:
                        import asyncio
                        asyncio.create_task(
                            self.autonomy_controller.process_trade_closure(trade_data)
                        )
                        logger.info(f"✅ process_trade_closure llamado para {symbol}")
                    except Exception as e:
                        logger.error(f"❌ Error calling process_trade_closure: {e}")

            elif test_mode_active:
                logger.debug(f"⏭️ No guardando trade (Test Mode activo, ya lo guardó test_mode)")

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
            # 🔧 FIX: Inicializar variables ANTES del try para evitar UnboundLocalError
            import time
            from datetime import datetime
            trade_time = int(time.time() * 1000)  # Default timestamp
            exit_price = position.get('mark_price', 0)  # Default exit price
            last_trade = {}  # Default empty dict
            reason = 'AUTO_CLOSE'  # Default reason

            # Obtener últimos trades para este símbolo
            recent_trades = self.client.get_user_trades(symbol, limit=10)

            if not recent_trades:
                logger.warning(f"⚠️ No recent trades found for {symbol}")
                return None

            # El trade más reciente debería ser el cierre
            last_trade = recent_trades[-1]
            trade_time = last_trade.get('time', int(time.time() * 1000))

            # Obtener exit price (puede estar en 'price' o 'avgPrice')
            exit_price = float(last_trade.get('price', last_trade.get('avgPrice', 0)))
            logger.debug(f"Exit price del trade: ${exit_price:,.2f}")

            # Verificar que es reciente (últimos 10 segundos)
            current_time = int(time.time() * 1000)
            if (current_time - trade_time) > 10000:  # Más de 10 segundos
                logger.warning(f"⚠️ Last trade for {symbol} is old ({(current_time - trade_time)/1000}s ago)")

            # Obtener P&L realizado desde income history
            # CRÍTICO: Usar startTime/endTime para obtener el income correcto
            try:
                # Buscar REALIZED_PNL en los últimos 60 segundos
                end_time = int(time.time() * 1000)
                start_time = end_time - 60000  # Últimos 60 segundos

                income = self.client.get_income_history(
                    symbol=symbol,
                    income_type='REALIZED_PNL',
                    limit=10,
                    start_time=start_time,
                    end_time=end_time
                )

                # Encontrar el income más reciente para este símbolo
                realized_pnl = 0
                if income:
                    # Ordenar por timestamp descendente y tomar el más reciente
                    income_sorted = sorted(income, key=lambda x: x.get('time', 0), reverse=True)
                    if income_sorted:
                        realized_pnl = float(income_sorted[0]['income'])
                        logger.debug(f"P&L obtenido de income history: ${realized_pnl:+.2f}")

                # Si no se encontró, calcular desde unrealized_pnl
                if realized_pnl == 0 and position.get('unrealized_pnl', 0) != 0:
                    realized_pnl = position.get('unrealized_pnl', 0)
                    logger.debug(f"P&L calculado desde unrealized: ${realized_pnl:+.2f}")

            except Exception as e:
                logger.warning(f"⚠️ Error obteniendo income history: {e}")
                # Fallback: usar unrealized_pnl del position
                realized_pnl = position.get('unrealized_pnl', 0)
                logger.debug(f"P&L fallback desde unrealized: ${realized_pnl:+.2f}")

            # Determinar razón del cierre consultando órdenes recientes
            try:
                # Consultar últimas 10 órdenes del símbolo
                recent_orders = self.client.get_all_orders(symbol=symbol, limit=10)

                # Buscar órdenes FILLED recientes (últimos 30 segundos) con reduceOnly=True
                now = int(time.time() * 1000)
                recent_filled = [
                    o for o in recent_orders
                    if o['status'] == 'FILLED' and (now - o['updateTime']) < 30000
                ]

                # 🔧 FIX CRÍTICO: Ordenar por updateTime descendente (más reciente primero)
                recent_filled = sorted(recent_filled, key=lambda x: x.get('updateTime', 0), reverse=True)

                # Filtrar solo órdenes que cierran posición (reduceOnly=True o side opuesto)
                closing_orders = [
                    o for o in recent_filled
                    if o.get('reduceOnly', False) or o.get('type') in ['STOP_MARKET', 'TAKE_PROFIT_MARKET']
                ]

                logger.debug(f"🔍 Órdenes de cierre encontradas: {len(closing_orders)}")
                for order in closing_orders:
                    logger.debug(f"   - {order.get('type')}: {order.get('orderId')} (reduceOnly={order.get('reduceOnly')})")

                # Detectar tipo de cierre basado en el tipo de orden MÁS RECIENTE
                for order in closing_orders:
                    order_type = order.get('type', '')
                    order_id = order.get('orderId')

                    if order_type == 'STOP_MARKET':
                        reason = 'STOP_LOSS'
                        logger.info(f"🛑 Cierre detectado: STOP_LOSS (order_id={order_id})")
                        break
                    elif order_type == 'TAKE_PROFIT_MARKET':
                        reason = 'TAKE_PROFIT'
                        logger.info(f"🎯 Cierre detectado: TAKE_PROFIT (order_id={order_id})")
                        break
                    elif order_type == 'MARKET' and order.get('reduceOnly'):
                        reason = 'MANUAL'
                        logger.info(f"👤 Cierre detectado: MANUAL (order_id={order_id})")
                        break
                    elif order_type == 'LIMIT' and order.get('reduceOnly'):
                        reason = 'LIMIT_CLOSE'
                        logger.info(f"📊 Cierre detectado: LIMIT ORDER (order_id={order_id})")
                        break
                else:
                    # Si no encontramos órdenes de cierre específicas, es AUTO_CLOSE
                    logger.debug(f"ℹ️ No se encontró orden de cierre específica, usando AUTO_CLOSE")

            except Exception as e:
                logger.warning(f"⚠️ No se pudo determinar razón de cierre para {symbol}: {e}")
                reason = 'AUTO_CLOSE'

            # Construir info del cierre
            entry_price = position.get('entry_price', 1)
            quantity = abs(position.get('position_amt', 1))
            leverage = position.get('leverage', 1)

            # Calcular ROE% correctamente (P&L / Margen Inicial)
            initial_margin = (entry_price * quantity) / leverage if leverage > 0 else entry_price * quantity
            roe_pct = (realized_pnl / initial_margin) * 100 if initial_margin > 0 else 0

            close_info = {
                'symbol': symbol,
                'side': position.get('side', 'UNKNOWN'),
                'quantity': quantity,
                'entry_price': entry_price,
                'exit_price': exit_price,  # Ya calculado arriba con fallbacks
                'realized_pnl': realized_pnl,  # P&L en USDT (absoluto, de Binance)
                'realized_pnl_pct': roe_pct,  # ROE% (retorno sobre margen inicial)
                'leverage': leverage,
                'reason': reason,
                'trade_id': last_trade.get('id', 0),
                'timestamp': trade_time,
                'commission': float(last_trade.get('commission', 0)),
                'commission_asset': last_trade.get('commissionAsset', 'USDT')
            }

            logger.info(
                f"📊 Close info calculado: "
                f"Entry=${entry_price:.2f}, Exit=${exit_price:.2f}, "
                f"P&L=${realized_pnl:+.2f} ({roe_pct:+.2f}%)"
            )

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
                positions = self.update_positions()

                # Contar posiciones abiertas
                open_positions = {k: v for k, v in positions.items() if float(v.get('position_amt', 0)) != 0}

                if open_positions:
                    # Log compacto cada ciclo (10s)
                    logger.info(f"📊 Position Monitor: {len(open_positions)} posiciones abiertas")
                    for symbol, pos in open_positions.items():
                        pnl = float(pos.get('unrealized_pnl', 0))
                        entry = float(pos.get('entry_price', 0))
                        mark = float(pos.get('mark_price', 0))

                        # Calcular PNL%
                        position_amt = abs(float(pos.get('position_amt', 0)))
                        notional = position_amt * entry
                        pnl_pct = (pnl / notional * 100) if notional > 0 else 0

                        emoji = "📈" if pnl >= 0 else "📉"
                        logger.info(f"   {emoji} {symbol}: {pnl:+.2f} USDT ({pnl_pct:+.2f}%) | Entry: ${entry:.4f} | Mark: ${mark:.4f}")

                # Log detallado cada 60s
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
