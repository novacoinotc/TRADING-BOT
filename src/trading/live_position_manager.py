"""
Live Position Manager - Gestiona apertura/cierre de posiciones REALES en Binance Futures
Reemplaza PositionManager de paper trading para modo produccion
"""

import logging
import math
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.trading.binance_futures_client import BinanceFuturesClient
    from src.trading.live_portfolio import LivePortfolio

from src.trading.binance_futures_client import (
    OrderSide, OrderType, PositionSide, MarginType, TimeInForce,
    BinanceAPIError
)

logger = logging.getLogger(__name__)


class LivePositionManager:
    """
    Gestiona el ciclo de vida de las posiciones REALES:
    - Apertura de posiciones con ordenes de mercado
    - Configuracion de Stop Loss y Take Profit
    - Cierre por SL/TP o senal contraria
    - Manejo de errores de la API

    IMPORTANTE: Este manager ejecuta ordenes REALES en Binance.
    Todas las operaciones tienen impacto financiero real.
    """

    def __init__(
        self,
        binance_client: 'BinanceFuturesClient',
        portfolio: 'LivePortfolio',
        max_position_size_pct: float = 5.0,
        default_leverage: int = 10,
        default_margin_type: MarginType = MarginType.ISOLATED,
        use_hedge_mode: bool = False
    ):
        """
        Args:
            binance_client: Cliente de Binance Futures
            portfolio: LivePortfolio para tracking
            max_position_size_pct: Maximo % del portfolio por posicion
            default_leverage: Leverage por defecto
            default_margin_type: Tipo de margen (ISOLATED recomendado)
            use_hedge_mode: True para Hedge Mode (LONG/SHORT), False para One-way
        """
        self.client = binance_client
        self.portfolio = portfolio
        self.max_position_size_pct = max_position_size_pct
        self.default_leverage = default_leverage
        self.default_margin_type = default_margin_type
        self.use_hedge_mode = use_hedge_mode

        # Track active stop orders
        self.stop_orders: Dict[str, Dict] = {}  # {pair: {sl_order_id, tp_orders}}

        # Initialize position mode
        self._setup_position_mode()

        logger.info(
            f"LivePositionManager initialized - "
            f"Leverage: {default_leverage}x, "
            f"Margin: {default_margin_type.value}, "
            f"Mode: {'Hedge' if use_hedge_mode else 'One-way'}"
        )

    def _setup_position_mode(self):
        """Configura el modo de posicion en Binance"""
        try:
            self.client.set_position_mode(self.use_hedge_mode)
        except Exception as e:
            logger.error(f"Error setting position mode: {e}")

    def _get_position_side(self, side: str) -> PositionSide:
        """Determina el position side basado en el modo"""
        if not self.use_hedge_mode:
            return PositionSide.BOTH
        return PositionSide.LONG if side == 'BUY' else PositionSide.SHORT

    def _pair_to_symbol(self, pair: str) -> str:
        """Convierte BTC/USDT -> BTCUSDT (usando mapeo de símbolos del cliente)"""
        # Usar el método del cliente que maneja símbolos especiales como 1000SHIB
        return self.client.convert_pair_format(pair)

    def process_signal(
        self,
        pair: str,
        signal: Dict,
        current_price: float
    ) -> Optional[Dict]:
        """
        Procesa una senal de trading y ejecuta la accion correspondiente

        Args:
            pair: Par de trading (ej. BTC/USDT)
            signal: Dict con la senal (action, score, stop_loss, take_profit, etc.)
            current_price: Precio actual del mercado

        Returns:
            Trade ejecutado o None
        """
        action = signal.get('action')

        if action == 'HOLD':
            # Verificar si hay posicion abierta que necesite actualizacion
            if self.portfolio.has_position(pair):
                return self._check_exit_conditions(pair, current_price)
            return None

        # Si hay posicion abierta del lado contrario, cerrarla
        if self.portfolio.has_position(pair):
            existing_position = self.portfolio.get_position(pair)

            # Si la senal es contraria a la posicion actual, cerrar
            if existing_position['side'] != action:
                logger.info(f"Reverse signal detected for {pair}: {existing_position['side']} -> {action}")

                # Cerrar posicion existente
                closed_trade = self._close_position(pair, current_price, reason='SIGNAL_REVERSE')

                # Abrir nueva posicion del lado contrario
                if closed_trade:
                    return self._open_new_position(pair, signal, current_price)

            else:
                # Ya tenemos posicion del mismo lado, no hacer nada
                logger.debug(f"Already have {action} position in {pair}, ignoring signal")
                return None

        # No hay posicion, abrir nueva
        return self._open_new_position(pair, signal, current_price)

    def _open_new_position(
        self,
        pair: str,
        signal: Dict,
        current_price: float
    ) -> Optional[Dict]:
        """
        Abre nueva posicion REAL en Binance Futures

        Args:
            pair: Par de trading
            signal: Senal de trading
            current_price: Precio actual

        Returns:
            Posicion abierta o None si falla
        """
        action = signal['action']
        stop_loss = signal.get('stop_loss', 0)
        take_profit = signal.get('take_profit', {})
        symbol = self._pair_to_symbol(pair)

        # Obtener parametros de futuros de la senal
        leverage = signal.get('leverage', self.default_leverage)
        trade_type = signal.get('trade_type', 'FUTURES')

        # Validate price
        if current_price is None or current_price <= 0 or math.isnan(current_price) or math.isinf(current_price):
            logger.error(f"Invalid price for {pair}: {current_price}")
            return None

        # Calculate position size
        available_balance = self.portfolio.get_available_balance()
        equity = self.portfolio.get_equity()
        max_position_value = (equity * self.max_position_size_pct) / 100

        # Use the smaller of available balance and max position value
        position_value = min(available_balance * 0.95, max_position_value)  # 95% to leave margin for fees

        # Calculate collateral required (position_value / leverage)
        collateral_required = position_value / leverage

        if collateral_required < 10:  # Minimum $10 per trade
            logger.warning(f"Insufficient collateral for {pair}: ${collateral_required:.2f}")
            return None

        # Calculate quantity
        quantity = position_value / current_price

        if quantity <= 0 or math.isnan(quantity) or math.isinf(quantity):
            logger.error(f"Invalid quantity calculated for {pair}: {quantity}")
            return None

        try:
            # 1. Set leverage for the symbol
            self.client.set_leverage(symbol, leverage)

            # 2. Set margin type
            self.client.set_margin_type(symbol, self.default_margin_type)

            # 3. Determine order side and position side
            order_side = OrderSide.BUY if action == 'BUY' else OrderSide.SELL
            position_side = self._get_position_side(action)

            # 4. Place market order
            logger.info(f"Opening REAL position: {pair} {action} qty={quantity:.6f} leverage={leverage}x")

            order_result = self.client.place_market_order(
                symbol=symbol,
                side=order_side,
                quantity=quantity,
                position_side=position_side
            )

            if order_result.status not in ['FILLED', 'NEW', 'PARTIALLY_FILLED']:
                logger.error(f"Order not filled: {order_result.status}")
                return None

            # 5. Get actual execution price
            actual_entry_price = order_result.avg_price if order_result.avg_price > 0 else current_price
            actual_quantity = order_result.executed_qty if order_result.executed_qty > 0 else quantity

            # 6. Register position in portfolio tracking
            position = self.portfolio.register_opened_position(
                pair=pair,
                side=action,
                entry_price=actual_entry_price,
                quantity=actual_quantity,
                stop_loss=stop_loss,
                take_profit=take_profit,
                leverage=leverage,
                order_result=order_result.raw_response
            )

            # 7. Place stop loss and take profit orders
            self._place_protective_orders(
                symbol=symbol,
                pair=pair,
                side=action,
                quantity=actual_quantity,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_side=position_side
            )

            logger.info(
                f"REAL position opened: {pair} {action} @ ${actual_entry_price:.2f} "
                f"qty={actual_quantity:.6f} leverage={leverage}x"
            )

            return position

        except BinanceAPIError as e:
            logger.error(f"Binance API error opening position: [{e.code}] {e.message}")
            return None
        except Exception as e:
            logger.error(f"Error opening position for {pair}: {e}")
            return None

    def _place_protective_orders(
        self,
        symbol: str,
        pair: str,
        side: str,
        quantity: float,
        stop_loss: float,
        take_profit: Dict,
        position_side: PositionSide
    ):
        """
        Coloca ordenes de Stop Loss y Take Profit

        Args:
            symbol: Simbolo de Binance
            pair: Par de trading
            side: 'BUY' o 'SELL'
            quantity: Cantidad
            stop_loss: Precio de stop loss
            take_profit: Dict con tp1, tp2, tp3
            position_side: Position side
        """
        self.stop_orders[pair] = {'sl': None, 'tp': []}

        # Determine close side (opposite of entry)
        close_side = OrderSide.SELL if side == 'BUY' else OrderSide.BUY

        # 1. Place Stop Loss (STOP_MARKET)
        if stop_loss and stop_loss > 0:
            try:
                sl_order = self.client.place_stop_market_order(
                    symbol=symbol,
                    side=close_side,
                    quantity=quantity,
                    stop_price=stop_loss,
                    position_side=position_side,
                    reduce_only=True
                )
                self.stop_orders[pair]['sl'] = sl_order.order_id
                logger.info(f"Stop Loss placed for {pair} @ ${stop_loss:.2f}")

            except Exception as e:
                logger.error(f"Error placing stop loss for {pair}: {e}")

        # 2. Place Take Profit orders (TAKE_PROFIT_MARKET)
        # For multiple TPs, we need to split the quantity
        tp_levels = []
        if take_profit:
            if 'tp1' in take_profit and take_profit['tp1']:
                tp_levels.append(('TP1', take_profit['tp1'], 0.5))  # 50% at TP1
            if 'tp2' in take_profit and take_profit['tp2']:
                tp_levels.append(('TP2', take_profit['tp2'], 0.3))  # 30% at TP2
            if 'tp3' in take_profit and take_profit['tp3']:
                tp_levels.append(('TP3', take_profit['tp3'], 0.2))  # 20% at TP3

        for tp_name, tp_price, tp_pct in tp_levels:
            if tp_price and tp_price > 0:
                try:
                    tp_quantity = quantity * tp_pct
                    if tp_quantity > 0:
                        tp_order = self.client.place_take_profit_market_order(
                            symbol=symbol,
                            side=close_side,
                            quantity=tp_quantity,
                            stop_price=tp_price,
                            position_side=position_side,
                            reduce_only=True
                        )
                        self.stop_orders[pair]['tp'].append(tp_order.order_id)
                        logger.info(f"{tp_name} placed for {pair} @ ${tp_price:.2f} ({tp_pct*100:.0f}%)")

                except Exception as e:
                    logger.error(f"Error placing {tp_name} for {pair}: {e}")

    def update_positions(self, pair: str, current_price: float) -> Optional[Dict]:
        """
        Actualiza posicion con precio actual y verifica condiciones de salida

        Args:
            pair: Par de trading
            current_price: Precio actual del mercado

        Returns:
            Trade cerrado si se alcanzo SL/TP, None otherwise
        """
        if not self.portfolio.has_position(pair):
            return None

        # Update position with current price
        self.portfolio.update_position(pair, current_price)

        # Check exit conditions manually (in case stop orders haven't triggered)
        return self._check_exit_conditions(pair, current_price)

    def _check_exit_conditions(self, pair: str, current_price: float) -> Optional[Dict]:
        """
        Verifica si se deben cerrar posiciones por SL/TP

        NOTA: Las ordenes de stop ya estan en Binance, pero verificamos
        manualmente por si hay latencia o problemas.

        Args:
            pair: Par de trading
            current_price: Precio actual

        Returns:
            Trade cerrado o None
        """
        position = self.portfolio.get_position(pair)
        if not position:
            return None

        side = position['side']
        stop_loss = position.get('stop_loss')
        take_profit = position.get('take_profit', {})

        # Check liquidation first
        if position.get('liquidated'):
            logger.warning(f"Position {pair} was liquidated!")
            return self._close_position(pair, current_price, reason='LIQUIDATION')

        # Check Stop Loss
        if stop_loss and stop_loss > 0:
            if side == 'BUY' and current_price <= stop_loss:
                logger.info(f"Stop Loss hit for {pair}: ${current_price:.2f} <= ${stop_loss:.2f}")
                return self._close_position(pair, current_price, reason='STOP_LOSS')

            elif side == 'SELL' and current_price >= stop_loss:
                logger.info(f"Stop Loss hit for {pair}: ${current_price:.2f} >= ${stop_loss:.2f}")
                return self._close_position(pair, current_price, reason='STOP_LOSS')

        # Check Take Profit levels
        if take_profit:
            tp_levels = []
            if 'tp1' in take_profit and take_profit['tp1']:
                tp_levels.append(('TP1', take_profit['tp1']))
            if 'tp2' in take_profit and take_profit['tp2']:
                tp_levels.append(('TP2', take_profit['tp2']))
            if 'tp3' in take_profit and take_profit['tp3']:
                tp_levels.append(('TP3', take_profit['tp3']))

            for tp_name, tp_price in tp_levels:
                if side == 'BUY' and current_price >= tp_price:
                    logger.info(f"{tp_name} hit for {pair}: ${current_price:.2f} >= ${tp_price:.2f}")
                    return self._close_position(pair, current_price, reason=tp_name)

                elif side == 'SELL' and current_price <= tp_price:
                    logger.info(f"{tp_name} hit for {pair}: ${current_price:.2f} <= ${tp_price:.2f}")
                    return self._close_position(pair, current_price, reason=tp_name)

        return None

    def _close_position(
        self,
        pair: str,
        current_price: float,
        reason: str = 'MANUAL'
    ) -> Optional[Dict]:
        """
        Cierra una posicion REAL en Binance

        Args:
            pair: Par de trading
            current_price: Precio actual (para referencia)
            reason: Razon del cierre

        Returns:
            Trade cerrado o None
        """
        position = self.portfolio.get_position(pair)
        if not position:
            logger.warning(f"No position found for {pair}")
            return None

        symbol = self._pair_to_symbol(pair)
        side = position['side']
        quantity = position['quantity']
        position_side = self._get_position_side(side)

        try:
            # Cancel any existing stop orders first
            self._cancel_stop_orders(pair, symbol)

            # Determine close side (opposite of position)
            close_side = OrderSide.SELL if side == 'BUY' else OrderSide.BUY

            # Place market order to close position
            logger.info(f"Closing REAL position: {pair} {close_side.value} qty={quantity:.6f} reason={reason}")

            order_result = self.client.place_market_order(
                symbol=symbol,
                side=close_side,
                quantity=quantity,
                position_side=position_side,
                reduce_only=True
            )

            if order_result.status not in ['FILLED', 'NEW', 'PARTIALLY_FILLED']:
                logger.error(f"Close order not filled: {order_result.status}")
                return None

            # Get actual exit price
            actual_exit_price = order_result.avg_price if order_result.avg_price > 0 else current_price

            # Register closed position in portfolio
            closed_trade = self.portfolio.register_closed_position(
                pair=pair,
                exit_price=actual_exit_price,
                reason=reason,
                order_result=order_result.raw_response
            )

            logger.info(f"REAL position closed: {pair} @ ${actual_exit_price:.2f} | Reason: {reason}")

            return closed_trade

        except BinanceAPIError as e:
            logger.error(f"Binance API error closing position: [{e.code}] {e.message}")
            return None
        except Exception as e:
            logger.error(f"Error closing position for {pair}: {e}")
            return None

    def _cancel_stop_orders(self, pair: str, symbol: str):
        """Cancela todas las ordenes de stop para un par"""
        if pair in self.stop_orders:
            stop_info = self.stop_orders[pair]

            # Cancel SL order
            if stop_info.get('sl'):
                try:
                    self.client.cancel_order(symbol, order_id=stop_info['sl'])
                    logger.debug(f"Cancelled SL order for {pair}")
                except Exception as e:
                    logger.debug(f"Could not cancel SL order: {e}")

            # Cancel TP orders
            for tp_order_id in stop_info.get('tp', []):
                try:
                    self.client.cancel_order(symbol, order_id=tp_order_id)
                    logger.debug(f"Cancelled TP order for {pair}")
                except Exception as e:
                    logger.debug(f"Could not cancel TP order: {e}")

            del self.stop_orders[pair]

    def close_all_positions(
        self,
        current_prices: Dict[str, float],
        reason: str = 'MANUAL'
    ) -> List[Dict]:
        """
        Cierra todas las posiciones abiertas

        Args:
            current_prices: Dict con precios actuales {pair: price}
            reason: Razon del cierre

        Returns:
            Lista de trades cerrados
        """
        closed_trades = []
        positions_to_close = list(self.portfolio.positions.keys())

        for pair in positions_to_close:
            current_price = current_prices.get(pair)
            if current_price:
                closed_trade = self._close_position(pair, current_price, reason=reason)
                if closed_trade:
                    closed_trades.append(closed_trade)
            else:
                # Try to get price from Binance
                try:
                    symbol = self._pair_to_symbol(pair)
                    ticker = self.client.get_ticker_price(symbol)
                    price = float(ticker['price'])
                    closed_trade = self._close_position(pair, price, reason=reason)
                    if closed_trade:
                        closed_trades.append(closed_trade)
                except Exception as e:
                    logger.error(f"Could not close {pair}: {e}")

        logger.info(f"Closed {len(closed_trades)} positions. Reason: {reason}")
        return closed_trades

    def get_open_positions_count(self) -> int:
        """Retorna numero de posiciones abiertas"""
        return len(self.portfolio.positions)

    def get_position_info(self, pair: str) -> Optional[Dict]:
        """Retorna informacion de una posicion"""
        return self.portfolio.get_position(pair)

    def sync_positions_with_binance(self):
        """Sincroniza posiciones locales con las reales de Binance"""
        self.portfolio._sync_with_binance()
