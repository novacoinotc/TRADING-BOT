"""
Futures Trader - Ejecutor de trades en Binance Futures
Maneja apertura/cierre de posiciones con SL y TP
"""

import logging
import math
import time
from typing import Dict, Optional, Tuple, List
from decimal import Decimal

from .binance_client import BinanceClient, BinanceAPIError
from .utils import (
    validate_price,
    validate_quantity,
    round_step_size,
    round_tick_size,
    calculate_quantity
)

logger = logging.getLogger(__name__)


class FuturesTrader:
    """
    Ejecutor de trades para Binance USD-M Futures

    Responsabilidades:
    - Abrir posiciones LONG/SHORT
    - Configurar leverage y margin type
    - Colocar Stop Loss y Take Profit
    - Validar todos los parámetros antes de enviar
    - Cerrar posiciones
    """

    def __init__(
        self,
        client: BinanceClient,
        default_leverage: int = 3,
        use_isolated_margin: bool = True,
        telegram_bot = None
    ):
        """
        Args:
            client: Cliente de Binance
            default_leverage: Leverage por defecto (1-125x)
            use_isolated_margin: Usar margin isolated (recomendado)
            telegram_bot: Instancia de TelegramNotifier para notificaciones
        """
        self.client = client
        self.default_leverage = default_leverage
        self.use_isolated_margin = use_isolated_margin
        self.telegram_bot = telegram_bot

        # Cache de symbol info
        self._symbol_cache = {}

        logger.info(
            f"✅ Futures Trader inicializado | "
            f"Leverage: {default_leverage}x | "
            f"Margin: {'ISOLATED' if use_isolated_margin else 'CROSS'}"
        )

    def _get_symbol_filters(self, symbol: str) -> Dict:
        """
        Obtiene filtros de un símbolo (con cache)

        Args:
            symbol: Símbolo

        Returns:
            Dict: Filtros del símbolo (minQty, stepSize, tickSize, etc.)
        """
        if symbol in self._symbol_cache:
            return self._symbol_cache[symbol]

        symbol_info = self.client.get_symbol_info(symbol)
        if not symbol_info:
            raise ValueError(f"Símbolo no encontrado: {symbol}")

        # Extraer filtros relevantes
        filters = {}
        for f in symbol_info['filters']:
            if f['filterType'] == 'LOT_SIZE':
                filters['minQty'] = float(f['minQty'])
                filters['maxQty'] = float(f['maxQty'])
                filters['stepSize'] = float(f['stepSize'])
            elif f['filterType'] == 'PRICE_FILTER':
                filters['minPrice'] = float(f['minPrice'])
                filters['maxPrice'] = float(f['maxPrice'])
                filters['tickSize'] = float(f['tickSize'])
            elif f['filterType'] == 'MIN_NOTIONAL':
                filters['minNotional'] = float(f['notional'])

        # Añadir precision
        filters['quantityPrecision'] = symbol_info['quantityPrecision']
        filters['pricePrecision'] = symbol_info['pricePrecision']

        # Guardar en cache
        self._symbol_cache[symbol] = filters

        return filters

    def _validate_and_round_quantity(self, symbol: str, quantity: float) -> float:
        """
        Valida y redondea cantidad según filtros del símbolo

        Args:
            symbol: Símbolo
            quantity: Cantidad raw

        Returns:
            float: Cantidad redondeada y validada

        Raises:
            ValueError: Si cantidad es inválida
        """
        filters = self._get_symbol_filters(symbol)

        # Redondear a step size
        step_size = filters['stepSize']
        rounded = round_step_size(quantity, step_size)

        # Validar contra límites
        if rounded < filters['minQty']:
            raise ValueError(
                f"Cantidad muy pequeña: {rounded} < {filters['minQty']} (min)"
            )

        if rounded > filters['maxQty']:
            raise ValueError(
                f"Cantidad muy grande: {rounded} > {filters['maxQty']} (max)"
            )

        return rounded

    def _validate_and_round_price(self, symbol: str, price: float) -> float:
        """
        Valida y redondea precio según filtros del símbolo

        Args:
            symbol: Símbolo
            price: Precio raw

        Returns:
            float: Precio redondeado y validado

        Raises:
            ValueError: Si precio es inválido
        """
        filters = self._get_symbol_filters(symbol)

        # Redondear a tick size
        tick_size = filters['tickSize']
        rounded = round_tick_size(price, tick_size)

        # Validar contra límites
        if rounded < filters['minPrice']:
            raise ValueError(
                f"Precio muy bajo: {rounded} < {filters['minPrice']} (min)"
            )

        if rounded > filters['maxPrice']:
            raise ValueError(
                f"Precio muy alto: {rounded} > {filters['maxPrice']} (max)"
            )

        return rounded

    def setup_position(
        self,
        symbol: str,
        leverage: Optional[int] = None,
        margin_type: Optional[str] = None
    ) -> bool:
        """
        Configura leverage y margin type para un símbolo

        CRÍTICO: Ejecutar ANTES de abrir posición

        Args:
            symbol: Símbolo
            leverage: Leverage (si None, usa default)
            margin_type: 'ISOLATED' o 'CROSS' (si None, usa default)

        Returns:
            bool: True si se configuró correctamente
        """
        leverage = leverage or self.default_leverage
        margin_type = margin_type or ('ISOLATED' if self.use_isolated_margin else 'CROSS')

        try:
            # Configurar leverage
            result_leverage = self.client.change_leverage(symbol, leverage)
            logger.info(
                f"⚙️ Leverage configurado: {symbol} = {result_leverage['leverage']}x"
            )

            # Configurar margin type
            try:
                result_margin = self.client.change_margin_type(symbol, margin_type)
                logger.info(f"⚙️ Margin type configurado: {symbol} = {margin_type}")
            except BinanceAPIError as e:
                # Error -4046 = margin type ya está configurado (ignorar)
                if e.code == -4046:
                    logger.debug(f"Margin type ya configurado para {symbol}")
                else:
                    raise

            return True

        except Exception as e:
            logger.error(f"❌ Error configurando posición para {symbol}: {e}")
            return False

    def open_position(
        self,
        symbol: str,
        side: str,
        usdt_amount: float,
        stop_loss_pct: float,
        take_profit_pct: float,
        leverage: Optional[int] = None,
        current_price: Optional[float] = None
    ) -> Optional[Dict]:
        """
        Abre posición LONG o SHORT con SL y TP

        Flujo completo:
        1. Configura leverage y margin
        2. Obtiene precio actual
        3. Calcula cantidad
        4. Abre posición MARKET
        5. Coloca Stop Loss (STOP_MARKET)
        6. Coloca Take Profit (TAKE_PROFIT_MARKET)

        Args:
            symbol: Símbolo (ej: BTCUSDT)
            side: 'BUY' (LONG) o 'SELL' (SHORT)
            usdt_amount: Cantidad en USDT a usar
            stop_loss_pct: % de stop loss (ej: 2.0 = 2%)
            take_profit_pct: % de take profit (ej: 3.0 = 3%)
            leverage: Leverage opcional (si None, usa default)
            current_price: Precio actual opcional (si None, lo obtiene)

        Returns:
            Dict: Información del trade abierto con order IDs
            None: Si hubo error

        Example:
            >>> trader.open_position(
            ...     symbol='BTCUSDT',
            ...     side='BUY',
            ...     usdt_amount=100,
            ...     stop_loss_pct=2.0,
            ...     take_profit_pct=3.0,
            ...     leverage=3
            ... )
        """
        leverage = leverage or self.default_leverage

        logger.info(
            f"\n{'='*60}\n"
            f"📊 OPENING POSITION\n"
            f"Symbol: {symbol}\n"
            f"Side: {side}\n"
            f"USDT Amount: ${usdt_amount:.2f}\n"
            f"Leverage: {leverage}x\n"
            f"Stop Loss: {stop_loss_pct}%\n"
            f"Take Profit: {take_profit_pct}%\n"
            f"{'='*60}"
        )

        try:
            # PASO 1: Configurar leverage y margin
            if not self.setup_position(symbol, leverage=leverage):
                logger.error("❌ Failed to setup position")
                return None

            # PASO 2: Obtener precio actual
            if current_price is None:
                current_price = self.client.get_price(symbol)

            if not validate_price(current_price, symbol):
                logger.error("❌ Invalid current price")
                return None

            logger.info(f"📈 Current price: ${current_price:,.2f}")

            # PASO 3: Calcular cantidad
            filters = self._get_symbol_filters(symbol)
            step_size = filters['stepSize']

            quantity = calculate_quantity(
                usdt_amount=usdt_amount,
                price=current_price,
                leverage=leverage,
                step_size=step_size
            )

            # Validar cantidad
            try:
                quantity = self._validate_and_round_quantity(symbol, quantity)
            except ValueError as e:
                logger.error(f"❌ Invalid quantity: {e}")
                return None

            logger.info(f"📊 Quantity: {quantity} {symbol}")

            # PASO 4: Calcular SL y TP
            if side == 'BUY':  # LONG
                stop_loss_price = current_price * (1 - stop_loss_pct / 100)
                take_profit_price = current_price * (1 + take_profit_pct / 100)
            else:  # SELL (SHORT)
                stop_loss_price = current_price * (1 + stop_loss_pct / 100)
                take_profit_price = current_price * (1 - take_profit_pct / 100)

            # 🔍 VALIDACIÓN CRÍTICA: Verificar distancia mínima de TP
            min_distance_pct = 0.1  # Mínimo 0.1% de diferencia
            actual_tp_distance = abs((take_profit_price - current_price) / current_price) * 100

            if actual_tp_distance < min_distance_pct:
                logger.warning(
                    f"⚠️ TP demasiado cerca del entry ({actual_tp_distance:.3f}% < {min_distance_pct}%), "
                    f"NO se colocará orden de Take Profit"
                )
                take_profit_price = None  # No colocar TP
            else:
                # Validar y redondear precio de TP
                try:
                    take_profit_price = self._validate_and_round_price(symbol, take_profit_price)
                except ValueError as e:
                    logger.error(f"❌ Invalid TP price: {e}")
                    take_profit_price = None

            # Validar y redondear precio de SL
            try:
                stop_loss_price = self._validate_and_round_price(symbol, stop_loss_price)
            except ValueError as e:
                logger.error(f"❌ Invalid SL price: {e}")
                return None

            logger.info(f"🛑 Stop Loss: ${stop_loss_price:,.2f}")
            if take_profit_price:
                logger.info(f"🎯 Take Profit: ${take_profit_price:,.2f}")
            else:
                logger.info(f"🎯 Take Profit: NO colocado (distancia < {min_distance_pct}%)")

            # PASO 5: Abrir posición MARKET
            logger.info(f"📤 Sending MARKET order...")
            market_order = self.client.create_order(
                symbol=symbol,
                side=side,
                order_type='MARKET',
                quantity=quantity,
                position_side='BOTH'
            )

            logger.info(
                f"✅ Position opened!\n"
                f"   Order ID: {market_order['orderId']}\n"
                f"   Status: {market_order['status']}\n"
                f"   Price: ${float(market_order.get('avgPrice', current_price)):,.2f}"
            )

            # 🔧 FIX: Obtener precio real de entrada consultando la posición
            import time
            time.sleep(0.5)  # Esperar que Binance actualice

            # Consultar posición abierta para obtener avgPrice real
            entry_price = current_price  # Default fallback
            try:
                positions = self.client.get_position_risk(symbol=symbol)
                for pos in positions:
                    if float(pos.get('positionAmt', 0)) != 0:
                        entry_price = float(pos['entryPrice'])
                        logger.info(f"✅ Entry price confirmado: ${entry_price:,.2f}")
                        break
            except Exception as e:
                logger.warning(f"⚠️ Error obteniendo entry price, usando current_price: {e}")

            # PASO 6: Colocar Stop Loss
            logger.info(f"📤 Placing Stop Loss order...")
            sl_side = 'SELL' if side == 'BUY' else 'BUY'  # Opuesto a la posición
            sl_order = self.client.create_order(
                symbol=symbol,
                side=sl_side,
                order_type='STOP_MARKET',
                quantity=quantity,
                stop_price=stop_loss_price,
                position_side='BOTH',
                reduce_only=True  # Solo reduce posición, no invierte
            )

            logger.info(
                f"✅ Stop Loss placed!\n"
                f"   Order ID: {sl_order['orderId']}\n"
                f"   Stop Price: ${stop_loss_price:,.2f}"
            )

            # PASO 7: Colocar Take Profit (solo si es válido)
            tp_order = None
            if take_profit_price is not None:
                logger.info(f"📤 Placing Take Profit order...")
                try:
                    tp_order = self.client.create_order(
                        symbol=symbol,
                        side=sl_side,
                        order_type='TAKE_PROFIT_MARKET',
                        quantity=quantity,
                        stop_price=take_profit_price,
                        position_side='BOTH',
                        reduce_only=True
                    )

                    logger.info(
                        f"✅ Take Profit placed!\n"
                        f"   Order ID: {tp_order['orderId']}\n"
                        f"   Stop Price: ${take_profit_price:,.2f}"
                    )
                except BinanceAPIError as e:
                    logger.error(f"❌ Error placing TP (no crítico): {e.message}")
                    # NO fallar el trade por error de TP
            else:
                logger.warning(f"⚠️ Take Profit NO colocado (TP demasiado cerca del entry)")

            # Retornar información completa del trade
            trade_info = {
                'symbol': symbol,
                'side': side,
                'leverage': leverage,
                'quantity': quantity,
                'entry_price': entry_price,
                'stop_loss': stop_loss_price,
                'take_profit': take_profit_price,  # Puede ser None
                'usdt_amount': usdt_amount,
                'market_order_id': market_order['orderId'],
                'sl_order_id': sl_order['orderId'],
                'tp_order_id': tp_order['orderId'] if tp_order else None,  # Manejar None
                'timestamp': market_order['updateTime'],
                'status': 'OPEN'
            }

            logger.info(
                f"\n{'='*60}\n"
                f"✅ POSITION OPENED SUCCESSFULLY\n"
                f"{'='*60}\n"
            )

            # 🔴 CRÍTICO: Enviar notificación a Telegram (manejo robusto de async/sync)
            if self.telegram_bot:
                try:
                    # Construir mensaje de notificación
                    direction_emoji = "🟢" if side == 'BUY' else "🔴"
                    notional = trade_info['entry_price'] * trade_info['quantity'] * leverage
                    notification_message = (
                        f"{direction_emoji} **TRADE ABIERTO**\n"
                        f"━━━━━━━━━━━━━━━━━\n"
                        f"Par: `{symbol}`\n"
                        f"Dirección: **{'LONG' if side == 'BUY' else 'SHORT'}**\n"
                        f"Precio entrada: `${trade_info['entry_price']:,.2f}`\n"
                        f"Cantidad: `{trade_info['quantity']:.4f}`\n"
                        f"Nocional: `${notional:.2f} ({leverage}x leverage)`\n"
                        f"━━━━━━━━━━━━━━━━━\n"
                        f"🛑 Stop Loss: `${trade_info['stop_loss']:,.2f}` (-{stop_loss_pct}%)\n"
                        f"🎯 Take Profit: `${trade_info['take_profit']:,.2f}` (+{take_profit_pct}%)\n"
                        f"━━━━━━━━━━━━━━━━━\n"
                        f"Order ID: `{trade_info['market_order_id']}`"
                    )

                    import asyncio
                    import threading

                    # Verificar si hay event loop corriendo
                    try:
                        loop = asyncio.get_running_loop()
                        # Hay event loop activo, crear task
                        loop.create_task(self.telegram_bot.send_message(notification_message))
                        logger.info("📢 Trade notification queued in event loop")

                    except RuntimeError:
                        # No hay event loop, ejecutar en thread separado para no bloquear
                        def send_notification():
                            try:
                                # Verificar si el método es async o sync
                                if asyncio.iscoroutinefunction(self.telegram_bot.send_message):
                                    # Método async: crear nuevo event loop
                                    asyncio.run(self.telegram_bot.send_message(notification_message))
                                else:
                                    # Método sync: ejecutar directamente
                                    self.telegram_bot.send_message(notification_message)
                                logger.info("📢 Trade notification sent successfully")
                            except Exception as e:
                                logger.error(f"❌ Error sending notification in thread: {e}", exc_info=True)

                        # Iniciar thread daemon (no bloquea el shutdown)
                        notification_thread = threading.Thread(
                            target=send_notification,
                            daemon=True,
                            name=f"TelegramNotif-{symbol}"
                        )
                        notification_thread.start()
                        logger.info("📢 Trade notification sent in background thread")

                except Exception as e:
                    logger.error(f"❌ Failed to setup trade notification: {e}", exc_info=True)
                    # NO fallar el trade por error de notificación

            return trade_info

        except BinanceAPIError as e:
            logger.error(
                f"❌ Binance API Error opening position:\n"
                f"   Code: {e.code}\n"
                f"   Message: {e.message}"
            )
            return None

        except Exception as e:
            logger.error(f"❌ Unexpected error opening position: {e}", exc_info=True)
            return None

    def close_position(
        self,
        symbol: str,
        quantity: Optional[float] = None,
        reason: str = 'MANUAL'
    ) -> Optional[Dict]:
        """
        Cierra posición abierta

        Args:
            symbol: Símbolo
            quantity: Cantidad a cerrar (si None, cierra todo)
            reason: Razón del cierre

        Returns:
            Dict: Información del cierre
            None: Si hubo error
        """
        try:
            logger.info(f"\n{'='*60}\n🔒 CLOSING POSITION: {symbol}\nReason: {reason}\n{'='*60}")

            # Obtener posición actual
            positions = self.client.get_position_risk(symbol=symbol)
            position = None
            for p in positions:
                if p.get('symbol') == symbol and float(p.get('positionAmt', 0)) != 0:
                    position = p
                    break

            if not position:
                logger.warning(f"⚠️ No open position found for {symbol}")
                return None

            # Determinar cantidad a cerrar
            position_amt = abs(float(position.get('positionAmt', 0)))
            if quantity is None:
                quantity = position_amt
            else:
                quantity = min(quantity, position_amt)

            # Validar cantidad
            try:
                quantity = self._validate_and_round_quantity(symbol, quantity)
            except ValueError as e:
                logger.error(f"❌ Invalid quantity: {e}")
                return None

            # Determinar side opuesto
            current_side = 'LONG' if float(position.get('positionAmt', 0)) > 0 else 'SHORT'
            close_side = 'SELL' if current_side == 'LONG' else 'BUY'

            logger.info(
                f"📊 Closing {current_side} position\n"
                f"   Quantity: {quantity}\n"
                f"   Entry Price: ${float(position['entryPrice']):,.2f}"
            )

            # Cancelar órdenes abiertas (SL/TP)
            try:
                # 🔍 LOGGING CRÍTICO: Rastrear quién cancela SL
                import traceback
                logger.warning(f"⚠️ CANCELANDO ÓRDENES SL/TP: symbol={symbol}, reason={reason}")
                logger.warning(f"⚠️ TRACEBACK: {''.join(traceback.format_stack()[-5:])}")

                self.client.cancel_all_orders(symbol)
                logger.info("✅ Cancelled pending SL/TP orders")
            except Exception as e:
                logger.warning(f"⚠️ Could not cancel orders: {e}")

            # Cerrar posición con orden MARKET
            close_order = self.client.create_order(
                symbol=symbol,
                side=close_side,
                order_type='MARKET',
                quantity=quantity,
                position_side='BOTH',
                reduce_only=True
            )

            # Obtener precio de salida (avgPrice puede ser 0 inmediatamente después de MARKET order)
            exit_price = float(close_order.get('avgPrice', 0))

            # Si avgPrice es 0, intentar obtener de la orden actualizada
            if exit_price == 0:
                try:
                    order_id = close_order.get('orderId')
                    if order_id:
                        # Consultar la orden para obtener avgPrice actualizado
                        updated_order = self.client.get_order(symbol=symbol, order_id=order_id)
                        exit_price = float(updated_order.get('avgPrice', 0))
                        logger.debug(f"📊 avgPrice obtenido de orden actualizada: ${exit_price:,.2f}")
                except Exception as e:
                    logger.warning(f"⚠️ No se pudo obtener avgPrice actualizado: {e}")

            # Si aún es 0, usar precio actual del mercado como fallback
            if exit_price == 0:
                try:
                    ticker = self.client.get_ticker(symbol=symbol)
                    exit_price = float(ticker.get('lastPrice', 0))
                    logger.debug(f"📊 Usando precio actual del mercado como exit_price: ${exit_price:,.2f}")
                except Exception as e:
                    logger.warning(f"⚠️ No se pudo obtener precio del mercado: {e}")

            logger.info(
                f"✅ Position closed!\n"
                f"   Order ID: {close_order.get('orderId', 'N/A')}\n"
                f"   Exit Price: ${exit_price:,.2f}\n"
                f"   Status: {close_order.get('status', 'UNKNOWN')}"
            )

            # Calcular P&L (NO multiplicar por leverage según docs oficiales Binance)
            entry_price = float(position.get('entryPrice', 0))
            if current_side == 'LONG':
                pnl = (exit_price - entry_price) * quantity
            else:  # SHORT
                pnl = (entry_price - exit_price) * quantity

            # Obtener leverage (solo para calcular ROE%)
            leverage = int(position.get('leverage', 1))

            # Calcular ROE% (Return on Equity) - el leverage SÍ afecta esto
            initial_margin = (entry_price * quantity) / leverage if leverage > 0 else entry_price * quantity
            roe_pct = (pnl / initial_margin) * 100 if initial_margin > 0 else 0

            close_info = {
                'symbol': symbol,
                'side': current_side,
                'quantity': quantity,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,  # P&L en USDT (absoluto, SIN multiplicar por leverage)
                'pnl_pct': roe_pct,  # ROE% (retorno sobre margen inicial)
                'leverage': leverage,
                'order_id': close_order.get('orderId', 0),
                'reason': reason,
                'timestamp': close_order.get('updateTime', int(time.time() * 1000))
            }

            logger.info(
                f"\n{'='*60}\n"
                f"✅ POSITION CLOSED\n"
                f"P&L: ${pnl:+.2f} | ROE: {roe_pct:+.2f}% ({leverage}x leverage)\n"
                f"{'='*60}\n"
            )

            return close_info

        except BinanceAPIError as e:
            logger.error(
                f"❌ Binance API Error closing position:\n"
                f"   Code: {e.code}\n"
                f"   Message: {e.message}"
            )
            return None

        except Exception as e:
            logger.error(f"❌ Unexpected error closing position: {e}", exc_info=True)
            return None

    def close_all_positions(self, reason: str = 'MANUAL') -> List[Dict]:
        """
        Cierra TODAS las posiciones abiertas

        Args:
            reason: Razón del cierre

        Returns:
            List: Lista de posiciones cerradas
        """
        logger.warning(f"\n{'='*60}\n⚠️ CLOSING ALL POSITIONS\nReason: {reason}\n{'='*60}")

        closed = []

        try:
            # Obtener todas las posiciones
            positions = self.client.get_position_risk()

            for pos in positions:
                position_amt = float(pos['positionAmt'])
                if position_amt != 0:  # Tiene posición abierta
                    symbol = pos['symbol']
                    result = self.close_position(symbol, reason=reason)
                    if result:
                        closed.append(result)

            logger.info(f"✅ Closed {len(closed)} positions")
            return closed

        except Exception as e:
            logger.error(f"❌ Error closing all positions: {e}", exc_info=True)
            return closed

    async def modify_stop_loss(self, symbol: str, new_sl_price: float) -> bool:
        """
        Modifica el Stop Loss de una posición activa

        Args:
            symbol: Símbolo
            new_sl_price: Nuevo precio de Stop Loss

        Returns:
            bool: True si se modificó exitosamente
        """
        try:
            logger.info(f"🔄 Modificando Stop Loss para {symbol} a ${new_sl_price:.4f}")

            # Obtener posición actual
            positions = self.client.get_position_risk(symbol=symbol)
            position = None
            for p in positions:
                if p.get('symbol') == symbol and float(p.get('positionAmt', 0)) != 0:
                    position = p
                    break

            if not position:
                logger.warning(f"⚠️ No hay posición abierta para {symbol}")
                return False

            # Determinar side y cantidad
            position_amt = float(position.get('positionAmt', 0))
            current_side = 'LONG' if position_amt > 0 else 'SHORT'
            quantity = abs(position_amt)

            # Validar y redondear cantidad y precio
            quantity = self._validate_and_round_quantity(symbol, quantity)
            new_sl_price = self._validate_and_round_price(symbol, new_sl_price)

            # Cancelar órdenes de Stop Loss existentes
            try:
                open_orders = self.client.get_open_orders(symbol=symbol)
                for order in open_orders:
                    if order.get('type') == 'STOP_MARKET':
                        self.client.cancel_order(symbol=symbol, order_id=order['orderId'])
                        logger.info(f"✅ Cancelado SL anterior (Order ID: {order['orderId']})")
            except Exception as e:
                logger.warning(f"⚠️ Error cancelando SL anterior: {e}")

            # Colocar nuevo Stop Loss
            sl_side = 'SELL' if current_side == 'LONG' else 'BUY'
            sl_order = self.client.create_order(
                symbol=symbol,
                side=sl_side,
                order_type='STOP_MARKET',
                quantity=quantity,
                stop_price=new_sl_price,
                position_side='BOTH',
                reduce_only=True
            )

            logger.info(
                f"✅ Stop Loss modificado!\n"
                f"   Order ID: {sl_order['orderId']}\n"
                f"   Nuevo Stop Price: ${new_sl_price:.4f}"
            )

            return True

        except Exception as e:
            logger.error(f"❌ Error modificando Stop Loss para {symbol}: {e}", exc_info=True)
            return False

    async def close_partial_position(self, symbol: str, quantity: float) -> Optional[Dict]:
        """
        Cierra parcialmente una posición (ej: cerrar 50%)

        Args:
            symbol: Símbolo
            quantity: Cantidad a cerrar

        Returns:
            Dict: Información del cierre parcial
            None: Si hubo error
        """
        try:
            logger.info(f"💰 Cerrando posición parcial: {symbol} - {quantity} qty")

            # Usar el método close_position con cantidad especificada
            result = self.close_position(
                symbol=symbol,
                quantity=quantity,
                reason='PARTIAL_TP'
            )

            if result:
                logger.info(f"✅ Posición parcial cerrada: {quantity} qty de {symbol}")
            else:
                logger.error(f"❌ Error cerrando posición parcial de {symbol}")

            return result

        except Exception as e:
            logger.error(f"❌ Error en close_partial_position para {symbol}: {e}", exc_info=True)
            return None

    def _get_price_precision(self, symbol: str) -> int:
        """
        Obtiene la precisión de precio para un símbolo

        Args:
            symbol: Símbolo

        Returns:
            int: Número de decimales para el precio
        """
        try:
            filters = self._get_symbol_filters(symbol)
            return filters.get('pricePrecision', 2)
        except Exception as e:
            logger.warning(f"⚠️ No se pudo obtener precisión de precio para {symbol}: {e}")
            return 2  # Default 2 decimales
