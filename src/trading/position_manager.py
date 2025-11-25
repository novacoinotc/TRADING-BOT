"""
Position Manager - Gestiona apertura/cierre de posiciones según señales
"""
import logging
import math
from typing import Dict, Optional
from src.trading.portfolio import Portfolio

logger = logging.getLogger(__name__)


class PositionManager:
    """
    Gestiona el ciclo de vida de las posiciones:
    - Apertura según señales
    - Cierre por SL/TP
    - Cierre por señal contraria
    - Trailing stop (opcional)
    """

    def __init__(self, portfolio: Portfolio, max_position_size_pct: float = 5.0):
        """
        Args:
            portfolio: Instancia del Portfolio
            max_position_size_pct: Máximo % del portfolio por posición (default 5%)
        """
        self.portfolio = portfolio
        self.max_position_size_pct = max_position_size_pct

    def process_signal(self, pair: str, signal: Dict, current_price: float) -> Optional[Dict]:
        """
        Procesa una señal de trading y ejecuta la acción correspondiente

        Args:
            pair: Par de trading (ej. BTC/USDT)
            signal: Dict con la señal (action, score, stop_loss, take_profit, etc.)
            current_price: Precio actual del mercado

        Returns:
            Trade ejecutado o None
        """
        action = signal.get('action')

        if action == 'HOLD':
            # Verificar si hay posición abierta que necesite actualización
            if self.portfolio.has_position(pair):
                return self._check_exit_conditions(pair, current_price)
            return None

        # Si hay posición abierta del lado contrario, cerrarla
        if self.portfolio.has_position(pair):
            existing_position = self.portfolio.get_position(pair)

            # Si la señal es contraria a la posición actual, cerrar
            if existing_position['side'] != action:
                logger.info(f"🔄 Señal contraria detectada en {pair}: {existing_position['side']} → {action}")
                closed_trade = self.portfolio.close_position(pair, current_price, reason='SIGNAL_REVERSE')
                # Abrir nueva posición del lado contrario
                if closed_trade:
                    return self._open_new_position(pair, signal, current_price)

            else:
                # Ya tenemos posición del mismo lado, no hacer nada
                logger.debug(f"Ya existe posición {action} en {pair}, ignorando señal")
                return None

        # No hay posición, abrir nueva
        return self._open_new_position(pair, signal, current_price)

    def _open_new_position(self, pair: str, signal: Dict, current_price: float) -> Optional[Dict]:
        """
        Abre nueva posición en FUTURES (permite LONG y SHORT)

        Args:
            pair: Par de trading
            signal: Señal de trading
            current_price: Precio actual

        Returns:
            Posición abierta o None si no hay balance
        """
        action = signal['action']
        stop_loss = signal.get('stop_loss', 0)
        take_profit = signal.get('take_profit', {})

        # Obtener parámetros de futuros de la señal (si vienen del RL Agent)
        trade_type = signal.get('trade_type', 'FUTURES')  # Default FUTURES
        leverage = signal.get('leverage', 1)  # Default 1x (sin apalancamiento)

        # Calcular tamaño de posición
        available_balance = self.portfolio.get_available_balance()
        max_position_value = (self.portfolio.get_equity() * self.max_position_size_pct) / 100

        # Usar el menor entre balance disponible y máximo por posición
        position_value = min(available_balance, max_position_value)

        # VALIDACIÓN CRÍTICA: Verificar que current_price es válido ANTES de calcular quantity
        if current_price is None or current_price <= 0 or math.isnan(current_price) or math.isinf(current_price):
            logger.error(
                f"❌ PRECIO INVÁLIDO para {pair}: {current_price}\n"
                f"   No se puede calcular quantity. Rechazando apertura de posición."
            )
            return None

        # Para FUTURES, el colateral requerido es menor (position_value / leverage)
        if trade_type == 'FUTURES':
            collateral_required = position_value / leverage
            # VALIDACIÓN: Verificar que collateral no es inf/NaN
            if math.isnan(collateral_required) or math.isinf(collateral_required):
                logger.error(
                    f"❌ COLATERAL INVÁLIDO para {pair}: {collateral_required}\n"
                    f"   position_value={position_value}, leverage={leverage}. Rechazando posición."
                )
                return None
            if collateral_required < 10:
                logger.warning(f"Colateral insuficiente para futuros en {pair}: ${collateral_required:.2f}")
                return None
        else:
            if position_value < 10:  # Mínimo $10 por trade
                logger.warning(f"Balance insuficiente para abrir posición en {pair}: ${position_value:.2f}")
                return None

        # Calcular cantidad de cripto a comprar/vender
        quantity = position_value / current_price

        # VALIDACIÓN CRÍTICA: Verificar que quantity no es inf/NaN
        if math.isnan(quantity) or math.isinf(quantity):
            logger.error(
                f"❌ CANTIDAD INVÁLIDA calculada para {pair}: {quantity}\n"
                f"   position_value={position_value}, current_price={current_price}\n"
                f"   Esto indica división por cero o valores corruptos. Rechazando posición."
            )
            return None

        # Abrir posición
        position = self.portfolio.open_position(
            pair=pair,
            side=action,
            entry_price=current_price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trade_type=trade_type,
            leverage=leverage
        )

        return position

    def update_positions(self, pair: str, current_price: float) -> Optional[Dict]:
        """
        Actualiza posición con precio actual y verifica condiciones de salida

        Args:
            pair: Par de trading
            current_price: Precio actual del mercado

        Returns:
            Trade cerrado si se alcanzó SL/TP, None otherwise
        """
        if not self.portfolio.has_position(pair):
            return None

        # Actualizar posición con precio actual
        self.portfolio.update_position(pair, current_price)

        # Verificar condiciones de salida
        return self._check_exit_conditions(pair, current_price)

    def _check_exit_conditions(self, pair: str, current_price: float) -> Optional[Dict]:
        """
        Verifica si se deben cerrar posiciones por SL/TP

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
        entry_price = position['entry_price']
        stop_loss = position.get('stop_loss')
        take_profit = position.get('take_profit', {})

        # Verificar Stop Loss
        if stop_loss:
            if side == 'BUY' and current_price <= stop_loss:
                logger.info(f"🛑 Stop Loss alcanzado en {pair}: ${current_price:.2f} <= ${stop_loss:.2f}")
                return self.portfolio.close_position(pair, current_price, reason='STOP_LOSS')

            elif side == 'SELL' and current_price >= stop_loss:
                logger.info(f"🛑 Stop Loss alcanzado en {pair}: ${current_price:.2f} >= ${stop_loss:.2f}")
                return self.portfolio.close_position(pair, current_price, reason='STOP_LOSS')

        # Verificar Take Profit (usar TP más cercano alcanzado)
        if take_profit:
            tp_levels = []

            if 'tp1' in take_profit:
                tp_levels.append(('TP1', take_profit['tp1']))
            if 'tp2' in take_profit:
                tp_levels.append(('TP2', take_profit['tp2']))
            if 'tp3' in take_profit:
                tp_levels.append(('TP3', take_profit['tp3']))

            for tp_name, tp_price in tp_levels:
                if side == 'BUY' and current_price >= tp_price:
                    logger.info(f"🎯 {tp_name} alcanzado en {pair}: ${current_price:.2f} >= ${tp_price:.2f}")
                    return self.portfolio.close_position(pair, current_price, reason=tp_name)

                elif side == 'SELL' and current_price <= tp_price:
                    logger.info(f"🎯 {tp_name} alcanzado en {pair}: ${current_price:.2f} <= ${tp_price:.2f}")
                    return self.portfolio.close_position(pair, current_price, reason=tp_name)

        return None

    def close_all_positions(self, current_prices: Dict[str, float], reason: str = 'MANUAL'):
        """
        Cierra todas las posiciones abiertas

        Args:
            current_prices: Dict con precios actuales {pair: price}
            reason: Razón del cierre
        """
        positions_to_close = list(self.portfolio.positions.keys())

        for pair in positions_to_close:
            if pair in current_prices:
                current_price = current_prices[pair]
                self.portfolio.close_position(pair, current_price, reason=reason)
            else:
                logger.warning(f"No se puede cerrar {pair} - Precio no disponible")

        logger.info(f"🔒 Todas las posiciones cerradas. Razón: {reason}")

    def get_open_positions_count(self) -> int:
        """Retorna número de posiciones abiertas"""
        return len(self.portfolio.positions)

    def get_position_info(self, pair: str) -> Optional[Dict]:
        """Retorna información de una posición"""
        return self.portfolio.get_position(pair)
