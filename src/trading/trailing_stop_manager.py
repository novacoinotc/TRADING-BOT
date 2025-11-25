"""
Trailing Stop Manager - Stop Loss dinámico que sigue el precio

Este módulo implementa trailing stops automáticos que:
1. Protegen ganancias cuando el precio sube
2. Se ajustan automáticamente sin intervención humana
3. Convierten SL a breakeven después de cierta ganancia
4. Lock profits cada X% de movimiento

Ideal para scalping: asegura ganancias sin cortar profits prematuramente
"""

import logging
from typing import Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class TrailingStopState:
    """Estado del trailing stop para una posición"""
    pair: str
    side: str  # 'BUY' o 'SELL'
    entry_price: float
    current_stop_loss: float
    highest_price: float  # Para BUY (peak alcanzado)
    lowest_price: float  # Para SELL (valley alcanzado)
    initial_stop_loss: float  # SL original
    breakeven_activated: bool = False
    profit_locked: float = 0.0  # % de profit asegurado
    last_updated: datetime = field(default_factory=datetime.now)


class TrailingStopManager:
    """
    Gestor de trailing stops automáticos

    La IA controla todos los parámetros sin intervención humana
    """

    def __init__(self, config):
        self.config = config

        # Parámetros optimizables por la IA
        self.trailing_enabled = config.get('TRAILING_STOP_ENABLED', True)
        self.trailing_distance_pct = config.get('TRAILING_DISTANCE_PCT', 0.4)  # 0.3-0.7% (distancia del peak)
        self.breakeven_after_pct = config.get('BREAKEVEN_AFTER_PCT', 0.5)  # 0.3-1.0% (activar BE después de esta ganancia)
        self.lock_profit_step_pct = config.get('LOCK_PROFIT_STEP_PCT', 0.5)  # 0.3-0.8% (cada cuánto lockear)
        self.min_profit_to_lock_pct = config.get('MIN_PROFIT_TO_LOCK_PCT', 0.3)  # 0.2-0.5% (mínimo para lockear)

        # Estado de trailing stops activos {pair: TrailingStopState}
        self.active_trails: Dict[str, TrailingStopState] = {}

        logger.info(f"TrailingStopManager initialized: distance={self.trailing_distance_pct}%, breakeven_after={self.breakeven_after_pct}%")

    def start_trailing(
        self,
        pair: str,
        side: str,
        entry_price: float,
        initial_stop_loss: float,
        current_price: float
    ) -> None:
        """
        Inicia el trailing stop para una nueva posición

        Args:
            pair: Par de trading (e.g., 'BTC/USDT')
            side: 'BUY' o 'SELL'
            entry_price: Precio de entrada
            initial_stop_loss: Stop loss inicial (fijo)
            current_price: Precio actual del mercado
        """
        if not self.trailing_enabled:
            logger.info(f"⚠️ Trailing stops deshabilitados (config)")
            return

        trail_state = TrailingStopState(
            pair=pair,
            side=side,
            entry_price=entry_price,
            current_stop_loss=initial_stop_loss,
            highest_price=current_price if side == 'BUY' else entry_price,
            lowest_price=current_price if side == 'SELL' else entry_price,
            initial_stop_loss=initial_stop_loss,
            breakeven_activated=False,
            profit_locked=0.0
        )

        self.active_trails[pair] = trail_state
        logger.info(f"✅ Trailing stop iniciado para {pair} @ {entry_price:.6f} (SL inicial: {initial_stop_loss:.6f})")

    def update_trailing_stop(
        self,
        pair: str,
        current_price: float
    ) -> Optional[float]:
        """
        Actualiza el trailing stop basado en el precio actual

        Esta función se llama cada tick/minuto para ajustar el SL automáticamente

        Returns:
            Nuevo stop loss si cambió, None si no hubo cambios

        Lógica:
        1. Si profit >= breakeven_after_pct → Mover SL a breakeven
        2. Si profit >= min_profit_to_lock_pct → Activar trailing
        3. Si precio sube (BUY) → Ajustar SL hacia arriba (mantener distancia)
        4. Si precio baja (SELL) → Ajustar SL hacia abajo
        """
        if pair not in self.active_trails:
            return None

        trail = self.active_trails[pair]
        old_sl = trail.current_stop_loss

        # Calcular profit actual
        if trail.side == 'BUY':
            profit_pct = ((current_price - trail.entry_price) / trail.entry_price) * 100
            is_new_high = current_price > trail.highest_price
        else:  # SELL
            profit_pct = ((trail.entry_price - current_price) / trail.entry_price) * 100
            is_new_low = current_price < trail.lowest_price

        # **PASO 1: Activar Breakeven si alcanzamos el threshold**
        if not trail.breakeven_activated and profit_pct >= self.breakeven_after_pct:
            trail.current_stop_loss = trail.entry_price
            trail.breakeven_activated = True
            trail.profit_locked = 0.0
            logger.info(f"🎯 BREAKEVEN activado para {pair} (profit {profit_pct:.2f}% >= {self.breakeven_after_pct}%) - SL movido a {trail.entry_price:.6f}")

        # **PASO 2: Trailing activo (solo si profit > mínimo)**
        elif profit_pct >= self.min_profit_to_lock_pct:

            if trail.side == 'BUY' and is_new_high:
                # Nuevo high: actualizar trailing stop
                trail.highest_price = current_price

                # Calcular nuevo SL (distance% por debajo del high)
                new_sl = trail.highest_price * (1 - self.trailing_distance_pct / 100)

                # Solo subir el SL, nunca bajar
                if new_sl > trail.current_stop_loss:
                    # Calcular profit asegurado
                    locked_profit_pct = ((new_sl - trail.entry_price) / trail.entry_price) * 100

                    trail.current_stop_loss = new_sl
                    trail.profit_locked = locked_profit_pct
                    trail.last_updated = datetime.now()

                    logger.info(f"📈 Trailing SL subido para {pair}: {old_sl:.6f} → {new_sl:.6f} (high={trail.highest_price:.6f}, profit locked={locked_profit_pct:.2f}%)")

            elif trail.side == 'SELL' and is_new_low:
                # Nuevo low: actualizar trailing stop
                trail.lowest_price = current_price

                # Calcular nuevo SL (distance% por arriba del low)
                new_sl = trail.lowest_price * (1 + self.trailing_distance_pct / 100)

                # Solo bajar el SL, nunca subir
                if new_sl < trail.current_stop_loss:
                    # Calcular profit asegurado
                    locked_profit_pct = ((trail.entry_price - new_sl) / trail.entry_price) * 100

                    trail.current_stop_loss = new_sl
                    trail.profit_locked = locked_profit_pct
                    trail.last_updated = datetime.now()

                    logger.info(f"📉 Trailing SL bajado para {pair}: {old_sl:.6f} → {new_sl:.6f} (low={trail.lowest_price:.6f}, profit locked={locked_profit_pct:.2f}%)")

        # Retornar nuevo SL si cambió
        if trail.current_stop_loss != old_sl:
            return trail.current_stop_loss

        return None

    def check_stop_loss_hit(
        self,
        pair: str,
        current_price: float
    ) -> bool:
        """
        Verifica si el stop loss fue tocado

        Returns:
            True si debe cerrar posición (SL hit), False si continuar
        """
        if pair not in self.active_trails:
            return False

        trail = self.active_trails[pair]

        if trail.side == 'BUY':
            # SL tocado si precio cae por debajo
            if current_price <= trail.current_stop_loss:
                loss_pct = ((current_price - trail.entry_price) / trail.entry_price) * 100
                logger.warning(f"🛑 STOP LOSS HIT para {pair}: precio={current_price:.6f} <= SL={trail.current_stop_loss:.6f} (loss={loss_pct:.2f}%)")
                return True

        else:  # SELL
            # SL tocado si precio sube por encima
            if current_price >= trail.current_stop_loss:
                loss_pct = ((trail.entry_price - current_price) / trail.entry_price) * 100
                logger.warning(f"🛑 STOP LOSS HIT para {pair}: precio={current_price:.6f} >= SL={trail.current_stop_loss:.6f} (loss={loss_pct:.2f}%)")
                return True

        return False

    def stop_trailing(self, pair: str) -> None:
        """
        Detiene el trailing stop para un par (posición cerrada)
        """
        if pair in self.active_trails:
            trail = self.active_trails[pair]
            logger.info(f"✅ Trailing stop detenido para {pair} (profit locked={trail.profit_locked:.2f}%)")
            del self.active_trails[pair]

    def get_trailing_state(self, pair: str) -> Optional[TrailingStopState]:
        """
        Obtiene el estado actual del trailing stop

        Returns:
            TrailingStopState o None si no hay trailing activo
        """
        return self.active_trails.get(pair)

    def get_all_active_trails(self) -> Dict[str, TrailingStopState]:
        """
        Obtiene todos los trailing stops activos

        Returns:
            Dict {pair: TrailingStopState}
        """
        return self.active_trails.copy()

    def get_statistics(self) -> Dict:
        """
        Estadísticas de trailing stops

        Returns:
            Dict con métricas
        """
        if not self.active_trails:
            return {
                'active_trails': 0,
                'total_locked_profit': 0.0,
                'breakeven_activated': 0
            }

        total_locked = sum(t.profit_locked for t in self.active_trails.values())
        breakeven_count = sum(1 for t in self.active_trails.values() if t.breakeven_activated)

        return {
            'active_trails': len(self.active_trails),
            'total_locked_profit': total_locked,
            'avg_locked_profit': total_locked / len(self.active_trails) if self.active_trails else 0.0,
            'breakeven_activated': breakeven_count
        }


# Parámetros optimizables para agregar a config.py
TRAILING_STOP_PARAMS = {
    # Habilitación (la IA puede desactivarlo si no mejora performance)
    'TRAILING_STOP_ENABLED': True,  # True/False

    # Distancia del trailing (optimizable 0.3-0.7%)
    'TRAILING_DISTANCE_PCT': 0.4,  # % por debajo del peak/valley

    # Activación de breakeven (optimizable 0.3-1.0%)
    'BREAKEVEN_AFTER_PCT': 0.5,  # % ganancia para mover SL a breakeven

    # Lock profit step (optimizable 0.3-0.8%)
    'LOCK_PROFIT_STEP_PCT': 0.5,  # Cada cuánto % ajustar SL

    # Profit mínimo para activar trailing (optimizable 0.2-0.5%)
    'MIN_PROFIT_TO_LOCK_PCT': 0.3,  # % mínimo para empezar trailing
}
