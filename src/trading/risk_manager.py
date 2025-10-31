"""
Risk Manager - Gestión de riesgo del portfolio
Ajusta tamaños de posición dinámicamente basado en performance
"""
import logging
from typing import Dict
from src.trading.portfolio import Portfolio

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Gestiona el riesgo del portfolio:
    - Position sizing dinámico
    - Ajuste según drawdown
    - Límites de pérdidas
    - Diversificación
    """

    def __init__(
        self,
        portfolio: Portfolio,
        base_position_size_pct: float = 5.0,
        max_drawdown_limit: float = 20.0,
        max_positions: int = 10,
        max_risk_per_trade_pct: float = 2.0
    ):
        """
        Args:
            portfolio: Instancia del Portfolio
            base_position_size_pct: Tamaño base por posición (% del equity)
            max_drawdown_limit: Drawdown máximo permitido antes de reducir tamaño
            max_positions: Máximo número de posiciones simultáneas
            max_risk_per_trade_pct: Máximo riesgo por trade (% del equity)
        """
        self.portfolio = portfolio
        self.base_position_size_pct = base_position_size_pct
        self.max_drawdown_limit = max_drawdown_limit
        self.max_positions = max_positions
        self.max_risk_per_trade_pct = max_risk_per_trade_pct

    def calculate_position_size(self, signal: Dict, current_price: float) -> float:
        """
        Calcula tamaño óptimo de posición basado en múltiples factores

        Args:
            signal: Señal de trading con score, confidence, etc.
            current_price: Precio actual

        Returns:
            Tamaño de posición en % del equity
        """
        # Factor 1: Score de la señal (0-10)
        score = signal.get('score', 5.0)
        score_factor = min(score / 10.0, 1.0)  # Normalizar a 0-1

        # Factor 2: Confianza de la señal (0-100%)
        confidence = signal.get('confidence', 50)
        confidence_factor = min(confidence / 100.0, 1.0)

        # Factor 3: Estado del portfolio (drawdown)
        stats = self.portfolio.get_statistics()
        drawdown = stats['max_drawdown']

        if drawdown > self.max_drawdown_limit:
            # Reducir tamaño si drawdown es alto
            drawdown_factor = max(0.3, 1.0 - (drawdown / 100.0))
        else:
            drawdown_factor = 1.0

        # Factor 4: Win rate reciente
        win_rate = stats['win_rate']
        if win_rate > 60:
            winrate_factor = 1.2  # Aumentar si gana bien
        elif win_rate < 40:
            winrate_factor = 0.8  # Reducir si pierde mucho
        else:
            winrate_factor = 1.0

        # Calcular tamaño final
        position_size_pct = (
            self.base_position_size_pct *
            score_factor *
            confidence_factor *
            drawdown_factor *
            winrate_factor
        )

        # Limitar entre 1% y 10% del equity
        position_size_pct = max(1.0, min(position_size_pct, 10.0))

        logger.debug(
            f"Position sizing: Base={self.base_position_size_pct}% | "
            f"Score={score_factor:.2f} | Conf={confidence_factor:.2f} | "
            f"DD={drawdown_factor:.2f} | WR={winrate_factor:.2f} → "
            f"Final={position_size_pct:.2f}%"
        )

        return position_size_pct

    def can_open_position(self, pair: str) -> tuple[bool, str]:
        """
        Verifica si se puede abrir nueva posición

        Args:
            pair: Par de trading

        Returns:
            (puede_abrir, razón)
        """
        # Check 1: Máximo de posiciones simultáneas
        if len(self.portfolio.positions) >= self.max_positions:
            return False, f"Máximo de posiciones alcanzado ({self.max_positions})"

        # Check 2: Balance disponible
        if self.portfolio.get_available_balance() < 100:
            return False, "Balance insuficiente (< $100)"

        # Check 3: Drawdown excesivo
        stats = self.portfolio.get_statistics()
        if stats['max_drawdown'] > self.max_drawdown_limit * 1.5:  # 150% del límite
            return False, f"Drawdown excesivo ({stats['max_drawdown']:.1f}% > {self.max_drawdown_limit * 1.5:.1f}%)"

        # Check 4: Ya existe posición en este par
        if self.portfolio.has_position(pair):
            return False, f"Ya existe posición abierta en {pair}"

        return True, "OK"

    def should_reduce_risk(self) -> bool:
        """
        Determina si se debe reducir el riesgo por mal performance

        Returns:
            True si se debe reducir riesgo
        """
        stats = self.portfolio.get_statistics()

        # Reducir si:
        # 1. Drawdown > 15%
        if stats['max_drawdown'] > 15.0:
            return True

        # 2. Win rate < 35%
        if stats['win_rate'] < 35.0 and stats['total_trades'] > 10:
            return True

        # 3. Pérdida neta > -5% del capital inicial
        if stats['roi'] < -5.0:
            return True

        return False

    def get_risk_level(self) -> str:
        """
        Retorna nivel de riesgo actual del portfolio

        Returns:
            'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
        """
        stats = self.portfolio.get_statistics()
        drawdown = stats['max_drawdown']

        if drawdown < 5.0:
            return 'LOW'
        elif drawdown < 10.0:
            return 'MEDIUM'
        elif drawdown < 20.0:
            return 'HIGH'
        else:
            return 'CRITICAL'

    def calculate_risk_reward_ratio(self, entry_price: float, stop_loss: float,
                                   take_profit: float) -> float:
        """
        Calcula ratio riesgo/recompensa

        Args:
            entry_price: Precio de entrada
            stop_loss: Precio de stop loss
            take_profit: Precio de take profit

        Returns:
            Ratio R/R (ej. 3.0 = 1:3)
        """
        if not stop_loss or not take_profit:
            return 0.0

        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)

        if risk == 0:
            return 0.0

        return round(reward / risk, 2)

    def get_risk_report(self) -> Dict:
        """
        Genera reporte completo de riesgo

        Returns:
            Dict con métricas de riesgo
        """
        stats = self.portfolio.get_statistics()

        return {
            'risk_level': self.get_risk_level(),
            'current_drawdown': stats['max_drawdown'],
            'max_drawdown_limit': self.max_drawdown_limit,
            'positions_open': len(self.portfolio.positions),
            'max_positions': self.max_positions,
            'available_balance': self.portfolio.get_available_balance(),
            'equity': self.portfolio.get_equity(),
            'should_reduce_risk': self.should_reduce_risk(),
            'position_size_pct': self.base_position_size_pct,
            'profit_factor': stats['profit_factor'],
            'win_rate': stats['win_rate']
        }
