"""
A/B Testing System - Prueba múltiples estrategias en paralelo

Este sistema permite a la IA:
1. Probar dos conjuntos de parámetros simultáneamente
2. Comparar performance en tiempo real
3. Decidir automáticamente cuál estrategia es mejor
4. Switch completo a la estrategia ganadora

Metodología:
- Estrategia A: Parámetros actuales (control)
- Estrategia B: Parámetros nuevos (experimental)
- Split: 50/50 del capital (configurable)
- Duración: N trades o N días
- Métrica: Win rate, Profit factor, Sharpe ratio
"""

import logging
import json
from typing import Dict, List, Optional, Literal
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StrategyVariant:
    """Variante de estrategia para A/B test"""
    name: str  # 'A' o 'B'
    parameters: Dict  # Parámetros de la estrategia
    description: str
    allocated_capital_pct: float  # % del capital total

    # Métricas de performance
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_profit: float = 0.0
    total_loss: float = 0.0
    max_drawdown: float = 0.0

    # Timestamps
    started_at: datetime = field(default_factory=datetime.now)
    last_trade_at: Optional[datetime] = None


@dataclass
class ABTestResult:
    """Resultado del A/B test"""
    winner: str  # 'A', 'B', o 'INCONCLUSIVE'
    confidence: float  # 0-1 (qué tan seguro está)
    metrics_comparison: Dict
    recommendation: str
    should_switch: bool


class ABTestingManager:
    """
    Gestor de A/B testing automático

    La IA controla todo el proceso sin intervención humana
    """

    def __init__(self, config):
        self.config = config

        # Parámetros de A/B testing (optimizables)
        self.enabled = config.get('AB_TESTING_ENABLED', False)  # Deshabilitado por defecto (experimental)
        self.test_duration_trades = config.get('AB_TEST_DURATION_TRADES', 50)  # 30-100 trades
        self.test_duration_days = config.get('AB_TEST_DURATION_DAYS', 7)  # 3-14 días
        self.capital_split = config.get('AB_TEST_CAPITAL_SPLIT', 0.5)  # 0.5 = 50/50
        self.min_confidence_to_switch = config.get('AB_TEST_MIN_CONFIDENCE', 0.8)  # 0.7-0.95
        self.metric_to_optimize = config.get('AB_TEST_METRIC', 'win_rate')  # 'win_rate', 'profit_factor', 'sharpe_ratio'

        # Estado actual del test
        self.active_test: Optional[Dict] = None
        self.strategy_a: Optional[StrategyVariant] = None
        self.strategy_b: Optional[StrategyVariant] = None

        # Historial de tests completados
        self.test_history: List[Dict] = []

        logger.info(f"ABTestingManager initialized: enabled={self.enabled}, duration={self.test_duration_trades} trades")

    def start_ab_test(
        self,
        current_parameters: Dict,
        new_parameters: Dict,
        description_a: str = "Estrategia actual (control)",
        description_b: str = "Estrategia nueva (experimental)"
    ) -> bool:
        """
        Inicia un nuevo A/B test

        Args:
            current_parameters: Parámetros actuales (Estrategia A)
            new_parameters: Parámetros nuevos a probar (Estrategia B)
            description_a: Descripción de A
            description_b: Descripción de B

        Returns:
            True si test iniciado, False si ya hay uno activo
        """
        if not self.enabled:
            logger.warning("⚠️ A/B Testing deshabilitado en config")
            return False

        if self.active_test:
            logger.warning("⚠️ Ya hay un A/B test activo")
            return False

        # Crear variantes
        self.strategy_a = StrategyVariant(
            name='A',
            parameters=current_parameters.copy(),
            description=description_a,
            allocated_capital_pct=self.capital_split
        )

        self.strategy_b = StrategyVariant(
            name='B',
            parameters=new_parameters.copy(),
            description=description_b,
            allocated_capital_pct=1.0 - self.capital_split
        )

        # Iniciar test
        self.active_test = {
            'started_at': datetime.now(),
            'target_trades': self.test_duration_trades,
            'target_end_date': datetime.now() + timedelta(days=self.test_duration_days),
            'status': 'running'
        }

        logger.info(f"🧪 A/B Test iniciado:")
        logger.info(f"  • Estrategia A ({self.capital_split*100:.0f}%): {description_a}")
        logger.info(f"  • Estrategia B ({(1-self.capital_split)*100:.0f}%): {description_b}")
        logger.info(f"  • Duración: {self.test_duration_trades} trades o {self.test_duration_days} días")
        logger.info(f"  • Métrica objetivo: {self.metric_to_optimize}")

        return True

    def assign_strategy_to_trade(self, pair: str) -> Literal['A', 'B']:
        """
        Asigna una estrategia a un nuevo trade

        Usa round-robin balanceado para mantener split 50/50

        Returns:
            'A' o 'B'
        """
        if not self.active_test or not self.strategy_a or not self.strategy_b:
            return 'A'  # Default a estrategia actual

        # Balancear según trades ejecutados
        total_a = self.strategy_a.total_trades
        total_b = self.strategy_b.total_trades
        total = total_a + total_b

        if total == 0:
            return 'A'  # Empezar con A

        # Mantener ratio cercano al capital_split
        current_ratio_a = total_a / total
        target_ratio_a = self.capital_split

        # Si A tiene menos trades de lo esperado, usar A
        if current_ratio_a < target_ratio_a - 0.05:
            return 'A'
        # Si A tiene más trades de lo esperado, usar B
        elif current_ratio_a > target_ratio_a + 0.05:
            return 'B'
        else:
            # Alternar
            return 'B' if total % 2 == 0 else 'A'

    def get_parameters_for_strategy(self, strategy: Literal['A', 'B']) -> Dict:
        """
        Obtiene los parámetros de una estrategia

        Args:
            strategy: 'A' o 'B'

        Returns:
            Dict de parámetros
        """
        if strategy == 'A' and self.strategy_a:
            return self.strategy_a.parameters.copy()
        elif strategy == 'B' and self.strategy_b:
            return self.strategy_b.parameters.copy()
        else:
            return {}

    def record_trade_result(
        self,
        strategy: Literal['A', 'B'],
        profit_pct: float,
        exit_reason: str
    ) -> None:
        """
        Registra resultado de trade para una estrategia

        Args:
            strategy: 'A' o 'B'
            profit_pct: Profit/loss en %
            exit_reason: 'TAKE_PROFIT', 'STOP_LOSS', etc.
        """
        if not self.active_test:
            return

        variant = self.strategy_a if strategy == 'A' else self.strategy_b

        if not variant:
            return

        # Actualizar estadísticas
        variant.total_trades += 1
        variant.last_trade_at = datetime.now()

        if profit_pct > 0:
            variant.winning_trades += 1
            variant.total_profit += profit_pct
        else:
            variant.losing_trades += 1
            variant.total_loss += abs(profit_pct)

        # Calcular drawdown (simplificado)
        # En implementación real, sería más complejo
        if profit_pct < 0:
            variant.max_drawdown = max(variant.max_drawdown, abs(profit_pct))

        logger.debug(f"📊 Estrategia {strategy}: Trade #{variant.total_trades} → {profit_pct:+.2f}%")

        # Verificar si test debe terminar
        self._check_test_completion()

    def _check_test_completion(self) -> None:
        """
        Verifica si el A/B test debe terminar

        Condiciones:
        1. Se alcanzó target_trades
        2. Se alcanzó target_end_date
        3. Diferencia estadísticamente significativa (early stop)
        """
        if not self.active_test or not self.strategy_a or not self.strategy_b:
            return

        total_trades = self.strategy_a.total_trades + self.strategy_b.total_trades

        # Condición 1: Trades alcanzados
        if total_trades >= self.active_test['target_trades']:
            logger.info(f"✅ A/B Test completado: {total_trades} trades alcanzados")
            self._complete_test()
            return

        # Condición 2: Tiempo alcanzado
        if datetime.now() >= self.active_test['target_end_date']:
            logger.info(f"✅ A/B Test completado: {self.test_duration_days} días alcanzados")
            self._complete_test()
            return

        # Condición 3: Early stop si diferencia muy clara (opcional)
        # Implementar statistical significance test (z-test o t-test)
        # Por ahora, skip

    def _complete_test(self) -> None:
        """
        Completa el A/B test y genera resultado
        """
        if not self.strategy_a or not self.strategy_b:
            return

        # Calcular métricas finales
        result = self._evaluate_test()

        # Guardar en historial
        self.test_history.append({
            'started_at': self.active_test['started_at'],
            'completed_at': datetime.now(),
            'strategy_a': self._variant_to_dict(self.strategy_a),
            'strategy_b': self._variant_to_dict(self.strategy_b),
            'result': {
                'winner': result.winner,
                'confidence': result.confidence,
                'should_switch': result.should_switch,
                'recommendation': result.recommendation
            }
        })

        # Log resultado
        logger.info(f"🏁 A/B Test RESULTADO:")
        logger.info(f"  • Winner: {result.winner}")
        logger.info(f"  • Confidence: {result.confidence*100:.1f}%")
        logger.info(f"  • Recomendación: {result.recommendation}")
        logger.info(f"  • Switch automático: {result.should_switch}")

        # Log métricas comparativas
        logger.info(f"\n📊 Métricas Comparativas:")
        for metric, values in result.metrics_comparison.items():
            logger.info(f"  • {metric}: A={values['A']:.2f} vs B={values['B']:.2f}")

        # Auto-switch si ganó B y confidence suficiente
        if result.should_switch:
            logger.info(f"🔄 AUTO-SWITCHING a Estrategia B (ganadora)")
            # Aquí se debería aplicar los parámetros de B globalmente
            # Requiere integración con autonomy_controller

        # Limpiar estado
        self.active_test = None
        self.strategy_a = None
        self.strategy_b = None

    def _evaluate_test(self) -> ABTestResult:
        """
        Evalúa el resultado del A/B test

        Returns:
            ABTestResult con winner y recomendación
        """
        if not self.strategy_a or not self.strategy_b:
            return ABTestResult(
                winner='INCONCLUSIVE',
                confidence=0.0,
                metrics_comparison={},
                recommendation="Test inválido",
                should_switch=False
            )

        # Calcular métricas para ambas estrategias
        metrics_a = self._calculate_metrics(self.strategy_a)
        metrics_b = self._calculate_metrics(self.strategy_b)

        # Comparar según métrica objetivo
        if self.metric_to_optimize == 'win_rate':
            score_a = metrics_a['win_rate']
            score_b = metrics_b['win_rate']
        elif self.metric_to_optimize == 'profit_factor':
            score_a = metrics_a['profit_factor']
            score_b = metrics_b['profit_factor']
        elif self.metric_to_optimize == 'sharpe_ratio':
            score_a = metrics_a.get('sharpe_ratio', 0)
            score_b = metrics_b.get('sharpe_ratio', 0)
        else:
            # Default: win_rate
            score_a = metrics_a['win_rate']
            score_b = metrics_b['win_rate']

        # Determinar ganador
        if score_b > score_a * 1.05:  # B gana por >5%
            winner = 'B'
            improvement = ((score_b - score_a) / score_a) * 100 if score_a > 0 else 100
            confidence = min(improvement / 20.0, 1.0)  # +20% = 100% confidence
        elif score_a > score_b * 1.05:  # A gana por >5%
            winner = 'A'
            improvement = ((score_a - score_b) / score_b) * 100 if score_b > 0 else 100
            confidence = min(improvement / 20.0, 1.0)
        else:
            winner = 'INCONCLUSIVE'
            confidence = 0.5  # Empate técnico

        # Recomendación
        if winner == 'B' and confidence >= self.min_confidence_to_switch:
            recommendation = f"Estrategia B ganó con {confidence*100:.1f}% confidence. Recomendar switch."
            should_switch = True
        elif winner == 'A':
            recommendation = f"Estrategia A (actual) sigue siendo mejor. Mantener."
            should_switch = False
        else:
            recommendation = f"Resultado inconcluso. Considerar test más largo."
            should_switch = False

        # Métricas comparativas
        metrics_comparison = {
            'win_rate': {'A': metrics_a['win_rate'], 'B': metrics_b['win_rate']},
            'profit_factor': {'A': metrics_a['profit_factor'], 'B': metrics_b['profit_factor']},
            'avg_profit': {'A': metrics_a['avg_profit'], 'B': metrics_b['avg_profit']},
            'total_profit': {'A': self.strategy_a.total_profit, 'B': self.strategy_b.total_profit}
        }

        return ABTestResult(
            winner=winner,
            confidence=confidence,
            metrics_comparison=metrics_comparison,
            recommendation=recommendation,
            should_switch=should_switch
        )

    def _calculate_metrics(self, variant: StrategyVariant) -> Dict:
        """
        Calcula métricas de una variante

        Returns:
            Dict con métricas
        """
        if variant.total_trades == 0:
            return {
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_profit': 0.0
            }

        win_rate = (variant.winning_trades / variant.total_trades) * 100

        profit_factor = (
            variant.total_profit / variant.total_loss
            if variant.total_loss > 0 else float('inf')
        )

        avg_profit = (variant.total_profit - variant.total_loss) / variant.total_trades

        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_profit': avg_profit,
            'max_drawdown': variant.max_drawdown
        }

    def _variant_to_dict(self, variant: StrategyVariant) -> Dict:
        """
        Convierte StrategyVariant a Dict para JSON

        Returns:
            Dict serializable
        """
        return {
            'name': variant.name,
            'description': variant.description,
            'parameters': variant.parameters,
            'total_trades': variant.total_trades,
            'winning_trades': variant.winning_trades,
            'losing_trades': variant.losing_trades,
            'total_profit': variant.total_profit,
            'total_loss': variant.total_loss,
            'max_drawdown': variant.max_drawdown,
            'started_at': variant.started_at.isoformat(),
            'last_trade_at': variant.last_trade_at.isoformat() if variant.last_trade_at else None
        }

    def is_test_active(self) -> bool:
        """
        Verifica si hay un test activo

        Returns:
            True si hay test en curso
        """
        return self.active_test is not None

    def get_test_status(self) -> Optional[Dict]:
        """
        Obtiene estado del test activo

        Returns:
            Dict con estado o None si no hay test
        """
        if not self.active_test or not self.strategy_a or not self.strategy_b:
            return None

        total_trades = self.strategy_a.total_trades + self.strategy_b.total_trades
        progress_pct = (total_trades / self.active_test['target_trades']) * 100

        return {
            'status': 'running',
            'started_at': self.active_test['started_at'].isoformat(),
            'target_trades': self.active_test['target_trades'],
            'current_trades': total_trades,
            'progress_pct': progress_pct,
            'strategy_a': {
                'trades': self.strategy_a.total_trades,
                'win_rate': (self.strategy_a.winning_trades / self.strategy_a.total_trades * 100) if self.strategy_a.total_trades > 0 else 0
            },
            'strategy_b': {
                'trades': self.strategy_b.total_trades,
                'win_rate': (self.strategy_b.winning_trades / self.strategy_b.total_trades * 100) if self.strategy_b.total_trades > 0 else 0
            }
        }


# Parámetros optimizables para config.py
AB_TESTING_PARAMS = {
    # Habilitación (experimental, deshabilitado por defecto)
    'AB_TESTING_ENABLED': False,  # True/False

    # Duración del test (optimizables)
    'AB_TEST_DURATION_TRADES': 50,  # 30-100 trades
    'AB_TEST_DURATION_DAYS': 7,  # 3-14 días

    # Split de capital (optimizable 0.3-0.7)
    'AB_TEST_CAPITAL_SPLIT': 0.5,  # 0.5 = 50/50

    # Confidence mínima para switch (optimizable 0.7-0.95)
    'AB_TEST_MIN_CONFIDENCE': 0.8,  # 80% confidence

    # Métrica a optimizar
    'AB_TEST_METRIC': 'win_rate',  # 'win_rate', 'profit_factor', 'sharpe_ratio'
}
