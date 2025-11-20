"""
Trade Management Learning - Sistema de aprendizaje para decisiones de gestión de trades
Registra cada decisión del Trade Manager y evalúa su calidad en retrospectiva
"""
import logging
from typing import Dict, List, Optional
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class TradeManagementLearning:
    """
    Sistema de aprendizaje para Trade Manager.

    Funcionalidades:
    - Registra cada decisión tomada (breakeven, trailing, partial_tp, close)
    - Evalúa la calidad de cada decisión después del cierre del trade
    - Guarda estadísticas para mejorar futuras decisiones
    - Exporta historial para intelligence export
    """

    def __init__(self, max_history: int = 200):
        """
        Args:
            max_history: Número máximo de decisiones a guardar
        """
        self.max_history = max_history
        self.actions_history = []  # Lista de todas las acciones registradas

        # Tracking de acciones por símbolo (antes de que el trade cierre)
        self._pending_evaluations = {}  # {symbol: [actions]}

        # Estadísticas de aprendizaje
        self.stats = {
            'total_actions': 0,
            'total_evaluated': 0,
            'breakeven_good': 0,
            'breakeven_bad': 0,
            'breakeven_neutral': 0,
            'trailing_good': 0,
            'trailing_bad': 0,
            'trailing_neutral': 0,
            'partial_tp_good': 0,
            'partial_tp_bad': 0,
            'partial_tp_neutral': 0,
            'close_good': 0,
            'close_bad': 0,
            'close_neutral': 0,
            'reversal_good': 0,
            'reversal_bad': 0,
            'reversal_neutral': 0,
        }

        logger.info("✅ Trade Management Learning inicializado")
        logger.info(f"   📊 Historial máximo: {max_history} decisiones")

    def record_action(
        self,
        symbol: str,
        action_type: str,
        pnl_at_decision: float,
        pnl_pct_at_decision: float,
        conditions: Dict,
        decision_info: Dict
    ):
        """
        Guarda una acción del Trade Manager para evaluación futura

        Args:
            symbol: Símbolo del trade
            action_type: 'breakeven', 'trailing', 'partial_tp', 'close_adverse', 'reversal'
            pnl_at_decision: P&L en USDT en el momento de la decisión
            pnl_pct_at_decision: P&L en % en el momento de la decisión
            conditions: Condiciones de mercado (del _analyze_market_conditions)
            decision_info: Info de la decisión (confidence, reasons, risk_score)
        """
        try:
            action_record = {
                'symbol': symbol,
                'action_type': action_type,
                'timestamp': datetime.now().isoformat(),
                'pnl_usdt_at_decision': pnl_at_decision,
                'pnl_pct_at_decision': pnl_pct_at_decision,
                'market_conditions': {
                    'reversal_risk': conditions.get('reversal_risk', 0),
                    'confidence': conditions.get('confidence', 0),
                    'should_secure_profits': conditions.get('should_secure_profits', False),
                    'should_let_run': conditions.get('should_let_run', False),
                    'market_regime': conditions.get('market_regime', 'UNKNOWN'),
                    'sentiment_score': conditions.get('sentiment_score', 0),
                },
                'decision_info': {
                    'confidence': decision_info.get('confidence', 0),
                    'risk_score': decision_info.get('risk_score', 0),
                    'reasons': decision_info.get('reasons', []),
                },
                # Será llenado en evaluate_actions()
                'evaluation': None,
                'quality_score': None,
            }

            # Guardar en pending evaluations (por símbolo)
            if symbol not in self._pending_evaluations:
                self._pending_evaluations[symbol] = []

            self._pending_evaluations[symbol].append(action_record)

            # Actualizar stats
            self.stats['total_actions'] += 1

            logger.debug(
                f"📝 Action recorded: {symbol} - {action_type} "
                f"(P&L: {pnl_pct_at_decision:+.2f}%, Conf: {decision_info.get('confidence', 0):.0%})"
            )

        except Exception as e:
            logger.error(f"❌ Error recording action: {e}", exc_info=True)

    def evaluate_actions(
        self,
        symbol: str,
        final_pnl_pct: float,
        highest_pnl_pct_reached: float,
        close_reason: str
    ):
        """
        Evalúa la calidad de todas las acciones tomadas para un símbolo
        después de que el trade cierra

        Args:
            symbol: Símbolo del trade
            final_pnl_pct: P&L final al cerrar (%)
            highest_pnl_pct_reached: P&L máximo alcanzado durante el trade (%)
            close_reason: Razón del cierre (STOP_LOSS, TAKE_PROFIT, etc.)
        """
        try:
            if symbol not in self._pending_evaluations:
                logger.debug(f"No hay acciones pendientes de evaluar para {symbol}")
                return

            actions = self._pending_evaluations[symbol]

            if not actions:
                return

            logger.info(
                f"📊 Evaluando {len(actions)} acción(es) para {symbol} "
                f"(Final P&L: {final_pnl_pct:+.2f}%, Max: {highest_pnl_pct_reached:+.2f}%)"
            )

            for action in actions:
                quality_score, evaluation = self._evaluate_single_action(
                    action,
                    final_pnl_pct,
                    highest_pnl_pct_reached,
                    close_reason
                )

                action['evaluation'] = evaluation
                action['quality_score'] = quality_score
                action['final_pnl_pct'] = final_pnl_pct
                action['highest_pnl_reached'] = highest_pnl_pct_reached
                action['close_reason'] = close_reason

                # Actualizar estadísticas
                self._update_stats(action['action_type'], evaluation)

                # Logging de evaluación
                logger.info(
                    f"   ✓ {action['action_type']}: {evaluation} "
                    f"(score: {quality_score:.2f}, "
                    f"P&L@decision: {action['pnl_pct_at_decision']:+.2f}% → "
                    f"Final: {final_pnl_pct:+.2f}%)"
                )

            # Mover acciones evaluadas al historial permanente
            self.actions_history.extend(actions)

            # Mantener solo las últimas max_history
            if len(self.actions_history) > self.max_history:
                self.actions_history = self.actions_history[-self.max_history:]

            # Limpiar pending evaluations
            del self._pending_evaluations[symbol]

            logger.info(f"✅ Evaluación completada para {symbol}")

        except Exception as e:
            logger.error(f"❌ Error evaluating actions for {symbol}: {e}", exc_info=True)

    def _evaluate_single_action(
        self,
        action: Dict,
        final_pnl_pct: float,
        highest_pnl_pct_reached: float,
        close_reason: str
    ) -> tuple:
        """
        Evalúa una sola acción y retorna (quality_score, evaluation)

        Quality Score: 0-1 (0 = mala decisión, 1 = excelente decisión)
        Evaluation: 'GOOD', 'NEUTRAL', 'BAD'
        """
        action_type = action['action_type']
        pnl_at_decision = action['pnl_pct_at_decision']

        # CRITERIOS DE EVALUACIÓN POR TIPO DE ACCIÓN

        if action_type == 'breakeven':
            # Breakeven: Bueno si evitó pérdidas o aseguró algo de ganancia
            # Malo si dejó de ganar mucho más

            if final_pnl_pct < 0:
                # Trade terminó en pérdida, pero breakeven protegió algo
                return (0.8, 'GOOD')  # Buena decisión, protegió capital

            elif final_pnl_pct < pnl_at_decision:
                # P&L bajó después de breakeven
                return (0.9, 'GOOD')  # Muy buena decisión, evitó caída

            elif highest_pnl_pct_reached > pnl_at_decision * 2:
                # Dejó de ganar mucho más (el doble o más)
                return (0.3, 'BAD')  # Mala decisión, cortó ganancias prematuramente

            else:
                # Trade continuó subiendo un poco
                return (0.6, 'NEUTRAL')  # Neutral, no hizo daño pero pudo ganar más

        elif action_type == 'trailing':
            # Trailing: Bueno si capturó buen movimiento
            # Malo si se ejecutó SL cuando pudo haber ganado más

            if final_pnl_pct > pnl_at_decision * 1.5:
                # Capturó mucho más movimiento
                return (1.0, 'GOOD')  # Excelente decisión

            elif final_pnl_pct > pnl_at_decision:
                # Capturó algo más
                return (0.8, 'GOOD')  # Buena decisión

            elif close_reason == 'STOP_LOSS' and final_pnl_pct < pnl_at_decision * 0.5:
                # SL se ejecutó y perdió mucho del profit
                return (0.4, 'BAD')  # Mala decisión, SL muy apretado

            else:
                return (0.6, 'NEUTRAL')

        elif action_type == 'partial_tp':
            # Partial TP: Bueno si aseguró ganancias y evitó caída
            # Malo si cortó ganancias prematuramente

            if final_pnl_pct < pnl_at_decision * 0.5:
                # Trade cayó significativamente después
                return (0.9, 'GOOD')  # Muy buena decisión, aseguró ganancias

            elif highest_pnl_pct_reached > pnl_at_decision * 2:
                # Dejó de ganar mucho más (cortó ganancias)
                return (0.5, 'BAD')  # Mala decisión en retrospectiva

            else:
                return (0.7, 'GOOD')  # Buena decisión, aseguró algo

        elif action_type in ['close_adverse', 'reversal']:
            # Close: Bueno si evitó pérdidas mayores
            # Malo si cerró prematuramente

            if pnl_at_decision > 0 and final_pnl_pct < 0:
                # Cerró en ganancia, hubiera terminado en pérdida
                return (1.0, 'GOOD')  # Excelente decisión

            elif final_pnl_pct < pnl_at_decision:
                # Evitó caída
                return (0.85, 'GOOD')  # Buena decisión

            elif highest_pnl_pct_reached > pnl_at_decision * 1.5:
                # Cerró prematuramente, hubiera ganado mucho más
                return (0.3, 'BAD')  # Mala decisión

            else:
                return (0.6, 'NEUTRAL')

        # Default
        return (0.5, 'NEUTRAL')

    def _update_stats(self, action_type: str, evaluation: str):
        """Actualiza estadísticas de aprendizaje"""
        self.stats['total_evaluated'] += 1

        stat_key = f"{action_type}_{evaluation.lower()}"
        if stat_key in self.stats:
            self.stats[stat_key] += 1

    def get_statistics(self) -> Dict:
        """
        Retorna estadísticas de aprendizaje

        Returns:
            Dict con estadísticas detalladas
        """
        try:
            stats = self.stats.copy()

            # Calcular tasas de acierto por tipo de acción
            for action_type in ['breakeven', 'trailing', 'partial_tp', 'close', 'reversal']:
                good = stats.get(f'{action_type}_good', 0)
                bad = stats.get(f'{action_type}_bad', 0)
                neutral = stats.get(f'{action_type}_neutral', 0)
                total = good + bad + neutral

                if total > 0:
                    stats[f'{action_type}_success_rate'] = (good / total) * 100
                else:
                    stats[f'{action_type}_success_rate'] = 0

            # Success rate general
            total_evaluated = stats.get('total_evaluated', 0)
            if total_evaluated > 0:
                total_good = sum(stats.get(f'{t}_good', 0) for t in ['breakeven', 'trailing', 'partial_tp', 'close', 'reversal'])
                stats['overall_success_rate'] = (total_good / total_evaluated) * 100
            else:
                stats['overall_success_rate'] = 0

            return stats

        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}

    def get_insights(self) -> List[str]:
        """
        Genera insights basados en el aprendizaje

        Returns:
            Lista de insights legibles
        """
        insights = []

        try:
            stats = self.get_statistics()

            # Insight general
            total_evaluated = stats.get('total_evaluated', 0)
            if total_evaluated > 0:
                overall_rate = stats.get('overall_success_rate', 0)
                insights.append(f"📊 {total_evaluated} decisiones evaluadas, {overall_rate:.1f}% exitosas")

            # Insights por tipo de acción
            for action_type in ['breakeven', 'trailing', 'partial_tp', 'close']:
                rate = stats.get(f'{action_type}_success_rate', 0)
                good = stats.get(f'{action_type}_good', 0)
                bad = stats.get(f'{action_type}_bad', 0)
                total = good + bad + stats.get(f'{action_type}_neutral', 0)

                if total >= 5:  # Solo si hay suficientes datos
                    if rate >= 70:
                        insights.append(f"✅ {action_type.upper()}: Buenas decisiones ({rate:.0f}% exitosas, n={total})")
                    elif rate < 50:
                        insights.append(f"⚠️ {action_type.upper()}: Mejorar decisiones ({rate:.0f}% exitosas, n={total})")

            # Insight sobre acciones más comunes
            most_common = max(
                ['breakeven', 'trailing', 'partial_tp', 'close'],
                key=lambda t: stats.get(f'{t}_good', 0) + stats.get(f'{t}_bad', 0) + stats.get(f'{t}_neutral', 0)
            )
            insights.append(f"📈 Acción más común: {most_common.upper()}")

        except Exception as e:
            logger.error(f"Error generating insights: {e}")

        return insights

    def export_to_json(self) -> Dict:
        """
        Exporta historial completo para guardar en intelligence

        Returns:
            Dict con todo el historial y estadísticas
        """
        try:
            return {
                'statistics': self.get_statistics(),
                'insights': self.get_insights(),
                'actions_history': self.actions_history[-100:],  # Últimas 100 acciones
                'total_actions_recorded': len(self.actions_history),
                'last_update': datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Error exporting to JSON: {e}")
            return {}

    def save_to_file(self, filepath: str = 'data/trade_management_learning.json'):
        """Guarda el historial en archivo JSON"""
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)

            data = self.export_to_json()

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(f"✅ Trade Management Learning guardado en {filepath}")

        except Exception as e:
            logger.error(f"❌ Error guardando learning: {e}")

    def load_from_file(self, filepath: str = 'data/trade_management_learning.json'):
        """Carga el historial desde archivo JSON"""
        try:
            if not Path(filepath).exists():
                logger.info(f"No existe archivo de learning en {filepath}")
                return

            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.stats = data.get('statistics', self.stats)
            self.actions_history = data.get('actions_history', [])

            logger.info(
                f"✅ Trade Management Learning cargado desde {filepath} "
                f"({len(self.actions_history)} acciones)"
            )

        except Exception as e:
            logger.error(f"❌ Error cargando learning: {e}")

    def get_best_practices(self) -> Dict:
        """
        Analiza el historial y retorna mejores prácticas aprendidas

        Returns:
            Dict con recomendaciones basadas en datos
        """
        recommendations = {
            'breakeven': None,
            'trailing': None,
            'partial_tp': None,
            'close': None,
        }

        try:
            # Analizar cada tipo de acción
            for action_type in recommendations.keys():
                actions_of_type = [
                    a for a in self.actions_history
                    if a.get('action_type') == action_type and a.get('evaluation')
                ]

                if len(actions_of_type) < 5:  # Necesitamos datos suficientes
                    continue

                # Calcular P&L promedio al tomar la decisión en decisiones GOOD
                good_actions = [a for a in actions_of_type if a.get('evaluation') == 'GOOD']

                if good_actions:
                    avg_pnl_at_good_decision = sum(
                        a.get('pnl_pct_at_decision', 0) for a in good_actions
                    ) / len(good_actions)

                    avg_confidence = sum(
                        a.get('decision_info', {}).get('confidence', 0) for a in good_actions
                    ) / len(good_actions)

                    recommendations[action_type] = {
                        'optimal_pnl_range': f"{avg_pnl_at_good_decision:.1f}%",
                        'optimal_confidence': f"{avg_confidence:.0%}",
                        'sample_size': len(good_actions),
                    }

        except Exception as e:
            logger.error(f"Error analyzing best practices: {e}")

        return recommendations
