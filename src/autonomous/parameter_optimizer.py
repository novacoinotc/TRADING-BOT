"""
Parameter Optimizer - Optimización autónoma de TODOS los parámetros
Usa búsqueda adaptativa y meta-learning para encontrar configuraciones óptimas
"""
import numpy as np
import logging
from typing import Dict, List, Any, Tuple
from datetime import datetime
import copy

logger = logging.getLogger(__name__)


class ParameterOptimizer:
    """
    Optimizador autónomo de parámetros
    - Modifica TODOS los parámetros sin limitaciones
    - Usa búsqueda inteligente basada en resultados
    - Aprende qué cambios funcionan mejor
    - Notifica cada modificación a Telegram
    """

    def __init__(self):
        """Inicializa optimizador con rangos de búsqueda para cada parámetro"""

        # Definir rangos de búsqueda para TODOS los parámetros
        # Sin limitaciones - la IA tiene control TOTAL
        # NOTA: Balance inicial (PAPER_TRADING_INITIAL_BALANCE) está PROTEGIDO - nunca se modifica
        self.parameter_ranges = {
            # Trading Configuration
            'CHECK_INTERVAL': (60, 300, 'int'),  # 1-5 minutos
            'CONSERVATIVE_THRESHOLD': (3.0, 8.0, 'float'),  # Score threshold
            'FLASH_THRESHOLD': (4.0, 8.0, 'float'),
            'FLASH_MIN_CONFIDENCE': (40, 80, 'int'),
            'PROFIT_THRESHOLD': (0.5, 3.0, 'float'),  # % profit target

            # Technical Indicators
            'RSI_OVERSOLD': (20, 40, 'int'),
            'RSI_OVERBOUGHT': (60, 80, 'int'),
            'RSI_PERIOD': (7, 21, 'int'),
            'MACD_FAST': (8, 16, 'int'),
            'MACD_SLOW': (20, 30, 'int'),
            'MACD_SIGNAL': (7, 12, 'int'),
            'EMA_SHORT': (5, 15, 'int'),
            'EMA_MEDIUM': (15, 30, 'int'),
            'EMA_LONG': (40, 60, 'int'),
            'BB_PERIOD': (15, 25, 'int'),
            'BB_STD': (1.5, 2.5, 'float'),

            # Risk Management
            'BASE_POSITION_SIZE_PCT': (2.0, 8.0, 'float'),
            'MAX_DRAWDOWN_LIMIT': (10.0, 25.0, 'float'),
            'MAX_POSITIONS': (5, 12, 'int'),
            'MAX_RISK_PER_TRADE_PCT': (1.0, 3.0, 'float'),

            # ML Model Hyperparameters
            'N_ESTIMATORS': (100, 300, 'int'),
            'MAX_DEPTH': (3, 7, 'int'),
            'LEARNING_RATE': (0.01, 0.15, 'float'),
            'SUBSAMPLE': (0.6, 0.9, 'float'),
            'COLSAMPLE_BYTREE': (0.6, 0.9, 'float'),
            'MIN_CHILD_WEIGHT': (1, 5, 'int'),
            'GAMMA': (0.0, 0.3, 'float'),
            'REG_ALPHA': (0.0, 0.3, 'float'),
            'REG_LAMBDA': (0.5, 2.0, 'float'),
        }

        # Historial de configuraciones probadas y sus resultados
        self.trial_history: List[Dict] = []

        # Mejor configuración encontrada hasta ahora
        self.best_config: Dict = {}
        self.best_performance: float = -float('inf')

        # Configuración actual
        self.current_config: Dict = {}

        # Número de trials realizados
        self.total_trials = 0

        # Learning: trackear qué parámetros tienen mayor impacto
        self.parameter_importance: Dict[str, float] = {
            param: 1.0 for param in self.parameter_ranges.keys()
        }

        logger.info(f"🎯 Parameter Optimizer inicializado con {len(self.parameter_ranges)} parámetros optimizables")

    def suggest_parameter_changes(
        self,
        current_performance: Dict,
        exploration_factor: float = 0.3
    ) -> Dict[str, Any]:
        """
        Sugiere cambios de parámetros basado en performance actual

        Args:
            current_performance: Métricas actuales (win_rate, roi, sharpe, etc.)
            exploration_factor: Qué tan agresivo explorar (0-1)

        Returns:
            Dict con parámetros sugeridos y razones
        """
        # Extraer métrica clave de performance
        performance_score = self._calculate_performance_score(current_performance)

        logger.info(f"📊 Performance actual: {performance_score:.3f}")

        # Decidir estrategia: explorar vs optimizar
        if np.random.random() < exploration_factor or self.total_trials < 10:
            # EXPLORAR: cambios aleatorios para descubrir nuevas configuraciones
            new_config = self._generate_random_config()
            strategy = "EXPLORATION"
            logger.info("🔍 Estrategia: EXPLORACIÓN (búsqueda aleatoria)")
        else:
            # OPTIMIZAR: cambios inteligentes basados en historial
            new_config = self._generate_optimized_config(current_performance)
            strategy = "OPTIMIZATION"
            logger.info("🎯 Estrategia: OPTIMIZACIÓN (basado en historial)")

        # Identificar qué parámetros cambiaron
        changes = self._identify_changes(self.current_config, new_config)

        # Registrar trial
        self.total_trials += 1

        return {
            'config': new_config,
            'changes': changes,
            'strategy': strategy,
            'trial_number': self.total_trials,
            'reason': self._generate_change_reason(changes, strategy, current_performance)
        }

    def _calculate_performance_score(self, metrics: Dict) -> float:
        """
        Calcula score compuesto de performance
        Combina múltiples métricas en un solo valor

        Args:
            metrics: Dict con métricas (win_rate, roi, sharpe_ratio, etc.)

        Returns:
            Score de performance (mayor es mejor)
        """
        # Extraer métricas clave
        win_rate = metrics.get('win_rate', 0) / 100.0  # 0-1
        roi = metrics.get('roi', 0) / 100.0  # Normalized
        sharpe = metrics.get('sharpe_ratio', 0)
        profit_factor = metrics.get('profit_factor', 1.0)
        max_drawdown = abs(metrics.get('max_drawdown', 0)) / 100.0  # 0-1

        # Ponderación de métricas
        score = (
            win_rate * 0.25 +           # 25% win rate
            roi * 0.30 +                # 30% ROI
            sharpe * 0.15 +             # 15% Sharpe ratio
            (profit_factor - 1) * 0.20 + # 20% Profit factor
            (1 - max_drawdown) * 0.10   # 10% drawdown (invertido)
        )

        return score

    def _generate_random_config(self) -> Dict[str, Any]:
        """Genera configuración aleatoria dentro de los rangos permitidos"""
        config = {}

        for param, (min_val, max_val, dtype) in self.parameter_ranges.items():
            if dtype == 'int':
                config[param] = np.random.randint(min_val, max_val + 1)
            elif dtype == 'float':
                config[param] = np.random.uniform(min_val, max_val)

        return config

    def _generate_optimized_config(self, current_performance: Dict) -> Dict[str, Any]:
        """
        Genera configuración optimizada basada en historial
        Usa meta-learning para identificar parámetros más importantes
        """
        if not self.best_config:
            return self._generate_random_config()

        # Partir de la mejor configuración conocida
        config = copy.deepcopy(self.best_config)

        # Modificar 2-4 parámetros basado en importancia
        num_changes = np.random.randint(2, 5)

        # Seleccionar parámetros a modificar (priorizando los más importantes)
        params_by_importance = sorted(
            self.parameter_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        params_to_modify = [p[0] for p in params_by_importance[:num_changes]]

        # Modificar parámetros seleccionados
        for param in params_to_modify:
            if param not in self.parameter_ranges:
                continue

            min_val, max_val, dtype = self.parameter_ranges[param]
            current_val = config.get(param, (min_val + max_val) / 2)

            # Perturbación adaptativa (±20% del rango)
            range_size = max_val - min_val
            perturbation = np.random.uniform(-0.2, 0.2) * range_size

            if dtype == 'int':
                new_val = int(np.clip(current_val + perturbation, min_val, max_val))
            else:
                new_val = np.clip(current_val + perturbation, min_val, max_val)

            config[param] = new_val

        return config

    def _identify_changes(self, old_config: Dict, new_config: Dict) -> List[Dict]:
        """Identifica qué parámetros cambiaron y en cuánto"""
        changes = []

        for param, new_val in new_config.items():
            old_val = old_config.get(param)
            if old_val != new_val:
                change_pct = 0
                if old_val and old_val != 0:
                    change_pct = ((new_val - old_val) / abs(old_val)) * 100

                changes.append({
                    'parameter': param,
                    'old_value': old_val,
                    'new_value': new_val,
                    'change_pct': change_pct
                })

        return changes

    def _generate_change_reason(self, changes: List[Dict], strategy: str,
                                performance: Dict) -> str:
        """Genera explicación humana de por qué se hicieron los cambios"""
        if not changes:
            return "Sin cambios - configuración óptima mantenida"

        win_rate = performance.get('win_rate', 0)
        roi = performance.get('roi', 0)

        reasons = []

        # Razón basada en performance
        if win_rate < 45:
            reasons.append(f"Win rate bajo ({win_rate:.1f}%) - ajustando selectividad")
        elif win_rate > 60:
            reasons.append(f"Win rate alto ({win_rate:.1f}%) - aumentando agresividad")

        if roi < -2:
            reasons.append(f"ROI negativo ({roi:.1f}%) - reduciendo riesgo")
        elif roi > 5:
            reasons.append(f"ROI positivo ({roi:.1f}%) - manteniendo estrategia ganadora")

        # Razón basada en estrategia
        if strategy == "EXPLORATION":
            reasons.append("Explorando nuevas configuraciones para encontrar óptimos")
        else:
            reasons.append("Optimizando configuración basado en mejores resultados previos")

        # Top 3 cambios más significativos
        top_changes = sorted(changes, key=lambda x: abs(x['change_pct']), reverse=True)[:3]
        for change in top_changes:
            param = change['parameter']
            old = change['old_value']
            new = change['new_value']
            reasons.append(f"  • {param}: {old} → {new}")

        return "\n".join(reasons)

    def record_trial_result(self, config: Dict, performance: Dict):
        """
        Registra resultado de un trial y actualiza aprendizaje

        Args:
            config: Configuración probada
            performance: Métricas obtenidas
        """
        performance_score = self._calculate_performance_score(performance)

        # Guardar en historial
        self.trial_history.append({
            'trial_number': self.total_trials,
            'config': config,
            'performance': performance,
            'score': performance_score,
            'timestamp': datetime.now().isoformat()
        })

        # Actualizar mejor configuración si superó la anterior
        if performance_score > self.best_performance:
            improvement = performance_score - self.best_performance
            self.best_performance = performance_score
            self.best_config = copy.deepcopy(config)

            logger.info(
                f"🎉 NUEVA MEJOR CONFIGURACIÓN! Score: {performance_score:.3f} "
                f"(+{improvement:.3f} mejora)"
            )

            # Notificación especial para mejoras
            return {
                'improved': True,
                'improvement': improvement,
                'new_best_score': performance_score
            }

        # Actualizar importancia de parámetros (meta-learning)
        self._update_parameter_importance(config, performance_score)

        return {
            'improved': False,
            'score': performance_score,
            'best_score': self.best_performance
        }

    def _update_parameter_importance(self, config: Dict, score: float):
        """
        Actualiza importancia de parámetros basado en correlación con performance
        Meta-learning: aprende qué parámetros importan más
        """
        if len(self.trial_history) < 5:
            return  # Necesita al menos 5 trials para aprender

        # Para cada parámetro, calcular correlación con score
        for param in self.parameter_ranges.keys():
            param_values = []
            scores = []

            for trial in self.trial_history[-20:]:  # Últimos 20 trials
                if param in trial['config']:
                    param_values.append(trial['config'][param])
                    scores.append(trial['score'])

            if len(param_values) >= 5:
                # Correlación simple
                correlation = np.corrcoef(param_values, scores)[0, 1]
                if not np.isnan(correlation):
                    # Actualizar importancia (promedio móvil)
                    self.parameter_importance[param] = (
                        0.7 * self.parameter_importance[param] +
                        0.3 * abs(correlation)
                    )

    def get_optimization_statistics(self) -> Dict:
        """Retorna estadísticas del proceso de optimización"""
        if not self.trial_history:
            return {
                'total_trials': 0,
                'best_score': 0,
                'improvement_rate': 0,
                'top_parameters': []
            }

        # Calcular tasa de mejora
        recent_scores = [t['score'] for t in self.trial_history[-10:]]
        improvement_rate = np.mean(np.diff(recent_scores)) if len(recent_scores) > 1 else 0

        # Top parámetros por importancia
        top_params = sorted(
            self.parameter_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        return {
            'total_trials': self.total_trials,
            'best_score': self.best_performance,
            'current_score': self.trial_history[-1]['score'] if self.trial_history else 0,
            'improvement_rate': improvement_rate,
            'top_parameters': top_params,
            'exploration_exhaustion': min(self.total_trials / 100.0, 1.0)  # 0-1
        }

    def save_to_dict(self) -> Dict:
        """Exporta optimizador para persistencia"""
        return {
            'trial_history': self.trial_history,
            'best_config': self.best_config,
            'best_performance': self.best_performance,
            'current_config': self.current_config,
            'total_trials': self.total_trials,
            'parameter_importance': self.parameter_importance,
            'timestamp': datetime.now().isoformat()
        }

    def load_from_dict(self, data: Dict):
        """Carga optimizador desde persistencia"""
        self.trial_history = data.get('trial_history', [])
        self.best_config = data.get('best_config', {})
        self.best_performance = data.get('best_performance', -float('inf'))
        self.current_config = data.get('current_config', {})
        self.total_trials = data.get('total_trials', 0)
        self.parameter_importance = data.get('parameter_importance', {})

        logger.info(
            f"✅ Parameter Optimizer cargado: {self.total_trials} trials, "
            f"mejor score: {self.best_performance:.3f}"
        )
