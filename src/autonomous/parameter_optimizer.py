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
        # Sin limitaciones - la IA tiene control TOTAL sobre parámetros listados
        #
        # PARÁMETROS PROTEGIDOS (NO MODIFICABLES POR IA):
        # - PAPER_TRADING_INITIAL_BALANCE: $50,000 USDT (fijo)
        # - STOP_LOSS: Basado en ATR (lógica fija en análisis técnico)
        #
        # PARÁMETROS AHORA OPTIMIZABLES (con límites conservadores):
        # - TAKE_PROFITS: 0.3-2.0% (dinámicos según oportunidad)
        # - News Triggers: thresholds de importance, engagement, social buzz
        # - Multi-Layer Confidence: weights de cada capa
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

            # GROWTH API - News-Triggered Trading (NUEVO)
            'NEWS_IMPORTANCE_THRESHOLD': (0.25, 0.55, 'float'),  # % important votes
            'NEWS_ENGAGEMENT_THRESHOLD': (15, 50, 'int'),  # saves + comments
            'SOCIAL_BUZZ_THRESHOLD': (5, 20, 'int'),  # min social posts
            'RECENT_NEWS_WINDOW_MIN': (3, 15, 'int'),  # minutos para "reciente"
            'PRE_PUMP_MIN_SCORE': (65, 90, 'int'),  # score mínimo para pre-pump signal

            # GROWTH API - Multi-Layer Confidence Weights (NUEVO)
            'IMPORTANCE_WEIGHT': (5, 15, 'int'),  # importance layer weight
            'SOCIAL_BUZZ_WEIGHT': (4, 12, 'int'),  # social buzz layer weight
            'MARKET_CAP_WEIGHT': (3, 8, 'int'),  # market cap layer weight
            'MULTI_LAYER_MIN_ALIGNMENT': (0.5, 0.9, 'float'),  # alineación mínima entre timeframes

            # Dynamic Take Profits (NUEVO - antes fijos)
            'TP1_BASE_PCT': (0.25, 0.5, 'float'),  # TP1 base (scalping)
            'TP2_BASE_PCT': (0.6, 1.2, 'float'),  # TP2 base (medio)
            'TP3_BASE_PCT': (1.0, 2.0, 'float'),  # TP3 base (agresivo)
            'DYNAMIC_TP_MULTIPLIER': (1.0, 2.5, 'float'),  # multiplicador en oportunidades críticas
            'HIGH_CRITICALITY_THRESHOLD': (80, 95, 'int'),  # score para usar TPs altos

            # Fear & Greed Index Thresholds (NUEVO)
            'FEAR_GREED_EXTREME_THRESHOLD': (15, 30, 'int'),  # threshold para miedo/greed extremo
            'FEAR_GREED_OPPORTUNITY_BOOST': (1.1, 2.0, 'float'),  # boost en miedo extremo

            # ML Confidence Thresholds (NUEVO)
            'ML_CONFIDENCE_THRESHOLD': (0.55, 0.85, 'float'),  # confianza mínima para usar ML predictions

            # ===== NUEVOS PARÁMETROS - AUTONOMÍA 100% =====

            # Smart Order Routing (Spot vs Futures dinámico) - 7 parámetros
            'MIN_CONFIDENCE_FOR_FUTURES': (60.0, 85.0, 'float'),  # % confianza mínima para usar futures
            'MIN_WINRATE_FOR_FUTURES': (45.0, 65.0, 'float'),  # % win rate mínimo para usar futures
            'MAX_DRAWDOWN_FOR_FUTURES': (5.0, 15.0, 'float'),  # % drawdown máximo para permitir futures
            'VOLATILITY_THRESHOLD_FUTURES': (0.015, 0.03, 'float'),  # volatilidad mínima para futures
            'CONSERVATIVE_LEVERAGE': (2, 5, 'int'),  # leverage conservador
            'BALANCED_LEVERAGE': (5, 10, 'int'),  # leverage balanceado
            'AGGRESSIVE_LEVERAGE': (10, 20, 'int'),  # leverage agresivo

            # Trailing Stops Automáticos - 4 parámetros
            'TRAILING_DISTANCE_PCT': (0.3, 0.7, 'float'),  # distancia del trailing stop
            'BREAKEVEN_AFTER_PCT': (0.3, 1.0, 'float'),  # profit para activar breakeven
            'LOCK_PROFIT_STEP_PCT': (0.3, 0.8, 'float'),  # step de lock de profits
            'MIN_PROFIT_TO_LOCK_PCT': (0.2, 0.5, 'float'),  # mínimo profit para empezar trailing

            # Position Sizing Agresivo - 1 parámetro (ya existe BASE_POSITION_SIZE_PCT pero ampliar max)
            'MAX_POSITION_SIZE_PCT': (8.0, 12.0, 'float'),  # máximo position size (antes 8%, ahora hasta 12%)

            # Anomaly Detection System - 4 parámetros
            'PERFORMANCE_DEGRADATION_THRESHOLD': (5.0, 20.0, 'float'),  # % degradación para alertar
            'OUTLIER_STD_THRESHOLD': (2.0, 4.0, 'float'),  # desviaciones estándar para outliers
            'MIN_TRADES_FOR_DETECTION': (10, 50, 'int'),  # mínimo trades para detectar anomalías
            'ANOMALY_LOOKBACK_WINDOW': (30, 100, 'int'),  # ventana de trades a considerar

            # A/B Testing de Estrategias - 5 parámetros
            'AB_TEST_DURATION_TRADES': (30, 100, 'int'),  # duración del A/B test en trades
            'AB_TEST_DURATION_DAYS': (3, 14, 'int'),  # duración del A/B test en días
            'AB_TEST_CAPITAL_SPLIT': (0.3, 0.7, 'float'),  # split de capital entre estrategias
            'AB_TEST_MIN_CONFIDENCE': (0.7, 0.95, 'float'),  # confianza mínima para switch

            # TOTAL: 62 parámetros optimizables (+21 nuevos)
            'ML_HIGH_CONFIDENCE_THRESHOLD': (0.75, 0.95, 'float'),  # confianza para boost agresivo
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
        """
        Genera explicación DETALLADA de por qué se hicieron los cambios
        Incluye: diagnóstico, objetivo, cambios específicos, y expectativa de resultado
        """
        if not changes:
            return "Sin cambios - configuración óptima mantenida"

        win_rate = performance.get('win_rate', 0)
        roi = performance.get('roi', 0)
        sharpe = performance.get('sharpe_ratio', 0)
        drawdown = performance.get('max_drawdown', 0)
        total_trades = performance.get('total_trades', 0)

        reasons = []

        # SECCIÓN 1: DIAGNÓSTICO DE PERFORMANCE ACTUAL
        reasons.append("=== DIAGNÓSTICO ===")

        # Análisis de win rate
        if win_rate < 40:
            reasons.append(f"⚠️ Win Rate CRÍTICO: {win_rate:.1f}% (objetivo: 50%+)")
            reasons.append("   → Problema: Demasiados trades perdedores, señales de baja calidad")
        elif win_rate < 50:
            reasons.append(f"⚠️ Win Rate BAJO: {win_rate:.1f}% (objetivo: 50%+)")
            reasons.append("   → Necesita ajustar selectividad de señales")
        elif win_rate > 70:
            reasons.append(f"✅ Win Rate EXCELENTE: {win_rate:.1f}%")
            reasons.append("   → Podemos ser más agresivos para aumentar frecuencia")
        else:
            reasons.append(f"✅ Win Rate SALUDABLE: {win_rate:.1f}%")

        # Análisis de ROI
        if roi < -5:
            reasons.append(f"🚨 ROI MUY NEGATIVO: {roi:.2f}% - REDUCIR RIESGO URGENTE")
        elif roi < 0:
            reasons.append(f"⚠️ ROI NEGATIVO: {roi:.2f}% - Estrategia necesita ajustes")
        elif roi > 10:
            reasons.append(f"🎉 ROI EXCELENTE: {roi:.2f}% - Estrategia funcionando muy bien")
        else:
            reasons.append(f"ROI ACTUAL: {roi:.2f}%")

        # Análisis de drawdown
        if drawdown > 15:
            reasons.append(f"⚠️ Drawdown ALTO: {drawdown:.1f}% - Reducir tamaño de posiciones")
        elif drawdown > 10:
            reasons.append(f"⚠️ Drawdown MODERADO: {drawdown:.1f}%")

        # SECCIÓN 2: ESTRATEGIA Y OBJETIVO
        reasons.append("\n=== ESTRATEGIA ===")
        if strategy == "EXPLORATION":
            reasons.append("🔍 EXPLORACIÓN: Probando configuraciones nuevas para descubrir mejores setups")
            reasons.append(f"   → Trials completados: {self.total_trials}")
            reasons.append("   → Objetivo: Salir de óptimos locales y encontrar mejores configuraciones")
        else:
            reasons.append("🎯 OPTIMIZACIÓN: Refinando configuración basado en resultados previos")
            reasons.append(f"   → Usando aprendizajes de {len(self.trial_history)} trials anteriores")
            reasons.append("   → Objetivo: Mejorar configuración actual incrementalmente")

        # SECCIÓN 3: CAMBIOS ESPECÍFICOS CON RAZONAMIENTO
        reasons.append("\n=== CAMBIOS REALIZADOS ===")
        reasons.append(f"Total de parámetros modificados: {len(changes)}\n")

        # Agrupar cambios por categoría
        risk_changes = [c for c in changes if any(x in c['parameter'] for x in ['RISK', 'POSITION_SIZE', 'DRAWDOWN'])]
        indicator_changes = [c for c in changes if any(x in c['parameter'] for x in ['RSI', 'MACD', 'EMA', 'BB'])]
        threshold_changes = [c for c in changes if 'THRESHOLD' in c['parameter']]
        ml_changes = [c for c in changes if any(x in c['parameter'] for x in ['ESTIMATORS', 'DEPTH', 'LEARNING'])]

        if risk_changes:
            reasons.append("📊 GESTIÓN DE RIESGO:")
            for change in risk_changes[:3]:  # Top 3
                param = change['parameter']
                old, new = change['old_value'], change['new_value']

                # Validar que old y new no sean None antes de comparar
                if old is not None and new is not None:
                    direction = "↑" if new > old else "↓"
                else:
                    direction = "→"

                reasons.append(f"   {direction} {param}: {old} → {new}")
                if 'POSITION_SIZE' in param:
                    if old is not None and new is not None and new > old:
                        reasons.append("      Razón: Incrementar exposición en mercado favorable")
                    elif old is not None and new is not None:
                        reasons.append("      Razón: Reducir exposición para proteger capital")

        if indicator_changes:
            reasons.append("\n📈 INDICADORES TÉCNICOS:")
            for change in indicator_changes[:3]:
                param = change['parameter']
                old, new = change['old_value'], change['new_value']

                # Validar que old y new no sean None antes de comparar
                if old is not None and new is not None:
                    direction = "↑" if new > old else "↓"
                else:
                    direction = "→"

                reasons.append(f"   {direction} {param}: {old} → {new}")
                if 'RSI' in param:
                    reasons.append("      Razón: Ajustar sensibilidad a sobrecompra/sobreventa")
                elif 'MACD' in param:
                    reasons.append("      Razón: Mejorar detección de cambios de tendencia")

        if threshold_changes:
            reasons.append("\n🎯 UMBRALES DE SEÑALES:")
            for change in threshold_changes[:3]:
                param = change['parameter']
                old, new = change['old_value'], change['new_value']

                # Validar que old y new no sean None antes de comparar
                if old is not None and new is not None:
                    direction = "↑" if new > old else "↓"
                else:
                    direction = "→"

                reasons.append(f"   {direction} {param}: {old} → {new}")
                if old is not None and new is not None:
                    if new > old:
                        reasons.append("      Razón: Aumentar selectividad - solo señales de mayor calidad")
                    else:
                        reasons.append("      Razón: Reducir selectividad - aumentar frecuencia de trades")

        if ml_changes:
            reasons.append("\n🧠 MODELO MACHINE LEARNING:")
            for change in ml_changes[:2]:
                param = change['parameter']
                old, new = change['old_value'], change['new_value']
                reasons.append(f"   • {param}: {old} → {new}")
            reasons.append("      Razón: Ajustar complejidad y capacidad de aprendizaje del modelo")

        # SECCIÓN 4: EXPECTATIVA
        reasons.append("\n=== RESULTADO ESPERADO ===")
        if win_rate < 45:
            reasons.append("🎯 Objetivo inmediato: Aumentar win rate a 50%+")
            reasons.append("   → Aumentando selectividad de señales")
            reasons.append("   → Mejorando precisión de indicadores")
        elif roi < 0:
            reasons.append("🎯 Objetivo inmediato: Revertir a ROI positivo")
            reasons.append("   → Reduciendo tamaño de posiciones")
            reasons.append("   → Protegiendo capital con mejor gestión de riesgo")
        else:
            reasons.append("🎯 Objetivo: Optimizar estrategia exitosa para maximizar retornos")
            reasons.append("   → Manteniendo lo que funciona")
            reasons.append("   → Refinando parámetros para mejor performance")

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
                # Filtrar valores inválidos (inf, -inf, nan) antes de correlación
                valid_indices = [
                    i for i, score in enumerate(scores)
                    if np.isfinite(score)  # Filtra inf, -inf, y nan
                ]

                if len(valid_indices) >= 5:  # Necesitamos al menos 5 valores válidos
                    valid_param_values = [param_values[i] for i in valid_indices]
                    valid_scores = [scores[i] for i in valid_indices]

                    # Correlación simple con valores válidos
                    try:
                        correlation = np.corrcoef(valid_param_values, valid_scores)[0, 1]
                        if not np.isnan(correlation):
                            # Actualizar importancia (promedio móvil)
                            self.parameter_importance[param] = (
                                0.7 * self.parameter_importance[param] +
                                0.3 * abs(correlation)
                            )
                    except (ValueError, RuntimeWarning):
                        # Si falla el cálculo, simplemente skip
                        pass

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

    def analyze_performance(self, current_metrics: Dict) -> Dict:
        """
        🧠 Analiza performance actual y sugiere cambios en tiempo real

        Esta función se llama desde DecisionBrain para obtener sugerencias
        de ajustes de parámetros basadas en el rendimiento actual.

        Args:
            current_metrics: Dict con métricas actuales
                - win_rate: % de trades ganadores
                - avg_profit: promedio de ganancia por trade
                - total_trades: total de trades ejecutados
                - CONSERVATIVE_THRESHOLD: threshold actual (si disponible)
                - TP1_BASE_PCT: take profit base actual (si disponible)

        Returns:
            Dict con sugerencias:
                - immediate_changes: cambios a aplicar inmediatamente
                - recommended_changes: cambios sugeridos para considerar
                - performance_analysis: análisis del rendimiento
        """
        suggestions = {
            'immediate_changes': {},
            'recommended_changes': {},
            'performance_analysis': {}
        }

        win_rate = current_metrics.get('win_rate', 0)
        avg_profit = current_metrics.get('avg_profit', 0)
        total_trades = current_metrics.get('total_trades', 0)

        # No sugerir cambios si no hay suficientes trades
        if total_trades < 10:
            suggestions['performance_analysis']['status'] = 'insufficient_data'
            suggestions['performance_analysis']['message'] = 'Necesitan más trades para análisis'
            return suggestions

        # ========================================
        # ANÁLISIS DE WIN RATE
        # ========================================
        if win_rate < 50:
            # Win rate muy bajo - ser más selectivo
            suggestions['performance_analysis']['win_rate_status'] = 'critical'
            suggestions['immediate_changes']['CONSERVATIVE_THRESHOLD'] = min(
                self.parameter_ranges.get('CONSERVATIVE_THRESHOLD', (3.0, 8.0, 'float'))[1],
                current_metrics.get('CONSERVATIVE_THRESHOLD', 5.0) + 0.5
            )
            suggestions['recommended_changes']['FLASH_MIN_CONFIDENCE'] = min(
                self.parameter_ranges.get('FLASH_MIN_CONFIDENCE', (40, 80, 'int'))[1],
                current_metrics.get('FLASH_MIN_CONFIDENCE', 50) + 5
            )

        elif win_rate < 60:
            # Win rate bajo - ajustar moderadamente
            suggestions['performance_analysis']['win_rate_status'] = 'low'
            suggestions['recommended_changes']['CONSERVATIVE_THRESHOLD'] = min(
                self.parameter_ranges.get('CONSERVATIVE_THRESHOLD', (3.0, 8.0, 'float'))[1],
                current_metrics.get('CONSERVATIVE_THRESHOLD', 5.0) + 0.3
            )

        elif win_rate > 80:
            # Win rate alto - podemos ser más agresivos
            suggestions['performance_analysis']['win_rate_status'] = 'excellent'
            suggestions['recommended_changes']['CONSERVATIVE_THRESHOLD'] = max(
                self.parameter_ranges.get('CONSERVATIVE_THRESHOLD', (3.0, 8.0, 'float'))[0],
                current_metrics.get('CONSERVATIVE_THRESHOLD', 5.0) - 0.3
            )

        else:
            suggestions['performance_analysis']['win_rate_status'] = 'good'

        # ========================================
        # ANÁLISIS DE PROFITABILIDAD
        # ========================================
        if avg_profit < 0.3:
            # Profits muy bajos - TPs más cercanos
            suggestions['performance_analysis']['profit_status'] = 'low'
            suggestions['recommended_changes']['TP1_BASE_PCT'] = max(
                0.3,  # Mínimo 0.3%
                current_metrics.get('TP1_BASE_PCT', 0.5) * 0.9
            )

        elif avg_profit > 1.5:
            # Profits buenos - podemos extender TPs
            suggestions['performance_analysis']['profit_status'] = 'excellent'
            suggestions['recommended_changes']['TP1_BASE_PCT'] = min(
                1.5,  # Máximo 1.5%
                current_metrics.get('TP1_BASE_PCT', 0.5) * 1.1
            )

        else:
            suggestions['performance_analysis']['profit_status'] = 'good'

        # ========================================
        # RESUMEN
        # ========================================
        suggestions['performance_analysis']['summary'] = {
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'total_trades': total_trades,
            'changes_suggested': len(suggestions['immediate_changes']) + len(suggestions['recommended_changes'])
        }

        if suggestions['immediate_changes']:
            logger.info(
                f"📊 Parameter Optimizer: {len(suggestions['immediate_changes'])} cambios inmediatos sugeridos "
                f"(Win rate: {win_rate:.1f}%, Avg profit: {avg_profit:.2f}%)"
            )

        return suggestions

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

    def load_from_dict(self, data: Dict, merge: bool = False):
        """
        Carga optimizador desde persistencia

        Args:
            data: Diccionario con estado del optimizador
            merge: Si True, combina con datos existentes (acumula trials y actualiza bests)
                   Si False, reemplaza completamente (comportamiento por defecto)
        """
        if merge:
            # MODO MERGE: Combinar trials y actualizar mejores configuraciones
            loaded_trials = data.get('trial_history', [])
            loaded_best_config = data.get('best_config', {})
            loaded_best_performance = data.get('best_performance', -float('inf'))
            loaded_total_trials = data.get('total_trials', 0)
            loaded_param_importance = data.get('parameter_importance', {})

            # Agregar trials históricos
            self.trial_history.extend(loaded_trials)

            # Mantener solo últimos 1000 trials
            if len(self.trial_history) > 1000:
                self.trial_history = self.trial_history[-1000:]

            # Actualizar mejor configuración si la importada es mejor
            if loaded_best_performance > self.best_performance:
                self.best_config = loaded_best_config
                self.best_performance = loaded_best_performance
                logger.info(f"🏆 Nueva mejor configuración importada: score {loaded_best_performance:.3f}")

            # Acumular total de trials
            self.total_trials += loaded_total_trials

            # Merge parameter importance (promediar)
            for param, importance in loaded_param_importance.items():
                if param in self.parameter_importance:
                    # Promediar importancia
                    self.parameter_importance[param] = (
                        self.parameter_importance[param] * 0.5 + importance * 0.5
                    )
                else:
                    self.parameter_importance[param] = importance

            logger.info(
                f"🔄 Parameter Optimizer MERGED: {loaded_total_trials} trials importados, "
                f"Total acumulado: {self.total_trials} trials, "
                f"mejor score: {self.best_performance:.3f}"
            )
        else:
            # MODO REPLACE: Reemplazar completamente (comportamiento original)
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
