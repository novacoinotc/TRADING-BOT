"""
Decision Brain - Cerebro central que coordina los 4 sistemas de decisión
Integra: RL Agent, ML Predictor, Trade Manager, Parameter Optimizer
Usa: Los 24 servicios disponibles en tiempo real

🧠 ARQUITECTURA:
- Recopila datos de TODOS los servicios
- ML Predictor analiza patrones
- RL Agent decide acción
- Trade Manager gestiona posiciones
- Parameter Optimizer ajusta parámetros
- Todos aprenden de cada trade
"""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class DecisionBrain:
    """
    🧠 CEREBRO CENTRAL DE DECISIONES
    Coordina todos los sistemas y servicios para decisiones óptimas
    """

    def __init__(
        self,
        rl_agent=None,
        ml_system=None,
        trade_manager=None,
        parameter_optimizer=None,
        feature_aggregator=None,
        sentiment_analyzer=None,
        regime_detector=None,
        orderbook_analyzer=None
    ):
        """
        Inicializa el cerebro central con todos los componentes

        Args:
            rl_agent: Agente de Reinforcement Learning
            ml_system: Sistema de ML para predicciones
            trade_manager: Gestor de trades activos
            parameter_optimizer: Optimizador de parámetros
            feature_aggregator: Agregador de features (Arsenal)
            sentiment_analyzer: Analizador de sentimiento
            regime_detector: Detector de régimen de mercado
            orderbook_analyzer: Analizador de orderbook
        """
        self.rl_agent = rl_agent
        self.ml_system = ml_system
        self.trade_manager = trade_manager
        self.parameter_optimizer = parameter_optimizer
        self.feature_aggregator = feature_aggregator
        self.sentiment_analyzer = sentiment_analyzer
        self.regime_detector = regime_detector
        self.orderbook_analyzer = orderbook_analyzer

        # Contador de análisis
        self.analysis_count = 0
        self.trades_analyzed = 0

        # Historial de decisiones para aprendizaje
        self.decision_history: List[Dict] = []
        self.max_history = 500

        # Trade experiences para ML
        self.trade_experiences: List[Dict] = []

        # Métricas de rendimiento
        self.performance_metrics = {
            'total_decisions': 0,
            'trades_executed': 0,
            'profitable_trades': 0,
            'total_pnl': 0.0,
            'services_used': {}
        }

        # 🎯 SCALPING: Sistema de pesos dinámicos por rendimiento de cada IA
        # Tracking de aciertos de cada sistema para ajustar pesos automáticamente
        self.ia_performance = {
            'ml_system': {
                'total_predictions': 0,
                'correct_predictions': 0,
                'accuracy': 0.50,
                'recent_predictions': [],  # Últimas 50 predicciones
                'weight': 0.40  # Peso inicial
            },
            'rl_agent': {
                'total_predictions': 0,
                'correct_predictions': 0,
                'accuracy': 0.50,
                'recent_predictions': [],  # Últimas 50 predicciones
                'weight': 0.60  # Peso inicial
            },
            'trade_manager': {
                'total_actions': 0,
                'successful_actions': 0,
                'accuracy': 0.50,
                'recent_actions': [],  # Últimas 50 acciones
                'weight': 0.50  # Para tie-breakers
            }
        }
        self.max_recent_items = 50  # Máximo de items en listas recientes

        # Lista de servicios activos
        self._initialize_services()

        logger.info(f"🧠 Decision Brain inicializado")

    def _initialize_services(self):
        """Inicializa y cuenta los servicios disponibles"""
        self.active_services = {}

        # Servicios de análisis técnico (Arsenal/FeatureAggregator)
        if self.feature_aggregator:
            # Los módulos del Arsenal son atributos directos, no un dict 'modules'
            arsenal_module_names = [
                'correlation_matrix',
                'liquidation_heatmap',
                'funding_rate_analyzer',
                'volume_profile',
                'pattern_recognition',
                'session_trading',
                'order_flow'
            ]

            for name in arsenal_module_names:
                module = getattr(self.feature_aggregator, name, None)
                if module is not None:
                    self.active_services[f'arsenal_{name}'] = module
                    logger.debug(f"   ✅ Arsenal módulo: {name}")

            # También verificar los métodos wrapper que añadimos
            wrapper_methods = [
                'get_liquidation_levels',
                'get_funding_analysis',
                'get_volume_profile',
                'detect_patterns',
                'analyze_order_flow',
                'get_session_analysis',
                'get_correlations'
            ]

            available_wrappers = 0
            for method in wrapper_methods:
                if hasattr(self.feature_aggregator, method):
                    available_wrappers += 1

            logger.info(f"   🔧 Arsenal wrappers disponibles: {available_wrappers}/{len(wrapper_methods)}")

        # Servicios principales
        if self.sentiment_analyzer:
            self.active_services['sentiment'] = self.sentiment_analyzer
            logger.debug(f"   ✅ Servicio: sentiment_analyzer")

        if self.regime_detector:
            self.active_services['regime'] = self.regime_detector
            logger.debug(f"   ✅ Servicio: regime_detector")

        if self.orderbook_analyzer:
            self.active_services['orderbook'] = self.orderbook_analyzer
            logger.debug(f"   ✅ Servicio: orderbook_analyzer")

        if self.ml_system:
            self.active_services['ml_predictor'] = self.ml_system
            logger.debug(f"   ✅ Servicio: ml_system")

        if self.rl_agent:
            self.active_services['rl_agent'] = self.rl_agent
            logger.debug(f"   ✅ Servicio: rl_agent")

        if self.trade_manager:
            self.active_services['trade_manager'] = self.trade_manager
            logger.debug(f"   ✅ Servicio: trade_manager")

        if self.parameter_optimizer:
            self.active_services['parameter_optimizer'] = self.parameter_optimizer
            logger.debug(f"   ✅ Servicio: parameter_optimizer")

        # Log resumen
        logger.info(f"   📊 Servicios activos: {len(self.active_services)}")
        logger.info(f"      Arsenal: {sum(1 for k in self.active_services if k.startswith('arsenal_'))} módulos")
        logger.info(f"      Core: {sum(1 for k in self.active_services if not k.startswith('arsenal_'))} servicios")

    def analyze_opportunity(
        self,
        symbol: str,
        current_price: float,
        market_data: Dict,
        timeframe: str = '15m'
    ) -> Dict:
        """
        Análisis completo usando TODOS los servicios disponibles

        Args:
            symbol: Par de trading (ej: BTCUSDT)
            current_price: Precio actual
            market_data: Datos de mercado disponibles
            timeframe: Timeframe de análisis

        Returns:
            Dict con análisis completo y decisión consolidada
        """
        self.analysis_count += 1

        # 🧠 LOG DE ANÁLISIS CEREBRAL
        logger.info(f"🧠 DECISIÓN CEREBRAL #{self.analysis_count} para {symbol} @ ${current_price:,.2f}")
        logger.info(f"   📊 Servicios activos: {len(self.active_services)}")

        analysis = {
            'symbol': symbol,
            'price': current_price,
            'timeframe': timeframe,
            'timestamp': datetime.now().isoformat(),
            'services_data': {},
            'ml_prediction': None,
            'rl_decision': None,
            'trade_management': None,
            'param_suggestions': None,
            'final_decision': None
        }

        # ========================================
        # 1. RECOPILAR DATOS DE TODOS LOS SERVICIOS
        # ========================================
        services_data = self._collect_all_services_data(symbol, current_price, market_data)
        analysis['services_data'] = services_data

        # ========================================
        # 2. ML PREDICTOR - Analiza patrones
        # ========================================
        if self.ml_system:
            try:
                ml_features = self._extract_ml_features(services_data, market_data)
                ml_prediction = self._get_ml_prediction(symbol, ml_features, market_data)
                analysis['ml_prediction'] = ml_prediction
            except Exception as e:
                logger.debug(f"ML prediction error: {e}")
                analysis['ml_prediction'] = {'prediction': 'HOLD', 'confidence': 0.5}

        # ========================================
        # 3. RL AGENT - Decide acción autónoma
        # ========================================
        if self.rl_agent:
            try:
                rl_state = self._build_rl_state(services_data, analysis['ml_prediction'], market_data)
                rl_decision = self.rl_agent.get_autonomous_decision(rl_state)
                analysis['rl_decision'] = rl_decision
            except Exception as e:
                logger.debug(f"RL decision error: {e}")
                analysis['rl_decision'] = {
                    'action': 'SKIP',
                    'leverage': 3,
                    'tp_percentages': [0.5, 1.0, 1.5],
                    'position_size_pct': 5.0,
                    'confidence': 0.5
                }

        # ========================================
        # 4. TRADE MANAGER - Evalúa posiciones
        # ========================================
        if self.trade_manager:
            try:
                tm_analysis = self._get_trade_management_analysis(symbol, services_data)
                analysis['trade_management'] = tm_analysis
            except Exception as e:
                logger.debug(f"Trade management error: {e}")

        # ========================================
        # 5. PARAMETER OPTIMIZER - Sugiere ajustes
        # ========================================
        if self.parameter_optimizer:
            try:
                current_metrics = self._calculate_current_metrics()
                param_suggestions = self.parameter_optimizer.analyze_performance(current_metrics)
                analysis['param_suggestions'] = param_suggestions
            except Exception as e:
                logger.debug(f"Parameter optimizer error: {e}")

        # ========================================
        # 6. CONSOLIDAR DECISIÓN FINAL
        # ========================================
        analysis['final_decision'] = self._consolidate_decision(
            ml_prediction=analysis.get('ml_prediction'),
            rl_decision=analysis.get('rl_decision'),
            param_suggestions=analysis.get('param_suggestions'),
            trade_management=analysis.get('trade_management'),
            market_data=market_data
        )

        # Guardar en historial
        self._save_to_history(analysis)

        # Log detallado
        self._log_analysis(analysis)

        return analysis

    def _collect_all_services_data(
        self,
        symbol: str,
        current_price: float,
        market_data: Dict
    ) -> Dict:
        """Recopila datos de todos los servicios disponibles"""
        data = {}

        # 1. Régimen de mercado
        if self.regime_detector:
            try:
                regime = market_data.get('regime', 'SIDEWAYS')
                regime_confidence = market_data.get('regime_confidence', 0.5)
                data['regime'] = {
                    'regime': regime,
                    'confidence': regime_confidence,
                    'volatility': market_data.get('volatility', 'medium')
                }
            except Exception:
                pass

        # 2. Orderbook analysis
        if self.orderbook_analyzer:
            try:
                data['orderbook'] = {
                    'pressure': market_data.get('market_pressure', 'NEUTRAL'),
                    'bid_ask_ratio': market_data.get('bid_ask_ratio', 1.0),
                    'depth_imbalance': market_data.get('depth_imbalance', 0)
                }
            except Exception:
                pass

        # 3. Sentiment analysis
        if self.sentiment_analyzer:
            try:
                data['sentiment'] = {
                    'overall': market_data.get('overall_sentiment', 'neutral'),
                    'strength': market_data.get('sentiment_strength', 0),
                    'fear_greed': market_data.get('fear_greed_index', 50)
                }
            except Exception:
                pass

        # 4. Arsenal modules (Feature Aggregator)
        if self.feature_aggregator:
            try:
                # Liquidation heatmap
                if hasattr(self.feature_aggregator, 'get_liquidation_levels'):
                    liq_data = self.feature_aggregator.get_liquidation_levels(symbol)
                    if liq_data:
                        data['liquidation_heatmap'] = liq_data

                # Funding rate
                if hasattr(self.feature_aggregator, 'get_funding_analysis'):
                    funding_data = self.feature_aggregator.get_funding_analysis(symbol)
                    if funding_data:
                        data['funding_rate'] = funding_data

                # Volume profile
                if hasattr(self.feature_aggregator, 'get_volume_profile'):
                    vp_data = self.feature_aggregator.get_volume_profile(symbol)
                    if vp_data:
                        data['volume_profile'] = vp_data

                # Pattern recognition
                if hasattr(self.feature_aggregator, 'detect_patterns'):
                    patterns = self.feature_aggregator.detect_patterns(symbol)
                    if patterns:
                        data['patterns'] = patterns

                # Order flow
                if hasattr(self.feature_aggregator, 'analyze_order_flow'):
                    order_flow = self.feature_aggregator.analyze_order_flow(symbol)
                    if order_flow:
                        data['order_flow'] = order_flow

                # Session trading
                if hasattr(self.feature_aggregator, 'get_session_analysis'):
                    session = self.feature_aggregator.get_session_analysis()
                    if session:
                        data['session'] = session

                # Correlation matrix
                if hasattr(self.feature_aggregator, 'get_correlations'):
                    correlations = self.feature_aggregator.get_correlations(symbol)
                    if correlations:
                        data['correlations'] = correlations

            except Exception as e:
                logger.debug(f"Arsenal data collection error: {e}")

        # 5. Datos de mercado adicionales
        data['market'] = {
            'rsi': market_data.get('rsi', 50),
            'macd_signal': market_data.get('macd_signal', 'neutral'),
            'volume_ratio': market_data.get('volume_ratio', 1.0),
            'trend': market_data.get('trend', 'sideways'),
            'volatility_pct': market_data.get('volatility_pct', 2.0)
        }

        return data

    def _extract_ml_features(self, services_data: Dict, market_data: Dict) -> Dict:
        """Extrae features para ML desde todos los servicios"""
        features = {}

        # Features de régimen
        if 'regime' in services_data:
            features['regime'] = services_data['regime'].get('regime', 'SIDEWAYS')
            features['regime_confidence'] = services_data['regime'].get('confidence', 0.5)

        # Features de orderbook
        if 'orderbook' in services_data:
            features['orderbook_pressure'] = services_data['orderbook'].get('pressure', 'NEUTRAL')
            features['bid_ask_ratio'] = services_data['orderbook'].get('bid_ask_ratio', 1.0)

        # Features de sentiment
        if 'sentiment' in services_data:
            features['sentiment'] = services_data['sentiment'].get('overall', 'neutral')
            features['fear_greed'] = services_data['sentiment'].get('fear_greed', 50)

        # Features de mercado
        if 'market' in services_data:
            features.update(services_data['market'])

        # Features de liquidation
        if 'liquidation_heatmap' in services_data:
            features['near_liquidation'] = services_data['liquidation_heatmap'].get('near_cluster', False)

        # Features de funding
        if 'funding_rate' in services_data:
            features['funding_rate'] = services_data['funding_rate'].get('rate', 0)

        # Features de patterns
        if 'patterns' in services_data:
            features['pattern_detected'] = services_data['patterns'].get('detected', False)
            features['pattern_type'] = services_data['patterns'].get('type', 'none')

        # Features de order flow
        if 'order_flow' in services_data:
            features['order_flow_ratio'] = services_data['order_flow'].get('ratio', 1.0)

        return features

    def _get_ml_prediction(self, symbol: str, ml_features: Dict, market_data: Dict) -> Dict:
        """Obtiene predicción del ML system"""
        if not self.ml_system:
            return {'prediction': 'HOLD', 'confidence': 0.5}

        try:
            # Usar proceso existente si está disponible
            if hasattr(self.ml_system, 'get_prediction'):
                pred = self.ml_system.get_prediction(symbol, ml_features)
                return pred
            else:
                # Fallback básico
                return {
                    'prediction': market_data.get('ml_prediction', 'HOLD'),
                    'confidence': market_data.get('ml_confidence', 0.5)
                }
        except Exception as e:
            logger.debug(f"ML prediction error: {e}")
            return {'prediction': 'HOLD', 'confidence': 0.5}

    def _build_rl_state(
        self,
        services_data: Dict,
        ml_prediction: Dict,
        market_data: Dict
    ) -> Dict:
        """Construye estado para RL Agent con todos los datos"""
        state = {}

        # Copiar datos de mercado base
        state.update(market_data)

        # Agregar datos de servicios
        if 'regime' in services_data:
            state['regime'] = services_data['regime'].get('regime', 'SIDEWAYS')
            state['regime_confidence'] = services_data['regime'].get('confidence', 0.5)

        if 'orderbook' in services_data:
            state['market_pressure'] = services_data['orderbook'].get('pressure', 'NEUTRAL')

        if 'sentiment' in services_data:
            state['overall_sentiment'] = services_data['sentiment'].get('overall', 'neutral')
            state['sentiment_strength'] = services_data['sentiment'].get('strength', 0)
            state['fear_greed_index'] = services_data['sentiment'].get('fear_greed', 50)

        if 'patterns' in services_data:
            state['pattern_detected'] = services_data['patterns'].get('detected', False)
            state['pattern_confidence'] = services_data['patterns'].get('confidence', 0)

        if 'order_flow' in services_data:
            state['order_flow_ratio'] = services_data['order_flow'].get('ratio', 1.0)

        # Agregar predicción ML
        if ml_prediction:
            state['ml_prediction'] = ml_prediction.get('prediction', 'HOLD')
            state['ml_confidence'] = ml_prediction.get('confidence', 0.5)

        return state

    def _get_trade_management_analysis(self, symbol: str, services_data: Dict) -> Optional[Dict]:
        """Obtiene análisis de gestión de trade si hay posición abierta"""
        if not self.trade_manager:
            return None

        try:
            # Verificar si hay posición abierta
            if hasattr(self.trade_manager, 'has_position'):
                if not self.trade_manager.has_position(symbol):
                    return None

            # Analizar con todos los servicios
            if hasattr(self.trade_manager, 'analyze_position_with_all_services'):
                return self.trade_manager.analyze_position_with_all_services(symbol, services_data)

        except Exception as e:
            logger.debug(f"Trade management analysis error: {e}")

        return None

    def _calculate_current_metrics(self) -> Dict:
        """Calcula métricas actuales de rendimiento"""
        metrics = {
            'total_trades': self.performance_metrics.get('trades_executed', 0),
            'win_rate': 0,
            'avg_profit': 0
        }

        if metrics['total_trades'] > 0:
            profitable = self.performance_metrics.get('profitable_trades', 0)
            metrics['win_rate'] = (profitable / metrics['total_trades']) * 100
            metrics['avg_profit'] = self.performance_metrics.get('total_pnl', 0) / metrics['total_trades']

        return metrics

    def _consolidate_decision(
        self,
        ml_prediction: Optional[Dict],
        rl_decision: Optional[Dict],
        param_suggestions: Optional[Dict],
        trade_management: Optional[Dict],
        market_data: Dict
    ) -> Dict:
        """
        Combina todas las decisiones en una final consolidada
        """
        decision = {
            'action': 'SKIP',
            'leverage': 3,
            'tp_percentages': [0.5, 1.0, 1.5],
            'position_size_pct': 5.0,
            'ml_confidence': 0.5,
            'rl_confidence': 0.5,
            'consolidated_confidence': 0.5,
            'reasons': []
        }

        # 1. Si Trade Manager dice cerrar, prioridad máxima
        if trade_management and trade_management.get('action') == 'CLOSE':
            decision['action'] = 'CLOSE_POSITION'
            decision['reasons'].append(f"Trade Manager: {trade_management.get('reason', 'close signal')}")
            decision['consolidated_confidence'] = trade_management.get('confidence', 0.8)
            return decision

        # 2. Usar decisión del RL Agent como base
        if rl_decision:
            decision['action'] = rl_decision.get('action', 'SKIP')
            decision['leverage'] = rl_decision.get('leverage', 3)
            decision['tp_percentages'] = rl_decision.get('tp_percentages', [0.5, 1.0, 1.5])
            decision['position_size_pct'] = rl_decision.get('position_size_pct', 5.0)
            decision['rl_confidence'] = rl_decision.get('confidence', 0.5)

        # 3. Ajustar con predicción ML
        if ml_prediction:
            decision['ml_confidence'] = ml_prediction.get('confidence', 0.5)

            # Si RL dice SKIP pero ML tiene muy alta confianza, reconsiderar
            if decision['action'] == 'SKIP' and decision['ml_confidence'] > 0.85:
                ml_pred = ml_prediction.get('prediction', 'HOLD')
                if ml_pred in ['BUY', 'SELL']:
                    decision['reasons'].append(f"ML override: {ml_pred} con {decision['ml_confidence']:.1%} confianza")
                    # Podría forzar trade pero dejamos que RL aprenda
                    logger.info(f"🤔 RL dice SKIP pero ML muy confiado ({decision['ml_confidence']:.1%})")

        # 4. Aplicar sugerencias de Parameter Optimizer
        if param_suggestions and param_suggestions.get('immediate_changes'):
            changes = param_suggestions['immediate_changes']
            decision['reasons'].append(f"Param Optimizer ajustes: {list(changes.keys())}")

        # 5. Calcular confianza consolidada usando PESOS DINÁMICOS
        # 🎯 SCALPING: Los pesos se ajustan según qué IA está teniendo mejor rendimiento
        ml_weight = self.ia_performance['ml_system']['weight']
        rl_weight = self.ia_performance['rl_agent']['weight']

        # Normalizar pesos para que sumen 1.0
        total_weight = ml_weight + rl_weight
        ml_weight_normalized = ml_weight / total_weight
        rl_weight_normalized = rl_weight / total_weight

        decision['consolidated_confidence'] = (
            decision['ml_confidence'] * ml_weight_normalized +
            decision['rl_confidence'] * rl_weight_normalized
        )

        # 6. Si ML y RL no están de acuerdo, dar más peso al que tiene mejor historial
        if rl_decision and ml_prediction:
            ml_pred = ml_prediction.get('prediction', 'HOLD')
            rl_action = rl_decision.get('action', 'SKIP')

            # Si hay conflicto (ML sugiere algo, RL dice SKIP o viceversa)
            ml_says_trade = ml_pred in ['BUY', 'SELL'] and decision['ml_confidence'] > 0.65
            rl_says_trade = rl_action != 'SKIP'

            if ml_says_trade != rl_says_trade:
                # Quien tiene mejor accuracy histórica gana
                ml_accuracy = self.ia_performance['ml_system']['accuracy']
                rl_accuracy = self.ia_performance['rl_agent']['accuracy']

                logger.info(
                    f"⚔️ Conflicto ML vs RL: ML({ml_pred}, {ml_accuracy:.1%}) vs RL({rl_action}, {rl_accuracy:.1%})"
                )

                if ml_accuracy > rl_accuracy + 0.05:  # ML es mejor por >5%
                    if ml_says_trade and not rl_says_trade:
                        # ML override: podríamos considerar tradear
                        decision['reasons'].append(f"ML tiene mejor historial ({ml_accuracy:.1%} vs {rl_accuracy:.1%})")
                        logger.info(f"🤖 ML tiene mejor historial, considerando su sugerencia")
                elif rl_accuracy > ml_accuracy + 0.05:  # RL es mejor por >5%
                    if rl_says_trade and not ml_says_trade:
                        decision['reasons'].append(f"RL tiene mejor historial ({rl_accuracy:.1%} vs {ml_accuracy:.1%})")
                        logger.info(f"🤖 RL tiene mejor historial, siguiendo su decisión")

        # 7. Agregar side desde market_data
        decision['side'] = market_data.get('side', 'NEUTRAL')

        # Log de pesos usados
        logger.debug(f"📊 Pesos usados: ML={ml_weight_normalized:.2f}, RL={rl_weight_normalized:.2f}")

        return decision

    def _save_to_history(self, analysis: Dict):
        """Guarda análisis en historial"""
        self.decision_history.append({
            'timestamp': analysis['timestamp'],
            'symbol': analysis['symbol'],
            'decision': analysis['final_decision']['action'],
            'confidence': analysis['final_decision']['consolidated_confidence']
        })

        if len(self.decision_history) > self.max_history:
            self.decision_history = self.decision_history[-self.max_history:]

    def _log_analysis(self, analysis: Dict):
        """Log detallado del análisis"""
        decision = analysis['final_decision']
        services_count = len(analysis['services_data'])

        logger.info(f"""
🧠 DECISIÓN CEREBRAL para {analysis['symbol']}:
   Servicios usados: {services_count}
   Action: {decision['action']}
   Side: {decision.get('side', 'N/A')}
   Leverage: {decision['leverage']}x
   TPs: {[f'{tp:.2f}%' for tp in decision['tp_percentages']]}
   Size: {decision['position_size_pct']:.1f}%
   ML Conf: {decision['ml_confidence']:.1%}
   RL Conf: {decision['rl_confidence']:.1%}
   Final Conf: {decision['consolidated_confidence']:.1%}
""")

    def learn_from_trade(self, trade_result: Dict):
        """
        TODOS los sistemas aprenden del resultado del trade
        """
        symbol = trade_result.get('symbol', 'UNKNOWN')
        pnl_pct = trade_result.get('pnl_pct', 0)

        logger.info(f"🧠 Cerebro aprendiendo del trade {symbol}: {pnl_pct:+.2f}%")

        # Actualizar métricas
        self.performance_metrics['trades_executed'] += 1
        self.performance_metrics['total_pnl'] += pnl_pct
        if pnl_pct > 0:
            self.performance_metrics['profitable_trades'] += 1

        # 1. RL Agent aprende
        if self.rl_agent:
            try:
                reward = self._calculate_reward(trade_result)
                # El aprendizaje se hace vía learn_from_trade del RL Agent
                logger.debug(f"RL Agent: reward = {reward:.3f}")
            except Exception as e:
                logger.debug(f"RL learning error: {e}")

        # 2. ML System aprende (si tiene la capacidad)
        if self.ml_system and hasattr(self.ml_system, 'add_trade_result'):
            try:
                self.ml_system.add_trade_result(trade_result)
            except Exception as e:
                logger.debug(f"ML learning error: {e}")

        # 3. Trade Manager aprende
        if self.trade_manager and hasattr(self.trade_manager, 'learning'):
            try:
                self.trade_manager.learning.record_trade_outcome(trade_result)
            except Exception as e:
                logger.debug(f"Trade Manager learning error: {e}")

        # 4. Parameter Optimizer registra resultado
        if self.parameter_optimizer and hasattr(self.parameter_optimizer, 'record_trial_result'):
            try:
                self.parameter_optimizer.record_trial_result(
                    config=trade_result.get('config_used', {}),
                    performance={'pnl_pct': pnl_pct, 'duration': trade_result.get('duration', 0)}
                )
            except Exception as e:
                logger.debug(f"Parameter Optimizer learning error: {e}")

        # 5. Guardar experiencia completa
        self._save_trade_experience(trade_result)

        # 6. 🎯 SCALPING: Actualizar rendimiento de cada IA y recalcular pesos
        self._update_ia_performance(trade_result)

        self.trades_analyzed += 1

    def _update_ia_performance(self, trade_result: Dict):
        """
        🎯 SCALPING: Actualiza el rendimiento de cada IA basado en el resultado del trade

        Esto permite que los pesos se ajusten dinámicamente según qué IA
        está teniendo mejor rendimiento histórico.
        """
        symbol = trade_result.get('symbol', 'UNKNOWN')
        pnl_pct = trade_result.get('pnl_pct', 0)
        was_profitable = pnl_pct > 0

        logger.info(f"📊 Actualizando rendimiento de IAs para {symbol} (P&L: {pnl_pct:+.2f}%)")

        # Buscar la predicción original que hizo cada sistema para este trade
        # Por ahora usamos el resultado directo para ajustar accuracy

        # 1. Actualizar RL Agent stats
        rl_perf = self.ia_performance['rl_agent']
        rl_perf['total_predictions'] += 1
        if was_profitable:
            rl_perf['correct_predictions'] += 1

        # Agregar a lista reciente
        rl_perf['recent_predictions'].append(1 if was_profitable else 0)
        if len(rl_perf['recent_predictions']) > self.max_recent_items:
            rl_perf['recent_predictions'] = rl_perf['recent_predictions'][-self.max_recent_items:]

        # Calcular accuracy reciente (últimos 50 trades)
        if rl_perf['recent_predictions']:
            rl_perf['accuracy'] = sum(rl_perf['recent_predictions']) / len(rl_perf['recent_predictions'])

        # 2. Actualizar ML System stats (solo si participó en la decisión)
        ml_perf = self.ia_performance['ml_system']
        ml_perf['total_predictions'] += 1
        if was_profitable:
            ml_perf['correct_predictions'] += 1

        ml_perf['recent_predictions'].append(1 if was_profitable else 0)
        if len(ml_perf['recent_predictions']) > self.max_recent_items:
            ml_perf['recent_predictions'] = ml_perf['recent_predictions'][-self.max_recent_items:]

        if ml_perf['recent_predictions']:
            ml_perf['accuracy'] = sum(ml_perf['recent_predictions']) / len(ml_perf['recent_predictions'])

        # 3. 🎯 RECALCULAR PESOS basado en accuracy reciente
        # El sistema con mejor accuracy obtiene más peso
        rl_accuracy = rl_perf['accuracy']
        ml_accuracy = ml_perf['accuracy']

        # Pesos base: 60% RL, 40% ML
        # Ajuste: +/- 10% según diferencia de accuracy
        base_rl_weight = 0.60
        base_ml_weight = 0.40

        accuracy_diff = rl_accuracy - ml_accuracy

        # Ajustar pesos según quien está teniendo mejor rendimiento
        if accuracy_diff > 0.10:  # RL es 10%+ mejor
            rl_perf['weight'] = min(0.80, base_rl_weight + 0.15)
            ml_perf['weight'] = max(0.20, base_ml_weight - 0.15)
        elif accuracy_diff > 0.05:  # RL es 5%+ mejor
            rl_perf['weight'] = min(0.75, base_rl_weight + 0.10)
            ml_perf['weight'] = max(0.25, base_ml_weight - 0.10)
        elif accuracy_diff < -0.10:  # ML es 10%+ mejor
            rl_perf['weight'] = max(0.35, base_rl_weight - 0.15)
            ml_perf['weight'] = min(0.65, base_ml_weight + 0.15)
        elif accuracy_diff < -0.05:  # ML es 5%+ mejor
            rl_perf['weight'] = max(0.40, base_rl_weight - 0.10)
            ml_perf['weight'] = min(0.60, base_ml_weight + 0.10)
        else:  # Similar accuracy - pesos base
            rl_perf['weight'] = base_rl_weight
            ml_perf['weight'] = base_ml_weight

        logger.info(
            f"   📊 IA Performance Update:\n"
            f"      RL Agent: {rl_accuracy:.1%} accuracy → peso {rl_perf['weight']:.2f}\n"
            f"      ML System: {ml_accuracy:.1%} accuracy → peso {ml_perf['weight']:.2f}"
        )

    def get_ia_performance_stats(self) -> Dict:
        """
        🎯 Retorna estadísticas de rendimiento de cada IA para /stats

        Returns:
            Dict con métricas de rendimiento de ML y RL
        """
        return {
            'ml_system': {
                'accuracy': self.ia_performance['ml_system']['accuracy'],
                'weight': self.ia_performance['ml_system']['weight'],
                'total_predictions': self.ia_performance['ml_system']['total_predictions'],
                'correct_predictions': self.ia_performance['ml_system']['correct_predictions']
            },
            'rl_agent': {
                'accuracy': self.ia_performance['rl_agent']['accuracy'],
                'weight': self.ia_performance['rl_agent']['weight'],
                'total_predictions': self.ia_performance['rl_agent']['total_predictions'],
                'correct_predictions': self.ia_performance['rl_agent']['correct_predictions']
            },
            'best_performer': 'RL Agent' if self.ia_performance['rl_agent']['accuracy'] > self.ia_performance['ml_system']['accuracy'] else 'ML System'
        }

    def _calculate_reward(self, trade_result: Dict) -> float:
        """Calcula reward para RL desde resultado del trade"""
        pnl_pct = trade_result.get('pnl_pct', 0)
        leverage = trade_result.get('leverage', 1)
        return pnl_pct * leverage

    def _save_trade_experience(self, trade_result: Dict):
        """Guarda experiencia del trade para análisis futuro"""
        experience = {
            'timestamp': datetime.now().isoformat(),
            'symbol': trade_result.get('symbol'),
            'side': trade_result.get('side'),
            'pnl_pct': trade_result.get('pnl_pct'),
            'leverage': trade_result.get('leverage'),
            'duration': trade_result.get('duration'),
            'services_used': trade_result.get('services_used', [])
        }

        self.trade_experiences.append(experience)

        # Limitar tamaño
        if len(self.trade_experiences) > 1000:
            self.trade_experiences = self.trade_experiences[-1000:]

    def export_state(self) -> Dict:
        """Exporta estado completo del cerebro"""
        return {
            'version': '1.0',
            'timestamp': datetime.now().isoformat(),
            'analysis_count': self.analysis_count,
            'trades_analyzed': self.trades_analyzed,
            'performance_metrics': self.performance_metrics,
            'decision_history': self.decision_history[-100:],  # Últimas 100
            'trade_experiences': self.trade_experiences[-200:],  # Últimas 200
            'active_services': list(self.active_services.keys())
        }

    def get_state(self) -> Dict:
        """Alias para export_state() - usado por persistencia"""
        return self.export_state()

    def import_state(self, state: Dict):
        """Importa estado del cerebro"""
        if state.get('version') != '1.0':
            logger.warning("Version mismatch en estado del cerebro")

        self.analysis_count = state.get('analysis_count', 0)
        self.trades_analyzed = state.get('trades_analyzed', 0)
        self.performance_metrics = state.get('performance_metrics', self.performance_metrics)
        self.decision_history = state.get('decision_history', [])
        self.trade_experiences = state.get('trade_experiences', [])

        logger.info(f"🧠 Estado del cerebro importado: {self.trades_analyzed} trades analizados")
