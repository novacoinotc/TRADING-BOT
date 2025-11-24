"""
Trade Manager - Gestión INTELIGENTE de trades abiertos en tiempo real
Usa los 24 servicios del sistema para tomar decisiones basadas en datos, no porcentajes fijos
"""
import logging
import asyncio
from typing import Dict, Optional, List, Tuple
from datetime import datetime

from .trade_management_learning import TradeManagementLearning

logger = logging.getLogger(__name__)


class TradeManager:
    """
    Gestor INTELIGENTE de trades abiertos.

    NUEVA ARQUITECTURA: Decisiones basadas en 24 servicios del sistema:
    - RL Agent (Q-values para evaluar estado actual)
    - ML System (predicciones de próximos movimientos)
    - Sentiment Analyzer (Fear & Greed, News, Social)
    - Market Regime Detector (Bear/Bull/Sideways)
    - Feature Aggregator - Arsenal Avanzado:
      * Liquidation Zones (evitar stop hunts)
      * Funding Rate (sentiment extremo)
      * Volume Profile (soporte/resistencia)
      * Pattern Recognition (reversiones)
      * Order Flow (momentum institucional)
      * Session Trading (volatilidad por sesión)

    Capacidades:
    - Modificar SL/TP dinámicamente según condiciones de mercado
    - Cerrar posiciones anticipadamente si detecta reversión
    - Trailing stop loss inteligente
    - Partial take profit basado en confianza
    - Breakeven protection adaptativo según régimen
    """

    def __init__(
        self,
        position_monitor,
        futures_trader,
        rl_agent,
        ml_system,
        market_analyzer
    ):
        """
        Args:
            position_monitor: Monitor de posiciones
            futures_trader: Trader de Binance Futures
            rl_agent: Agente RL para decisiones
            ml_system: Sistema ML para análisis
            market_analyzer: Analizador de mercado (contiene todos los servicios)
        """
        self.position_monitor = position_monitor
        self.futures_trader = futures_trader
        self.rl_agent = rl_agent
        self.ml_system = ml_system
        self.market_analyzer = market_analyzer

        # Referencias a servicios específicos (serán asignados dinámicamente)
        self.sentiment_analyzer = None
        self.regime_detector = None
        self.feature_aggregator = None
        self.orderbook_analyzer = None

        # Intentar obtener servicios del market_analyzer
        if market_analyzer:
            self.sentiment_analyzer = getattr(market_analyzer, 'sentiment_system', None)
            self.regime_detector = getattr(market_analyzer, 'regime_detector', None)
            self.feature_aggregator = getattr(market_analyzer, 'feature_aggregator', None)
            self.orderbook_analyzer = getattr(market_analyzer, 'orderbook_analyzer', None)

        self._running = False
        self._check_interval = 30  # Revisar cada 30 segundos
        self._check_counter = 0  # Contador para heartbeat

        # Configuración DINÁMICA (se ajusta según condiciones de mercado)
        self.base_config = {
            'min_confidence_for_action': 0.65,  # Confianza mínima para ejecutar acción
            'high_confidence_threshold': 0.80,  # Alta confianza
            'reversal_confidence_threshold': 0.75,  # Para detectar reversiones
            'min_pnl_for_breakeven': 0.5,  # Mínimo 0.5% ganancia para considerar breakeven
            'min_pnl_for_partial': 2.0,  # Mínimo 2% para partial TP
            'max_drawdown_tolerance': 3.0,  # Máximo 3% de caída desde máximo (aumentado de 2%)
        }

        # Tracking de máximos/mínimos por posición
        self._position_highs = {}  # {symbol: highest_pnl_pct}
        self._position_lows = {}  # {symbol: lowest_pnl_pct}
        self._partial_closed = set()  # Símbolos donde ya se hizo partial TP
        self._breakeven_set = set()  # Símbolos donde ya se movió a breakeven

        # 🧠 Sistema de Aprendizaje
        self.learning = TradeManagementLearning(max_history=200)
        self.learning.load_from_file()  # Cargar historial anterior si existe

        # Tracking de posiciones abiertas (para detectar cierres)
        self._tracked_positions = set()  # {symbol}

        logger.info("✅ Trade Manager INTELIGENTE inicializado")
        logger.info("   📊 Servicios integrados:")
        logger.info(f"      - RL Agent: {'✅' if rl_agent else '❌'}")
        logger.info(f"      - ML System: {'✅' if ml_system else '❌'}")
        logger.info(f"      - Sentiment: {'✅' if self.sentiment_analyzer else '❌'}")
        logger.info(f"      - Regime Detector: {'✅' if self.regime_detector else '❌'}")
        logger.info(f"      - OrderBook: {'✅' if self.orderbook_analyzer else '❌'}")
        logger.info(f"      - Learning System: ✅ ({len(self.learning.actions_history)} acciones en historial)")

        # Log de módulos del Arsenal (Feature Aggregator)
        if self.feature_aggregator:
            logger.info("   🔧 Arsenal Avanzado:")
            logger.info(f"      - Liquidation Heatmap: {'✅' if hasattr(self.feature_aggregator, 'liquidation_heatmap') else '❌'}")
            logger.info(f"      - Funding Rate Analyzer: {'✅' if hasattr(self.feature_aggregator, 'funding_rate_analyzer') else '❌'}")
            logger.info(f"      - Volume Profile: {'✅' if hasattr(self.feature_aggregator, 'volume_profile') else '❌'}")
            logger.info(f"      - Pattern Recognition: {'✅' if hasattr(self.feature_aggregator, 'pattern_recognition') else '❌'}")
            logger.info(f"      - Order Flow: {'✅' if hasattr(self.feature_aggregator, 'order_flow') else '❌'}")
            logger.info(f"      - Session Trading: {'✅' if hasattr(self.feature_aggregator, 'session_trading') else '❌'}")
            logger.info(f"      - Correlation Matrix: {'✅' if hasattr(self.feature_aggregator, 'correlation_matrix') else '❌'}")
        else:
            logger.warning("   ⚠️ Feature Aggregator NO disponible - Arsenal desactivado")

    async def start_monitoring(self):
        """Inicia monitoreo activo de trades"""
        if self._running:
            logger.warning("⚠️ Trade Manager ya está corriendo")
            return

        self._running = True
        logger.info("🟢 Trade Manager INTELIGENTE: Iniciando monitoreo activo...")

        while self._running:
            try:
                await self._check_all_positions()
                await asyncio.sleep(self._check_interval)

            except asyncio.CancelledError:
                logger.info("⏸️ Trade Manager cancelled")
                break
            except Exception as e:
                logger.error(f"❌ Error en Trade Manager loop: {e}", exc_info=True)
                await asyncio.sleep(self._check_interval)

        self._running = False
        logger.info("🔴 Trade Manager detenido")

    def stop_monitoring(self):
        """Detiene monitoreo"""
        logger.info("🛑 Deteniendo Trade Manager...")
        self._running = False

    async def _detect_closed_positions(self):
        """Detecta posiciones que cerraron desde el último check"""
        try:
            # Obtener posiciones actuales
            current_positions = self.position_monitor.get_open_positions()
            current_symbols = set(current_positions.keys()) if current_positions else set()

            # Detectar símbolos que cerraron
            closed_symbols = self._tracked_positions - current_symbols

            if closed_symbols:
                logger.info(f"🔍 Detectados {len(closed_symbols)} trade(s) cerrado(s): {closed_symbols}")

            for symbol in closed_symbols:
                await self._evaluate_closed_position(symbol)

        except Exception as e:
            logger.error(f"Error detecting closed positions: {e}", exc_info=True)

    async def _evaluate_closed_position(self, symbol: str):
        """Evalúa las decisiones de un trade que cerró"""
        try:
            # Obtener información del cierre desde position monitor history
            # Si position_monitor tiene historial de trades cerrados
            closed_trades = getattr(self.position_monitor, 'closed_trades', {})

            if symbol in closed_trades:
                trade_info = closed_trades[symbol]
                final_pnl_pct = trade_info.get('pnl_pct', 0)
                close_reason = trade_info.get('reason', 'UNKNOWN')
            else:
                # Fallback: usar último P&L conocido
                final_pnl_pct = 0  # No podemos saber el final exacto
                close_reason = 'UNKNOWN'
                logger.warning(f"⚠️ No se encontró info de cierre para {symbol}, usando fallback")

            # Obtener el máximo P&L alcanzado
            highest_pnl = self._position_highs.get(symbol, final_pnl_pct)

            # Evaluar acciones
            logger.info(f"📊 Evaluando acciones de {symbol} (Final: {final_pnl_pct:+.2f}%, Max: {highest_pnl:+.2f}%)")

            self.learning.evaluate_actions(
                symbol=symbol,
                final_pnl_pct=final_pnl_pct,
                highest_pnl_pct_reached=highest_pnl,
                close_reason=close_reason
            )

            # Limpiar tracking
            self._position_highs.pop(symbol, None)
            self._position_lows.pop(symbol, None)
            self._partial_closed.discard(symbol)
            self._breakeven_set.discard(symbol)
            self._tracked_positions.discard(symbol)

            # Auto-guardar cada 10 evaluaciones
            if self.learning.stats['total_evaluated'] % 10 == 0:
                self.learning.save_to_file()
                logger.info(f"💾 Learning guardado ({self.learning.stats['total_evaluated']} evaluaciones)")

        except Exception as e:
            logger.error(f"Error evaluating closed position {symbol}: {e}", exc_info=True)

    async def _check_all_positions(self):
        """Revisa todas las posiciones abiertas y aplica gestión inteligente"""
        self._check_counter += 1

        # 🔍 PRIMERO: Detectar trades que cerraron
        await self._detect_closed_positions()

        # LUEGO: Gestionar trades abiertos
        positions = self.position_monitor.get_open_positions()

        if not positions:
            # Heartbeat cada 10 ciclos (~5 minutos) para mostrar que está activo
            if self._check_counter % 10 == 0:
                logger.info(f"💓 Trade Manager activo (ciclo #{self._check_counter}) - Sin posiciones abiertas")
            return

        logger.info(f"🔍 Trade Manager: Revisando {len(positions)} posición(es) abierta(s)")

        for symbol, position in positions.items():
            try:
                await self._manage_position(symbol, position)
            except Exception as e:
                logger.error(f"❌ Error gestionando {symbol}: {e}", exc_info=True)

    async def _manage_position(self, symbol: str, position: Dict):
        """
        Aplica gestión INTELIGENTE a una posición específica

        NUEVA LÓGICA: Analiza condiciones de mercado ANTES de cada decisión
        No usa porcentajes fijos, sino análisis de los 24 servicios

        Args:
            symbol: Símbolo del trade
            position: Datos de la posición
        """
        # Extraer datos de la posición
        pnl_pct = position.get('unrealized_pnl_pct', 0)
        pnl_usdt = position.get('unrealized_pnl', 0)
        entry_price = position.get('entry_price', 0)
        current_price = position.get('mark_price', 0)
        side = position.get('side', 'UNKNOWN')

        # Actualizar tracking de máximos/mínimos
        if symbol not in self._position_highs:
            self._position_highs[symbol] = pnl_pct
        if symbol not in self._position_lows:
            self._position_lows[symbol] = pnl_pct

        self._position_highs[symbol] = max(self._position_highs[symbol], pnl_pct)
        self._position_lows[symbol] = min(self._position_lows[symbol], pnl_pct)

        # Añadir a tracked positions (para learning system)
        self._tracked_positions.add(symbol)

        highest_pnl = self._position_highs[symbol]
        drawdown_from_high = highest_pnl - pnl_pct

        logger.info(
            f"📊 {symbol}: P&L={pnl_pct:+.2f}% (Max={highest_pnl:+.2f}%), "
            f"Drawdown={drawdown_from_high:.2f}%, Price=${current_price:.4f}, Side={side}"
        )

        # 🧠 ANALIZAR CONDICIONES DE MERCADO (los 24 servicios)
        market_conditions = await self._analyze_market_conditions(symbol, position)

        # 1️⃣ DECISIÓN INTELIGENTE: Breakeven Protection
        if pnl_pct >= self.base_config['min_pnl_for_breakeven'] and symbol not in self._breakeven_set:
            decision = self._get_intelligent_decision('breakeven', position, market_conditions)

            if decision['should_execute']:
                logger.info(
                    f"🛡️ {symbol}: BREAKEVEN DECISION "
                    f"(Confidence: {decision['confidence']:.0%}, Risk: {decision['risk_score']:.2f})"
                )
                logger.info(f"   Razones: {', '.join(decision['reasons'])}")
                await self._set_breakeven(symbol, position)
                self._breakeven_set.add(symbol)

                # 🧠 Registrar acción en learning system
                self.learning.record_action(
                    symbol=symbol,
                    action_type='breakeven',
                    pnl_at_decision=pnl_usdt,
                    pnl_pct_at_decision=pnl_pct,
                    conditions=market_conditions,
                    decision_info=decision
                )

        # 2️⃣ DECISIÓN INTELIGENTE: Trailing Stop
        if highest_pnl > 2.0:  # Solo si ya hay ganancia significativa
            decision = self._get_intelligent_decision('trailing', position, market_conditions)

            if decision['should_execute']:
                logger.info(
                    f"📈 {symbol}: TRAILING STOP DECISION "
                    f"(Confidence: {decision['confidence']:.0%}, Max P&L: {highest_pnl:+.2f}%)"
                )
                logger.info(f"   Razones: {', '.join(decision['reasons'])}")
                await self._apply_trailing_stop(symbol, position, highest_pnl, market_conditions)

                # 🧠 Registrar acción en learning system
                self.learning.record_action(
                    symbol=symbol,
                    action_type='trailing',
                    pnl_at_decision=pnl_usdt,
                    pnl_pct_at_decision=pnl_pct,
                    conditions=market_conditions,
                    decision_info=decision
                )

        # 3️⃣ DECISIÓN INTELIGENTE: Partial Take Profit
        if pnl_pct >= self.base_config['min_pnl_for_partial'] and symbol not in self._partial_closed:
            decision = self._get_intelligent_decision('partial_tp', position, market_conditions)

            if decision['should_execute']:
                logger.info(
                    f"💰 {symbol}: PARTIAL TP DECISION "
                    f"(Confidence: {decision['confidence']:.0%}, P&L: {pnl_pct:+.2f}%)"
                )
                logger.info(f"   Razones: {', '.join(decision['reasons'])}")
                await self._partial_take_profit(symbol, position)

                # 🧠 Registrar acción en learning system
                self.learning.record_action(
                    symbol=symbol,
                    action_type='partial_tp',
                    pnl_at_decision=pnl_usdt,
                    pnl_pct_at_decision=pnl_pct,
                    conditions=market_conditions,
                    decision_info=decision
                )

        # 4️⃣ DECISIÓN INTELIGENTE: Protección contra movimiento adverso
        # ⚠️ FIX CRÍTICO: NUNCA cerrar trades en profit por "adverse move"
        # Solo cerrar si: 1) La posición está en PÉRDIDA (pnl_pct < 0) Y 2) El drawdown es significativo
        if drawdown_from_high >= self.base_config['max_drawdown_tolerance']:
            # ✅ PROTECCIÓN: Si está en profit, NO cerrar por drawdown
            if pnl_pct > 0:
                logger.info(
                    f"🛡️ {symbol}: Drawdown {drawdown_from_high:.2f}% detectado PERO posición en profit "
                    f"(P&L: {pnl_pct:+.2f}%) - NO SE CIERRA"
                )
            else:
                # Solo considerar cierre si está en pérdida real
                decision = self._get_intelligent_decision('close_adverse', position, market_conditions)

                if decision['should_execute']:
                    logger.warning(
                        f"⚠️ {symbol}: CLOSE ADVERSE DECISION "
                        f"(Confidence: {decision['confidence']:.0%}, Drawdown: {drawdown_from_high:.2f}%, P&L: {pnl_pct:+.2f}%)"
                    )
                    logger.warning(f"   Razones: {', '.join(decision['reasons'])}")
                    await self._close_on_adverse_move(symbol, position, drawdown_from_high, pnl_pct)

                # 🧠 Registrar acción en learning system
                self.learning.record_action(
                    symbol=symbol,
                    action_type='close_adverse',
                    pnl_at_decision=pnl_usdt,
                    pnl_pct_at_decision=pnl_pct,
                    conditions=market_conditions,
                    decision_info=decision
                )

        # 5️⃣ DECISIÓN INTELIGENTE: Detección de reversión
        decision = self._get_intelligent_decision('reversal', position, market_conditions)

        if decision['should_execute']:
            logger.warning(
                f"🔄 {symbol}: REVERSAL DETECTED "
                f"(Confidence: {decision['confidence']:.0%}, Risk: {decision['risk_score']:.2f})"
            )
            logger.warning(f"   Razones: {', '.join(decision['reasons'])}")

            # 🧠 Registrar acción en learning system ANTES de cerrar
            self.learning.record_action(
                symbol=symbol,
                action_type='reversal',
                pnl_at_decision=pnl_usdt,
                pnl_pct_at_decision=pnl_pct,
                conditions=market_conditions,
                decision_info=decision
            )

            self.futures_trader.close_position(symbol, reason='AI_REVERSAL')
            # Limpiar tracking
            self._position_highs.pop(symbol, None)
            self._position_lows.pop(symbol, None)
            self._partial_closed.discard(symbol)
            self._breakeven_set.discard(symbol)

    async def _analyze_market_conditions(self, symbol: str, position: Dict) -> Dict:
        """
        Analiza condiciones de mercado usando LOS 24 SERVICIOS

        Esta es la función CLAVE que integra toda la inteligencia del sistema

        Returns:
            Dict con análisis completo:
            {
                'should_secure_profits': bool,
                'should_let_run': bool,
                'reversal_risk': float (0-1),
                'continuation_probability': float (0-1),
                'market_regime': str,
                'sentiment_score': float (-1 a 1),
                'ml_prediction': Dict,
                'rl_q_values': Dict,
                'arsenal_signals': Dict,
                'confidence': float (0-1),
                'reasons': List[str]
            }
        """
        conditions = {
            'should_secure_profits': False,
            'should_let_run': False,
            'reversal_risk': 0.0,
            'continuation_probability': 0.5,
            'market_regime': 'UNKNOWN',
            'sentiment_score': 0.0,
            'ml_prediction': None,
            'rl_q_values': None,
            'arsenal_signals': {},
            'confidence': 0.0,
            'reasons': []
        }

        try:
            side = position.get('side', 'UNKNOWN')
            pnl_pct = position.get('unrealized_pnl_pct', 0)

            # 📊 1. RL AGENT ANALYSIS (Q-values del estado actual)
            # NOTA: RL Agent es para ABRIR trades, no gestionarlos
            # Acciones: SKIP, OPEN_CONSERVATIVE, OPEN_NORMAL, OPEN_AGGRESSIVE, FUTURES_*
            # Si Q(SKIP) es alto, puede indicar que RL no recomendaría entrar ahora
            if self.rl_agent:
                try:
                    # Construir market_data para get_state_representation()
                    pair = symbol.replace('USDT', '/USDT')

                    # Necesitamos construir market_data completo
                    market_data_for_state = {
                        'pair': pair,
                        'action': 'BUY' if side == 'LONG' else 'SELL',
                        'rsi': 50,  # Default - idealmente obtener RSI real
                        'regime': conditions.get('market_regime', 'SIDEWAYS'),
                        'regime_strength': 'MEDIUM',
                        'orderbook': 'NEUTRAL',
                        'confidence': min(100, max(0, pnl_pct * 10)),  # Aproximación
                        'volatility': 'medium',
                        'cryptopanic_sentiment': 'neutral',
                        'news_volume': 0,
                        'total_trades': self.rl_agent.total_trades,
                        'success_rate': self.rl_agent.get_success_rate(),
                    }

                    # Obtener hash del estado usando método real del RL Agent
                    state_hash = self.rl_agent.get_state_representation(market_data_for_state)

                    # Acceder directamente a Q-table
                    if state_hash in self.rl_agent.q_table:
                        q_values = self.rl_agent.q_table[state_hash]
                        conditions['rl_q_values'] = q_values

                        # Analizar Q(SKIP) vs Q(acciones de apertura)
                        skip_q = q_values.get('SKIP', 0)

                        # Calcular promedio de Q-values de acciones de apertura
                        open_actions = [a for a in q_values.keys() if a != 'SKIP']
                        if open_actions:
                            avg_open_q = sum(q_values.get(a, 0) for a in open_actions) / len(open_actions)
                            max_open_q = max(q_values.get(a, 0) for a in open_actions)

                            # Si SKIP tiene Q-value alto, puede indicar mal momento
                            # Interpretación: RL Agent no abriría trade ahora → asegurar ganancias
                            if skip_q > avg_open_q + 0.5:
                                conditions['should_secure_profits'] = True
                                conditions['reasons'].append(
                                    f"RL: SKIP Q={skip_q:.2f} >> Open Q={avg_open_q:.2f} "
                                    f"(RL no abriría trade ahora → asegurar)"
                                )
                            elif max_open_q > skip_q + 0.5:
                                conditions['should_let_run'] = True
                                conditions['reasons'].append(
                                    f"RL: Open Q={max_open_q:.2f} >> SKIP Q={skip_q:.2f} "
                                    f"(RL vería buena oportunidad → dejar correr)"
                                )
                    else:
                        logger.debug(f"Estado {state_hash[:16]}... no encontrado en Q-table")

                except Exception as e:
                    logger.debug(f"Error en RL analysis: {e}")

            # 🧠 2. ML SYSTEM PREDICTION (predicción de próximo movimiento)
            if self.ml_system:
                try:
                    features = self._build_ml_features(position, symbol)
                    prediction = self.ml_system.predict(features)
                    conditions['ml_prediction'] = prediction

                    if prediction:
                        action = prediction.get('action', 'HOLD')
                        confidence = prediction.get('confidence', 0) / 100

                        # Si ML predice movimiento contrario a nuestra posición
                        if (side == 'LONG' and action == 'SELL') or (side == 'SHORT' and action == 'BUY'):
                            conditions['reversal_risk'] += confidence * 0.4  # 40% de peso
                            conditions['reasons'].append(f"ML: Predice {action} (conf={confidence:.0%})")
                        elif (side == 'LONG' and action == 'BUY') or (side == 'SHORT' and action == 'SELL'):
                            conditions['continuation_probability'] += confidence * 0.3
                            conditions['should_let_run'] = True
                            conditions['reasons'].append(f"ML: Confirma dirección (conf={confidence:.0%})")

                except Exception as e:
                    logger.debug(f"Error en ML prediction: {e}")

            # 📰 3. SENTIMENT ANALYSIS (Fear & Greed, News, Social)
            if self.sentiment_analyzer:
                try:
                    pair = symbol.replace('USDT', '/USDT')
                    base_currency = pair.split('/')[0]

                    sentiment_features = self.sentiment_analyzer.get_sentiment_features(pair)
                    if sentiment_features:
                        overall_sentiment = sentiment_features.get('overall_sentiment', 'neutral')
                        fear_greed = sentiment_features.get('fear_greed_index', 50)
                        news_volume = sentiment_features.get('news_volume', 0)

                        # Mapear sentiment a score (-1 a 1)
                        sentiment_map = {'bearish': -0.7, 'slightly_bearish': -0.3, 'neutral': 0,
                                       'slightly_bullish': 0.3, 'bullish': 0.7}
                        conditions['sentiment_score'] = sentiment_map.get(overall_sentiment, 0)

                        # Fear & Greed extremos
                        if fear_greed < 20:  # Extreme Fear
                            if side == 'SHORT':
                                conditions['should_secure_profits'] = True
                                conditions['reasons'].append(f"Sentiment: Extreme Fear ({fear_greed}) - LONG opportunity")
                        elif fear_greed > 80:  # Extreme Greed
                            if side == 'LONG':
                                conditions['should_secure_profits'] = True
                                conditions['reasons'].append(f"Sentiment: Extreme Greed ({fear_greed}) - TOP warning")

                        # Alto volumen de noticias puede indicar volatilidad
                        if news_volume > 10:
                            conditions['reasons'].append(f"Sentiment: Alto volumen noticias ({news_volume})")

                except Exception as e:
                    logger.debug(f"Error en Sentiment analysis: {e}")

            # 📈 4. MARKET REGIME DETECTOR (Bear/Bull/Sideways)
            if self.regime_detector:
                try:
                    # El regime detector está en market_analyzer, obtener régimen actual
                    regime_info = getattr(self.market_analyzer, 'current_regime', None)
                    if regime_info:
                        regime = regime_info.get('regime', 'UNKNOWN')
                        strength = regime_info.get('strength', 'MEDIUM')
                        conditions['market_regime'] = regime

                        # Ajustar estrategia según régimen
                        if regime == 'BEAR' and side == 'LONG':
                            conditions['should_secure_profits'] = True
                            conditions['reasons'].append(f"Regime: BEAR market (strength={strength}) - secure LONG")
                        elif regime == 'BULL' and side == 'SHORT':
                            conditions['should_secure_profits'] = True
                            conditions['reasons'].append(f"Regime: BULL market (strength={strength}) - secure SHORT")
                        elif regime == 'SIDEWAYS':
                            # En sideways, asegurar ganancias más rápido
                            if pnl_pct > 1.5:
                                conditions['should_secure_profits'] = True
                                conditions['reasons'].append(f"Regime: SIDEWAYS - secure at {pnl_pct:+.1f}%")

                except Exception as e:
                    logger.debug(f"Error en Regime detection: {e}")

            # 🔧 5. ARSENAL AVANZADO (Feature Aggregator)
            if self.feature_aggregator:
                try:
                    pair = symbol.replace('USDT', '/USDT')
                    current_price = position.get('mark_price', 0)

                    # 5.1 Liquidation Zones (evitar stop hunts)
                    liquidation_risk = self._check_liquidation_zones(symbol, position)
                    conditions['arsenal_signals']['liquidation_risk'] = liquidation_risk
                    if liquidation_risk > 0.7:
                        conditions['reversal_risk'] += 0.2
                        conditions['reasons'].append(f"Arsenal: Zona liquidación cercana (risk={liquidation_risk:.0%})")

                    # 5.2 Funding Rate (sentiment extremo)
                    funding_data = self._analyze_funding_rate_detailed(symbol)
                    if funding_data:
                        conditions['arsenal_signals']['funding'] = funding_data
                        if funding_data.get('is_extreme'):
                            conditions['reasons'].append(f"Arsenal: Funding {funding_data['sentiment']} ({funding_data['rate']:.4f}%)")

                    # 5.3 Volume Profile (soporte/resistencia)
                    volume_signal = self._analyze_volume_profile(symbol, position)
                    if volume_signal:
                        conditions['arsenal_signals']['volume'] = volume_signal
                        conditions['reasons'].append(volume_signal)

                    # 5.4 Pattern Recognition (reversiones)
                    pattern_signal = self._detect_reversal_patterns(symbol, position)
                    if pattern_signal:
                        conditions['reversal_risk'] += 0.3
                        conditions['arsenal_signals']['patterns'] = pattern_signal
                        conditions['reasons'].append(pattern_signal)

                    # 5.5 Order Flow Imbalance (presión compradora/vendedora)
                    order_flow_data = self._analyze_order_flow(symbol, position)
                    if order_flow_data:
                        conditions['arsenal_signals']['order_flow'] = order_flow_data
                        if order_flow_data.get('imbalance_significant'):
                            # Si el imbalance va contra nuestra posición
                            if (side == 'LONG' and order_flow_data['direction'] == 'SELL') or \
                               (side == 'SHORT' and order_flow_data['direction'] == 'BUY'):
                                conditions['reversal_risk'] += 0.2
                                conditions['reasons'].append(f"Arsenal: Order Flow contrario ({order_flow_data['direction']})")

                    # 5.6 Session Trading (volatilidad por sesión)
                    session_data = self._analyze_session(symbol)
                    if session_data:
                        conditions['arsenal_signals']['session'] = session_data
                        # En sesiones de alta volatilidad, ser más conservador
                        if session_data.get('volatility_multiplier', 1.0) > 1.3:
                            conditions['should_secure_profits'] = conditions.get('should_secure_profits', False) or (pnl_pct > 1.0)

                except Exception as e:
                    logger.debug(f"Error en Arsenal analysis: {e}")

            # 📊 6. ORDER BOOK ANALYSIS (presión buy/sell)
            if self.orderbook_analyzer:
                try:
                    ob_analysis = await self._analyze_orderbook(symbol)
                    if ob_analysis:
                        pressure = ob_analysis.get('pressure', 'NEUTRAL')

                        if pressure == 'SELL_PRESSURE' and side == 'LONG':
                            conditions['reversal_risk'] += 0.15
                            conditions['reasons'].append("OrderBook: Strong SELL pressure")
                        elif pressure == 'BUY_PRESSURE' and side == 'SHORT':
                            conditions['reversal_risk'] += 0.15
                            conditions['reasons'].append("OrderBook: Strong BUY pressure")

                except Exception as e:
                    logger.debug(f"Error en OrderBook analysis: {e}")

            # 🎯 CALCULAR CONFIANZA FINAL
            confidence_score = 0.0
            total_signals = 0

            if conditions['rl_q_values']:
                confidence_score += 0.25
                total_signals += 1
            if conditions['ml_prediction']:
                confidence_score += 0.25
                total_signals += 1
            if conditions['sentiment_score'] != 0:
                confidence_score += 0.15
                total_signals += 1
            if conditions['market_regime'] != 'UNKNOWN':
                confidence_score += 0.15
                total_signals += 1
            if conditions['arsenal_signals']:
                confidence_score += 0.20
                total_signals += 1

            conditions['confidence'] = confidence_score if total_signals > 0 else 0.0

            # Normalizar reversal_risk
            conditions['reversal_risk'] = min(1.0, conditions['reversal_risk'])

            # 🎯 LOG COMPLETO DEL ANÁLISIS MULTI-SERVICIO
            arsenal = conditions.get('arsenal_signals', {})

            # Construir resumen de indicadores
            indicators_summary = []

            # Fear & Greed
            if conditions.get('sentiment_score', 0) != 0:
                fg_value = "N/A"
                if self.sentiment_analyzer:
                    try:
                        sf = self.sentiment_analyzer.get_sentiment_features(symbol.replace('USDT', '/USDT'))
                        if sf:
                            fg_value = sf.get('fear_greed_index', 'N/A')
                    except:
                        pass
                indicators_summary.append(f"F&G: {fg_value}")

            # Regime
            if conditions.get('market_regime', 'UNKNOWN') != 'UNKNOWN':
                indicators_summary.append(f"Regime: {conditions['market_regime']}")

            # Funding
            if 'funding' in arsenal:
                funding = arsenal['funding']
                indicators_summary.append(f"Funding: {funding.get('rate', 0):.4f}% ({funding.get('sentiment', 'N/A')})")

            # Liquidation Risk
            if 'liquidation_risk' in arsenal:
                liq = arsenal['liquidation_risk']
                liq_status = '⚠️ ALTO' if liq > 0.7 else ('⚡ Medio' if liq > 0.3 else '✅ Bajo')
                indicators_summary.append(f"Liq Risk: {liq_status}")

            # Order Flow
            if 'order_flow' in arsenal:
                of = arsenal['order_flow']
                indicators_summary.append(f"Order Flow: {of.get('direction', 'N/A')} (ratio={of.get('ratio', 1.0):.2f})")

            # Session
            if 'session' in arsenal:
                sess = arsenal['session']
                indicators_summary.append(f"Session: {sess.get('name', 'N/A')} (vol={sess.get('volatility_multiplier', 1.0):.1f}x)")

            # ML Prediction
            if conditions.get('ml_prediction'):
                ml = conditions['ml_prediction']
                indicators_summary.append(f"ML: {ml.get('action', 'N/A')} ({ml.get('confidence', 0):.0f}%)")

            # Log principal
            logger.info(
                f"🧠 Market Analysis {symbol}: "
                f"Confidence={conditions['confidence']:.0%}, "
                f"Reversal Risk={conditions['reversal_risk']:.0%}, "
                f"Signals={len(conditions['reasons'])}"
            )

            # Log de indicadores si hay datos
            if indicators_summary:
                logger.info(f"   📊 {' | '.join(indicators_summary)}")

            # Log de razones específicas si hay
            if conditions['reasons']:
                for reason in conditions['reasons'][:3]:  # Max 3 razones para no saturar logs
                    logger.info(f"   • {reason}")

        except Exception as e:
            logger.error(f"❌ Error en market analysis: {e}", exc_info=True)

        return conditions

    def _get_intelligent_decision(
        self,
        action_type: str,
        position: Dict,
        market_conditions: Dict
    ) -> Dict:
        """
        Toma decisión inteligente usando análisis de mercado

        Args:
            action_type: 'breakeven', 'trailing', 'partial_tp', 'close_adverse', 'reversal'
            position: Datos de la posición
            market_conditions: Análisis de condiciones de mercado

        Returns:
            {
                'should_execute': bool,
                'confidence': float (0-1),
                'reasons': List[str],
                'risk_score': float (0-1)
            }
        """
        decision = {
            'should_execute': False,
            'confidence': 0.0,
            'reasons': [],
            'risk_score': 0.0
        }

        try:
            pnl_pct = position.get('unrealized_pnl_pct', 0)
            side = position.get('side', 'UNKNOWN')

            reversal_risk = market_conditions.get('reversal_risk', 0)
            should_secure = market_conditions.get('should_secure_profits', False)
            should_let_run = market_conditions.get('should_let_run', False)
            base_confidence = market_conditions.get('confidence', 0)
            market_reasons = market_conditions.get('reasons', [])

            if action_type == 'breakeven':
                # Breakeven: Mover SL a entrada cuando hay señales de asegurar ganancias
                if should_secure and base_confidence > 0.6:
                    decision['should_execute'] = True
                    decision['confidence'] = base_confidence
                    decision['reasons'] = ['Breakeven: Asegurar ganancias mínimas'] + market_reasons
                elif pnl_pct > 2.0 and reversal_risk > 0.5:
                    decision['should_execute'] = True
                    decision['confidence'] = 0.7
                    decision['reasons'] = [f'Breakeven: P&L={pnl_pct:.1f}%, Reversal Risk={reversal_risk:.0%}']

            elif action_type == 'trailing':
                # Trailing: Dejar correr ganancias si continuación probable
                if should_let_run and base_confidence > 0.65:
                    decision['should_execute'] = True
                    decision['confidence'] = base_confidence
                    decision['reasons'] = ['Trailing: Tendencia fuerte continúa'] + market_reasons
                elif pnl_pct > 5.0 and reversal_risk < 0.3:
                    decision['should_execute'] = True
                    decision['confidence'] = 0.75
                    decision['reasons'] = [f'Trailing: P&L alto ({pnl_pct:.1f}%), bajo riesgo reversión']

            elif action_type == 'partial_tp':
                # Partial TP: Tomar ganancias parciales si alta confianza
                if pnl_pct > 3.0 and (should_secure or reversal_risk > 0.4):
                    decision['should_execute'] = True
                    decision['confidence'] = max(base_confidence, 0.7)
                    decision['reasons'] = [f'Partial TP: P&L={pnl_pct:.1f}%, asegurar 50%'] + market_reasons
                elif pnl_pct > 6.0:  # P&L muy alto, siempre asegurar algo
                    decision['should_execute'] = True
                    decision['confidence'] = 0.85
                    decision['reasons'] = [f'Partial TP: P&L excepcional ({pnl_pct:.1f}%)']

            elif action_type == 'close_adverse':
                # Close Adverse: Cerrar si movimiento adverso significativo
                drawdown = self._position_highs.get(position.get('symbol', ''), 0) - pnl_pct
                if drawdown > 2.0 or reversal_risk > 0.7:
                    decision['should_execute'] = True
                    decision['confidence'] = 0.8
                    decision['reasons'] = [f'Close: Drawdown={drawdown:.1f}%, Reversal={reversal_risk:.0%}']

            elif action_type == 'reversal':
                # Reversal: Cerrar si alta probabilidad de reversión
                if reversal_risk > self.base_config['reversal_confidence_threshold']:
                    decision['should_execute'] = True
                    decision['confidence'] = reversal_risk
                    decision['reasons'] = [f'Reversal: High risk ({reversal_risk:.0%})'] + market_reasons

            decision['risk_score'] = reversal_risk

        except Exception as e:
            logger.error(f"Error en intelligent decision: {e}", exc_info=True)

        return decision

    def _build_rl_state(self, position: Dict, symbol: str) -> Dict:
        """Construye estado para RL Agent"""
        try:
            return {
                'symbol': symbol,
                'side': position.get('side', 'UNKNOWN'),
                'pnl_pct': position.get('unrealized_pnl_pct', 0),
                'entry_price': position.get('entry_price', 0),
                'current_price': position.get('mark_price', 0),
                'leverage': position.get('leverage', 1),
                'position_amt': position.get('position_amt', 0),
            }
        except Exception as e:
            logger.debug(f"Error building RL state: {e}")
            return {}

    def _build_ml_features(self, position: Dict, symbol: str) -> Dict:
        """Construye features para ML System"""
        try:
            return {
                'symbol': symbol,
                'pnl_pct': position.get('unrealized_pnl_pct', 0),
                'price': position.get('mark_price', 0),
                'side': 1 if position.get('side') == 'LONG' else -1,
            }
        except Exception as e:
            logger.debug(f"Error building ML features: {e}")
            return {}

    def _check_liquidation_zones(self, symbol: str, position: Dict) -> float:
        """Verifica proximidad a zonas de liquidación"""
        try:
            if not self.feature_aggregator:
                return 0.0

            pair = symbol.replace('USDT', '/USDT')
            current_price = position.get('mark_price', 0)

            # Verificar si feature_aggregator tiene liquidation_heatmap
            if not hasattr(self.feature_aggregator, 'liquidation_heatmap'):
                return 0.0

            liq_heatmap = self.feature_aggregator.liquidation_heatmap
            is_near, details = liq_heatmap.is_near_liquidation_zone(pair, current_price)

            if is_near and details:
                confidence = details.get('confidence', 0)
                logger.debug(f"Liquidation zone detected for {symbol}: confidence={confidence:.0%}")
                return confidence

            return 0.0

        except Exception as e:
            logger.debug(f"Error checking liquidation zones: {e}")
            return 0.0

    def _analyze_funding_rate(self, symbol: str) -> Optional[str]:
        """Analiza funding rate para detectar sentiment extremo"""
        try:
            if not self.feature_aggregator:
                return None

            pair = symbol.replace('USDT', '/USDT')

            # Verificar si feature_aggregator tiene funding_rate_analyzer
            if not hasattr(self.feature_aggregator, 'funding_rate_analyzer'):
                return None

            funding_analyzer = self.feature_aggregator.funding_rate_analyzer
            sentiment, strength, signal = funding_analyzer.get_funding_sentiment(pair)

            if strength == 'STRONG' and sentiment != 'neutral':
                return f"Funding: {sentiment.upper()} sentiment (strength={strength})"

            return None

        except Exception as e:
            logger.debug(f"Error analyzing funding rate: {e}")
            return None

    def _analyze_funding_rate_detailed(self, symbol: str) -> Optional[Dict]:
        """Analiza funding rate con detalles completos"""
        try:
            if not self.feature_aggregator:
                return None

            pair = symbol.replace('USDT', '/USDT')

            if not hasattr(self.feature_aggregator, 'funding_rate_analyzer'):
                return None

            funding_analyzer = self.feature_aggregator.funding_rate_analyzer
            sentiment, strength, signal = funding_analyzer.get_funding_sentiment(pair)

            # Obtener rate actual si está disponible
            rate = 0.0
            if hasattr(funding_analyzer, 'get_current_rate'):
                rate = funding_analyzer.get_current_rate(pair) or 0.0
            elif hasattr(funding_analyzer, '_funding_cache') and pair in funding_analyzer._funding_cache:
                rate = funding_analyzer._funding_cache[pair].get('rate', 0)

            return {
                'sentiment': sentiment,
                'strength': strength,
                'signal': signal,
                'rate': rate * 100,  # Convertir a porcentaje
                'is_extreme': strength == 'STRONG' and sentiment != 'neutral'
            }

        except Exception as e:
            logger.debug(f"Error analyzing funding rate detailed: {e}")
            return None

    def _analyze_order_flow(self, symbol: str, position: Dict) -> Optional[Dict]:
        """Analiza order flow imbalance"""
        try:
            if not self.feature_aggregator:
                return None

            pair = symbol.replace('USDT', '/USDT')

            if not hasattr(self.feature_aggregator, 'order_flow'):
                return None

            order_flow = self.feature_aggregator.order_flow

            # Obtener imbalance si está disponible
            if hasattr(order_flow, 'get_imbalance'):
                imbalance_data = order_flow.get_imbalance(pair)
                if imbalance_data:
                    ratio = imbalance_data.get('ratio', 1.0)
                    direction = 'BUY' if ratio > 1.2 else ('SELL' if ratio < 0.8 else 'NEUTRAL')

                    return {
                        'ratio': ratio,
                        'direction': direction,
                        'buy_volume': imbalance_data.get('buy_volume', 0),
                        'sell_volume': imbalance_data.get('sell_volume', 0),
                        'imbalance_significant': ratio > 1.5 or ratio < 0.67
                    }

            return None

        except Exception as e:
            logger.debug(f"Error analyzing order flow: {e}")
            return None

    def _analyze_session(self, symbol: str) -> Optional[Dict]:
        """Analiza sesión de trading actual"""
        try:
            if not self.feature_aggregator:
                return None

            if not hasattr(self.feature_aggregator, 'session_trading'):
                return None

            session_trading = self.feature_aggregator.session_trading

            if hasattr(session_trading, 'get_current_session'):
                session_name, session_info = session_trading.get_current_session()

                return {
                    'name': session_name,
                    'volatility_multiplier': session_info.get('volatility_multiplier', 1.0) if session_info else 1.0,
                    'is_active': session_info.get('is_active', True) if session_info else True
                }

            return None

        except Exception as e:
            logger.debug(f"Error analyzing session: {e}")
            return None

    def _analyze_volume_profile(self, symbol: str, position: Dict) -> Optional[str]:
        """Analiza volume profile para soporte/resistencia"""
        try:
            if not self.feature_aggregator:
                return None

            pair = symbol.replace('USDT', '/USDT')
            current_price = position.get('mark_price', 0)

            # Verificar si feature_aggregator tiene volume_profile
            if not hasattr(self.feature_aggregator, 'volume_profile'):
                return None

            volume_profile = self.feature_aggregator.volume_profile

            # Verificar proximidad a POC (Point of Control)
            is_near_poc, distance = volume_profile.is_near_poc(pair, current_price)
            in_value_area = volume_profile.is_in_value_area(pair, current_price)

            if is_near_poc:
                return f"Volume: Near POC (distance={distance:.1f}%)"
            elif in_value_area:
                return "Volume: In value area (strong support/resistance)"

            return None

        except Exception as e:
            logger.debug(f"Error analyzing volume profile: {e}")
            return None

    def _detect_reversal_patterns(self, symbol: str, position: Dict) -> Optional[str]:
        """Detecta patrones de reversión"""
        try:
            if not self.feature_aggregator:
                return None

            # Verificar si feature_aggregator tiene pattern_recognition
            if not hasattr(self.feature_aggregator, 'pattern_recognition'):
                return None

            pair = symbol.replace('USDT', '/USDT')
            pattern_recognition = self.feature_aggregator.pattern_recognition

            # Pattern recognition necesita OHLC data que no tenemos en tiempo real aquí
            # Por ahora, retornar None hasta tener acceso a OHLC en tiempo real
            # TODO: Integrar cuando tengamos OHLC data disponible

            return None

        except Exception as e:
            logger.debug(f"Error detecting patterns: {e}")
            return None

    async def _analyze_orderbook(self, symbol: str) -> Optional[Dict]:
        """Analiza order book para presión buy/sell"""
        if not self.orderbook_analyzer:
            return None

        try:
            # TODO: Implementar cuando orderbook_analyzer esté disponible
            return None
        except Exception as e:
            logger.debug(f"Error analyzing orderbook: {e}")
            return None

    async def _set_breakeven(self, symbol: str, position: Dict):
        """Mueve stop loss a precio de entrada (breakeven)"""
        try:
            entry_price = position.get('entry_price', 0)
            side = position.get('side', 'UNKNOWN')

            # Calcular nuevo SL en breakeven
            if side == 'LONG':
                new_sl = entry_price * 0.999  # -0.1% para evitar ejecución prematura
            else:  # SHORT
                new_sl = entry_price * 1.001  # +0.1%

            # Modificar SL en Binance
            await self.futures_trader.modify_stop_loss(symbol, new_sl)

            logger.info(f"✅ {symbol}: SL movido a breakeven ${new_sl:.4f}")

        except Exception as e:
            logger.error(f"❌ Error en breakeven para {symbol}: {e}")

    async def _apply_trailing_stop(
        self,
        symbol: str,
        position: Dict,
        highest_pnl: float,
        market_conditions: Dict
    ):
        """Aplica trailing stop loss DINÁMICO basado en condiciones"""
        try:
            current_price = position.get('mark_price', 0)
            side = position.get('side', 'UNKNOWN')

            # Distancia DINÁMICA según confianza
            confidence = market_conditions.get('confidence', 0.5)
            base_distance = 1.0

            # Mayor confianza = SL más apretado
            if confidence > 0.8:
                distance_pct = base_distance * 0.7  # 0.7%
            elif confidence > 0.65:
                distance_pct = base_distance  # 1.0%
            else:
                distance_pct = base_distance * 1.3  # 1.3%

            # Calcular precio del trailing stop
            if side == 'LONG':
                new_sl = current_price * (1 - distance_pct / 100)
            else:  # SHORT
                new_sl = current_price * (1 + distance_pct / 100)

            logger.info(
                f"📈 {symbol}: Trailing stop dinámico "
                f"(Distance: {distance_pct:.1f}%, Conf: {confidence:.0%})"
            )

            await self.futures_trader.modify_stop_loss(symbol, new_sl)
            logger.info(f"✅ {symbol}: Trailing stop aplicado ${new_sl:.4f}")

        except Exception as e:
            logger.error(f"❌ Error en trailing stop para {symbol}: {e}")

    async def _partial_take_profit(self, symbol: str, position: Dict):
        """Cierra 50% de la posición para asegurar ganancias"""
        try:
            quantity = abs(position.get('position_amt', 0))
            partial_qty = quantity * 0.5

            logger.info(f"💰 {symbol}: Ejecutando partial TP - Cerrando 50% ({partial_qty:.4f} qty)")

            # Cerrar 50% de la posición
            self.futures_trader.close_partial_position(symbol, partial_qty)

            self._partial_closed.add(symbol)
            logger.info(f"✅ {symbol}: Partial TP ejecutado exitosamente")

        except Exception as e:
            logger.error(f"❌ Error en partial TP para {symbol}: {e}")

    async def _close_on_adverse_move(self, symbol: str, position: Dict, drawdown: float, pnl_pct: float):
        """Cierra posición por movimiento adverso significativo (SOLO si está en pérdida)"""
        try:
            # ✅ Doble verificación: NUNCA cerrar si está en profit
            if pnl_pct > 0:
                logger.warning(
                    f"🛡️ {symbol}: ABORTANDO cierre - Posición en PROFIT ({pnl_pct:+.2f}%)"
                )
                return

            logger.warning(
                f"⚠️ {symbol}: Movimiento adverso detectado - Drawdown: {drawdown:.2f}%, "
                f"P&L actual: {pnl_pct:+.2f}% - CERRANDO posición en pérdida"
            )

            self.futures_trader.close_position(symbol, reason='ADVERSE_MOVE')

            # Limpiar tracking
            self._position_highs.pop(symbol, None)
            self._position_lows.pop(symbol, None)
            self._partial_closed.discard(symbol)
            self._breakeven_set.discard(symbol)

            logger.info(f"✅ {symbol}: Posición cerrada por movimiento adverso")

        except Exception as e:
            logger.error(f"❌ Error cerrando {symbol} por movimiento adverso: {e}")

    def get_management_stats(self) -> Dict:
        """Obtiene estadísticas de gestión de trades"""
        stats = {
            'positions_tracked': len(self._position_highs),
            'partial_tps_executed': len(self._partial_closed),
            'breakevens_set': len(self._breakeven_set),
            'position_highs': self._position_highs.copy(),
            'config': self.base_config.copy(),
            'services_active': {
                'rl_agent': self.rl_agent is not None,
                'ml_system': self.ml_system is not None,
                'sentiment': self.sentiment_analyzer is not None,
                'regime_detector': self.regime_detector is not None,
                'feature_aggregator': self.feature_aggregator is not None,
                'orderbook': self.orderbook_analyzer is not None,
            }
        }

        # 🧠 Añadir estadísticas del sistema de aprendizaje
        if self.learning:
            stats['learning'] = self.learning.get_statistics()
            stats['learning_insights'] = self.learning.get_insights()

        return stats
