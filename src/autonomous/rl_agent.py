"""
Reinforcement Learning Agent - Aprende de trades y optimiza decisiones
Usa Q-Learning con Experience Replay para aprender patrones de trading exitosos
"""
import numpy as np
import logging
from typing import Dict, List, Tuple
from collections import deque
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class RLAgent:
    """
    Agente de Reinforcement Learning para toma de decisiones autónomas
    - Aprende de outcomes de trades (rewards/penalties)
    - Ajusta estrategia basado en experiencia
    - Toma decisiones sobre agresividad, timing, position sizing
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        exploration_rate: float = 0.3,
        exploration_decay: float = 0.995,
        min_exploration: float = 0.05,
        memory_size: int = 10000
    ):
        """
        Args:
            learning_rate: Tasa de aprendizaje (alpha)
            discount_factor: Factor de descuento para recompensas futuras (gamma)
            exploration_rate: Probabilidad de explorar vs explotar (epsilon)
            exploration_decay: Decaimiento de exploration_rate
            min_exploration: Mínima exploration_rate
            memory_size: Tamaño del replay buffer
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration = min_exploration

        # Q-Table: {state_hash: {action: q_value}}
        self.q_table: Dict[str, Dict[str, float]] = {}

        # Experience Replay Memory
        self.memory = deque(maxlen=memory_size)

        # Estadísticas
        self.total_trades = 0
        self.successful_trades = 0
        self.total_reward = 0.0
        self.episode_rewards = []

        # State/Action history
        self.current_state = None
        self.current_action = None

        logger.info(f"🧠 RL Agent inicializado (LR={learning_rate}, Gamma={discount_factor}, Epsilon={exploration_rate})")

    def get_state_representation(self, market_data: Dict) -> str:
        """
        Convierte datos de mercado a representación de estado MULTIDIMENSIONAL
        Formato: "{pair}_{side}_{rsi_range}_{regime}_{regime_strength}_{orderbook}_{confidence_range}_{trade_count_tier}_{fg_tier}_{ml_signal}_{news_state}_{sentiment}"

        INTEGRACIÓN COMPLETA de los 16 servicios en el estado del RL Agent

        Args:
            market_data: Dict con TODOS los indicadores de los 16 servicios

        Returns:
            Hash único del estado (12 dimensiones)
        """
        # Extraer features clave (originales)
        pair = market_data.get('pair', 'UNKNOWN')
        side = market_data.get('side', 'NEUTRAL')  # BUY/SELL/NEUTRAL
        rsi = market_data.get('rsi', 50)
        regime = market_data.get('regime', 'SIDEWAYS')
        regime_strength = market_data.get('regime_strength', 'MEDIUM')  # LOW/MEDIUM/HIGH
        orderbook = market_data.get('orderbook', 'NEUTRAL')  # BUY_PRESSURE/SELL_PRESSURE/NEUTRAL
        confidence = market_data.get('confidence', 50)
        total_trades = market_data.get('total_trades', 0)

        # NUEVOS FEATURES de los 16 servicios
        fear_greed_index = market_data.get('fear_greed_index', 50)
        ml_prediction = market_data.get('ml_prediction', 'HOLD')
        ml_confidence = market_data.get('ml_confidence', 0)
        news_triggered = market_data.get('news_triggered', False)
        overall_sentiment = market_data.get('overall_sentiment', 'neutral')
        pre_pump_score = market_data.get('pre_pump_score', 0)
        multi_layer_alignment = market_data.get('multi_layer_alignment', 0)

        # Discretizar RSI
        if rsi < 30:
            rsi_range = 'OVERSOLD'
        elif rsi < 40:
            rsi_range = 'LOW'
        elif rsi < 60:
            rsi_range = 'NEUTRAL'
        elif rsi < 70:
            rsi_range = 'HIGH'
        else:
            rsi_range = 'OVERBOUGHT'

        # Discretizar confidence
        if confidence < 40:
            confidence_range = 'LOW'
        elif confidence < 60:
            confidence_range = 'MEDIUM'
        elif confidence < 80:
            confidence_range = 'HIGH'
        else:
            confidence_range = 'VERY_HIGH'

        # Determinar tier de experiencia (para leverage de futuros)
        if total_trades < 50:
            trade_count_tier = 'TC_0_50'
        elif total_trades < 100:
            trade_count_tier = 'TC_50_100'
        elif total_trades < 150:
            trade_count_tier = 'TC_100_150'
        elif total_trades < 500:
            trade_count_tier = 'TC_150_500'
        else:
            trade_count_tier = 'TC_500_PLUS'

        # NUEVA DIMENSIÓN: Fear & Greed tier
        if fear_greed_index < 25:
            fg_tier = 'FG_EXTREME_FEAR'  # Gran oportunidad de compra
        elif fear_greed_index < 45:
            fg_tier = 'FG_FEAR'
        elif fear_greed_index < 55:
            fg_tier = 'FG_NEUTRAL'
        elif fear_greed_index < 75:
            fg_tier = 'FG_GREED'
        else:
            fg_tier = 'FG_EXTREME_GREED'  # Cuidado con comprar

        # NUEVA DIMENSIÓN: ML Signal (con confianza mínima)
        if ml_confidence > 0.6:
            if ml_prediction == 'BUY':
                ml_signal = 'ML_BUY'
            elif ml_prediction == 'SELL':
                ml_signal = 'ML_SELL'
            else:
                ml_signal = 'ML_HOLD'
        else:
            ml_signal = 'ML_NONE'  # Confianza insuficiente

        # NUEVA DIMENSIÓN: News State
        news_state = 'NEWS_YES' if news_triggered else 'NEWS_NO'

        # NUEVA DIMENSIÓN: Sentiment
        if overall_sentiment == 'positive':
            sentiment = 'SENT_POS'
        elif overall_sentiment == 'negative':
            sentiment = 'SENT_NEG'
        else:
            sentiment = 'SENT_NEU'

        # Crear estado compuesto con 12 DIMENSIONES
        state = (
            f"{pair}_{side}_{rsi_range}_{regime}_{regime_strength}_{orderbook}_"
            f"{confidence_range}_{trade_count_tier}_{fg_tier}_{ml_signal}_{news_state}_{sentiment}"
        )

        return state

    def choose_action(self, state: str, available_actions: List[str]) -> str:
        """
        Selecciona acción usando epsilon-greedy policy

        Args:
            state: Estado actual
            available_actions: Lista de acciones disponibles

        Returns:
            Acción seleccionada
        """
        # Epsilon-greedy: explorar vs explotar
        if np.random.random() < self.exploration_rate:
            # Explorar: acción aleatoria
            action = np.random.choice(available_actions)
            logger.debug(f"🎲 Explorando: acción aleatoria '{action}'")
        else:
            # Explotar: mejor acción conocida
            if state in self.q_table:
                # Elegir acción con mayor Q-value
                state_actions = self.q_table[state]
                best_action = max(
                    available_actions,
                    key=lambda a: state_actions.get(a, 0.0)
                )
                action = best_action
                logger.debug(f"🎯 Explotando: mejor acción '{action}' (Q={state_actions.get(action, 0):.3f})")
            else:
                # Estado desconocido: acción aleatoria
                action = np.random.choice(available_actions)
                logger.debug(f"❓ Estado nuevo: acción aleatoria '{action}'")

        # Guardar estado y acción actual
        self.current_state = state
        self.current_action = action

        return action

    def decide_trade_action(self, market_data: Dict, max_leverage: int = 1) -> Dict:
        """
        Decide si abrir un trade y con qué parámetros basado en el estado del mercado

        Args:
            market_data: Datos del mercado (indicadores, sentiment, etc.)
            max_leverage: Máximo leverage permitido basado en experiencia (1-20x)

        Returns:
            Dict con decisión: {
                'should_trade': bool,
                'action': str,  # 'OPEN', 'SKIP'
                'trade_type': str,  # 'SPOT' o 'FUTURES'
                'position_size_multiplier': float,  # 1.0 = normal, 0.5 = conservador, 1.5 = agresivo
                'leverage': int,  # 1-20x (solo para FUTURES)
                'confidence': float  # 0-1
            }
        """
        # Obtener representación del estado
        state = self.get_state_representation(market_data)

        # 🔍 DEBUG: Log valores de entrada para diagnóstico
        pair = market_data.get('pair', 'UNKNOWN')
        side = market_data.get('side', 'NEUTRAL')
        input_confidence = market_data.get('confidence', 50)
        logger.info(
            f"🔍 RL INPUT: {pair} {side} | "
            f"confidence={input_confidence} | "
            f"rsi={market_data.get('rsi', 50):.0f} | "
            f"regime={market_data.get('regime', 'N/A')} | "
            f"ml_conf={market_data.get('ml_confidence', 0):.2f} | "
            f"fg={market_data.get('fear_greed_index', 50)}"
        )

        # ======================================================================
        # COMPOSITE SCORE: Calcular score de oportunidad usando LOS 16 SERVICIOS
        # ======================================================================
        composite_score = 0.0

        # 🔧 FIX CRÍTICO: BASE SCORE - Confidence general de la señal (peso fundamental: 5.0)
        # Esta es la base más importante: si la señal tiene alta confidence, el composite debe reflejarlo
        signal_confidence = market_data.get('confidence', 50)  # 0-100
        confidence_normalized = signal_confidence / 100.0  # Normalizar a 0-1

        # Convertir confidence a puntos (0-5.0 puntos)
        confidence_score = confidence_normalized * 5.0
        composite_score += confidence_score
        logger.debug(f"🎯 Base confidence: {signal_confidence:.1f}% → {confidence_score:.2f} puntos")

        # 🔧 FIX CRÍTICO: PATTERN DETECTION - Si hay patterns detectados (peso alto: 2.0)
        pattern_detected = market_data.get('pattern_detected', False)
        pattern_type = market_data.get('pattern_type', 'UNKNOWN')
        pattern_confidence = market_data.get('pattern_confidence', 0)

        if pattern_detected and pattern_confidence > 0.8:
            composite_score += 2.0
            logger.debug(f"🎯 Pattern detectado: {pattern_type} (conf={pattern_confidence:.2f}) (+2.0)")
        elif pattern_detected and pattern_confidence > 0.6:
            composite_score += 1.0
            logger.debug(f"🎯 Pattern detectado: {pattern_type} (conf={pattern_confidence:.2f}) (+1.0)")

        # 🔧 FIX CRÍTICO: ORDER FLOW BOOST - Si hay boost de order flow (peso moderado: 1.5)
        order_flow_ratio = market_data.get('order_flow_ratio', 1.0)
        if order_flow_ratio >= 1.5:  # Boost fuerte (1.5x+)
            composite_score += 1.5
            logger.debug(f"💹 Order flow boost fuerte: {order_flow_ratio:.2f}x (+1.5)")
        elif order_flow_ratio >= 1.2:  # Boost moderado (1.2x+)
            composite_score += 0.75
            logger.debug(f"💹 Order flow boost moderado: {order_flow_ratio:.2f}x (+0.75)")

        # 1. CryptoPanic pre-pump score (peso alto: 2.0)
        pre_pump_score = market_data.get('pre_pump_score', 0)
        if pre_pump_score > 70:
            composite_score += 2.0
            logger.debug(f"✨ Pre-pump score alto: {pre_pump_score} (+2.0)")
        elif pre_pump_score > 50:
            composite_score += 1.0

        # 2. Fear & Greed Index (oportunidad en miedo extremo)
        fear_greed_index = market_data.get('fear_greed_index', 50)
        if fear_greed_index < 20:  # Extreme fear = gran oportunidad
            composite_score += 2.0
            logger.debug(f"😨 Fear extremo: {fear_greed_index} (+2.0)")
        elif fear_greed_index < 35:  # Fear = oportunidad
            composite_score += 1.0
        elif fear_greed_index > 80:  # Extreme greed = cuidado
            composite_score -= 1.5
            logger.debug(f"🤑 Greed extremo: {fear_greed_index} (-1.5)")
        elif fear_greed_index > 70:
            composite_score -= 0.5

        # 3. ML Prediction (peso muy alto: 2.5)
        ml_prediction = market_data.get('ml_prediction', 'HOLD')
        ml_confidence = market_data.get('ml_confidence', 0)
        side = market_data.get('side', 'NEUTRAL')
        if ml_confidence > 0.7:
            if (ml_prediction == 'BUY' and side == 'BUY') or (ml_prediction == 'SELL' and side == 'SELL'):
                composite_score += 2.5
                logger.debug(f"🤖 ML confirma {ml_prediction} con {ml_confidence:.2f} confianza (+2.5)")
            elif ml_prediction != side and ml_prediction != 'HOLD':
                composite_score -= 1.5  # ML contradice la señal
                logger.debug(f"⚠️ ML contradice: predice {ml_prediction} pero señal es {side} (-1.5)")

        # 4. Multi-layer alignment (peso muy alto: 3.0)
        multi_layer_alignment = market_data.get('multi_layer_alignment', 0)
        if multi_layer_alignment > 0.8:
            composite_score += 3.0
            logger.debug(f"📊 Multi-layer alineación alta: {multi_layer_alignment:.2f} (+3.0)")
        elif multi_layer_alignment > 0.6:
            composite_score += 1.5

        # 5. News triggered (peso moderado: 1.0)
        news_triggered = market_data.get('news_triggered', False)
        news_confidence = market_data.get('news_trigger_confidence', 0)
        if news_triggered and news_confidence > 70:
            composite_score += 1.5
            logger.debug(f"📰 News triggered con alta confianza (+1.5)")
        elif news_triggered:
            composite_score += 0.5

        # 6. Order book pressure (peso: 1.0)
        market_pressure = market_data.get('market_pressure', 'NEUTRAL')
        if (market_pressure == 'BUY_PRESSURE' and side == 'BUY') or \
           (market_pressure == 'SELL_PRESSURE' and side == 'SELL'):
            composite_score += 1.0
            logger.debug(f"📖 Orderbook pressure alineado: {market_pressure} (+1.0)")
        elif market_pressure != 'NEUTRAL' and market_pressure != side + '_PRESSURE':
            composite_score -= 0.5

        # 7. Sentiment (peso moderado: 1.0)
        overall_sentiment = market_data.get('overall_sentiment', 'neutral')
        sentiment_strength = market_data.get('sentiment_strength', 0)
        if (overall_sentiment == 'positive' and side == 'BUY') or \
           (overall_sentiment == 'negative' and side == 'SELL'):
            if sentiment_strength > 0.7:
                composite_score += 1.5
                logger.debug(f"💭 Sentiment fuerte alineado: {overall_sentiment} ({sentiment_strength:.2f}) (+1.5)")
            else:
                composite_score += 0.5

        # 8. Regime confidence
        regime_confidence = market_data.get('regime_confidence', 0)
        if regime_confidence > 0.75:
            composite_score += 0.5

        # 🔍 DEBUG: Log detallado del composite score breakdown
        logger.info(
            f"🎯 COMPOSITE SCORE: {composite_score:.2f}/18 | "
            f"base_conf={confidence_score:.1f}/5 | "
            f"ml={'+2.5' if (ml_confidence > 0.7 and ((ml_prediction == 'BUY' and side == 'BUY') or (ml_prediction == 'SELL' and side == 'SELL'))) else '0'} | "
            f"multilayer={'+3' if multi_layer_alignment > 0.8 else ('+1.5' if multi_layer_alignment > 0.6 else '0')}"
        )

        # Acciones disponibles para trading (SPOT + FUTURES)
        available_actions = [
            'SKIP',  # No abrir trade
            # SPOT (4 acciones)
            'OPEN_CONSERVATIVE',  # Abrir spot con tamaño conservador (50% del normal)
            'OPEN_NORMAL',  # Abrir spot con tamaño normal (100%)
            'OPEN_AGGRESSIVE',  # Abrir spot con tamaño agresivo (150% del normal)
            # FUTURES (3 acciones - solo si max_leverage > 1)
            'FUTURES_LOW',  # Futuros con 20-40% del max leverage
            'FUTURES_MEDIUM',  # Futuros con 40-70% del max leverage
            'FUTURES_HIGH'  # Futuros con 70-100% del max leverage
        ]

        # Si no tiene leverage desbloqueado, remover acciones de futuros
        if max_leverage <= 1:
            available_actions = [a for a in available_actions if not a.startswith('FUTURES')]

        # ======================================================================
        # MODO EXPLORACIÓN: Decisión basada en composite score
        # ======================================================================
        # Threshold dinámico del sistema es ~4.5, usamos eso como referencia
        EXPLORATION_THRESHOLD = 4.5  # Consistente con DynamicThresholdManager

        original_exploration = self.exploration_rate

        # 🚀 MODO EXPLORACIÓN: Si score >= threshold, FORZAR trade (no SKIP)
        if composite_score >= EXPLORATION_THRESHOLD:
            # Score por encima del threshold = señal válida, NO permitir SKIP
            actions_without_skip = [a for a in available_actions if a != 'SKIP']

            if composite_score >= 6.0:
                # Señal ULTRA fuerte: 100% explotación, preferir FUTURES_HIGH
                self.exploration_rate = 0.0
                logger.info(f"🚀 Señal ULTRA FUERTE (score {composite_score:.2f}): favoreciendo FUTURES agresivo")

                # Elegir mejor acción de FUTURES disponible
                if 'FUTURES_HIGH' in actions_without_skip and max_leverage >= 5:
                    chosen_action = 'FUTURES_HIGH'
                elif 'FUTURES_MEDIUM' in actions_without_skip:
                    chosen_action = 'FUTURES_MEDIUM'
                elif 'FUTURES_LOW' in actions_without_skip:
                    chosen_action = 'FUTURES_LOW'
                else:
                    chosen_action = self.choose_action(state, actions_without_skip)

            elif composite_score >= 5.0:
                # Señal MUY fuerte: reducir exploración, preferir FUTURES_MEDIUM
                self.exploration_rate = max(0.05, self.exploration_rate * 0.2)
                logger.info(f"✅ Señal MUY FUERTE (score {composite_score:.2f}): favoreciendo FUTURES")

                if 'FUTURES_MEDIUM' in actions_without_skip:
                    chosen_action = 'FUTURES_MEDIUM'
                elif 'FUTURES_LOW' in actions_without_skip:
                    chosen_action = 'FUTURES_LOW'
                else:
                    chosen_action = self.choose_action(state, actions_without_skip)

            else:
                # Señal BUENA (4.5-5.0): permitir exploración pero sin SKIP
                self.exploration_rate = max(0.1, self.exploration_rate * 0.5)
                logger.info(f"👍 Señal BUENA (score {composite_score:.2f}): explorando sin SKIP")
                chosen_action = self.choose_action(state, actions_without_skip)

        elif composite_score < 2.0:
            # Score muy bajo: favorecer SKIP
            self.exploration_rate = min(0.9, self.exploration_rate * 2.0)
            logger.info(f"⚠️ Señal DÉBIL (score {composite_score:.2f}): favoreciendo SKIP")
            chosen_action = self.choose_action(state, available_actions)

        else:
            # Score intermedio (2.0-4.5): lógica normal con SKIP disponible
            logger.info(f"🤔 Señal INTERMEDIA (score {composite_score:.2f}): decisión RL normal")
            chosen_action = self.choose_action(state, available_actions)

        # Restaurar exploration_rate original
        self.exploration_rate = original_exploration

        # Convertir acción a decisión de trading
        if chosen_action == 'SKIP':
            decision = {
                'should_trade': False,
                'action': 'SKIP',
                'trade_type': 'SPOT',
                'position_size_multiplier': 0.0,
                'leverage': 1,
                'confidence': self._get_action_confidence(state, chosen_action),
                'chosen_action': chosen_action,
                'composite_score': composite_score  # Para telemetría/debugging
            }
        elif chosen_action.startswith('FUTURES'):
            # Acciones de FUTURES
            # 🔧 FIX: Cuando max_leverage es bajo (<=5), usar directamente
            # para evitar que int(3 * 0.3) = 0 → max(2, 0) = 2
            if max_leverage <= 5:
                # Con leverage bajo, usar max_leverage directamente
                leverage = max_leverage
                if chosen_action == 'FUTURES_LOW':
                    multiplier = 0.5
                elif chosen_action == 'FUTURES_MEDIUM':
                    multiplier = 1.0
                else:  # FUTURES_HIGH
                    multiplier = 1.5
            else:
                # Con leverage alto, escalar proporcionalmente
                if chosen_action == 'FUTURES_LOW':
                    leverage = max(3, int(max_leverage * 0.3))
                    multiplier = 0.5
                elif chosen_action == 'FUTURES_MEDIUM':
                    leverage = max(3, int(max_leverage * 0.55))
                    multiplier = 1.0
                else:  # FUTURES_HIGH
                    leverage = max(3, int(max_leverage * 0.85))
                    multiplier = 1.5

            decision = {
                'should_trade': True,
                'action': 'OPEN',
                'trade_type': 'FUTURES',
                'position_size_multiplier': multiplier,
                'leverage': leverage,
                'confidence': self._get_action_confidence(state, chosen_action),
                'chosen_action': chosen_action,
                'composite_score': composite_score  # Para telemetría/debugging
            }
        else:
            # Acciones LONG/SHORT conservadoras (OPEN_CONSERVATIVE, OPEN_NORMAL, OPEN_AGGRESSIVE)
            # Migrado a FUTURES con leverage 1x (equivalente a SPOT pero permite SHORT)
            if chosen_action == 'OPEN_CONSERVATIVE':
                multiplier = 0.5
            elif chosen_action == 'OPEN_AGGRESSIVE':
                multiplier = 1.5
            else:  # OPEN_NORMAL
                multiplier = 1.0

            decision = {
                'should_trade': True,
                'action': 'OPEN',
                'trade_type': 'FUTURES',  # Migrado a FUTURES (permite LONG y SHORT)
                'position_size_multiplier': multiplier,
                'leverage': 1,  # 1x = sin apalancamiento (equivalente a SPOT)
                'confidence': self._get_action_confidence(state, chosen_action),
                'chosen_action': chosen_action,
                'composite_score': composite_score  # Para telemetría/debugging
            }

        logger.info(
            f"🤖 RL Decision: {chosen_action} | "
            f"Trade: {decision['should_trade']} | "
            f"Type: {decision['trade_type']} | "
            f"Size: {decision['position_size_multiplier']:.1f}x | "
            f"Leverage: {decision['leverage']}x | "
            f"Confidence: {decision['confidence']:.2f}"
        )

        return decision

    def _get_action_confidence(self, state: str, action: str) -> float:
        """
        Calcula confianza en una acción basado en Q-values

        Returns:
            Confianza entre 0 y 1
        """
        if state not in self.q_table or action not in self.q_table[state]:
            # 🔧 Aumentado de 0.3 a 0.6 para permitir operar en estados similares
            # Con 116M estados posibles y 224 trades, necesitamos interpolar
            return 0.6  # Confianza moderada - permitir aprendizaje activo

        q_value = self.q_table[state][action]
        all_q_values = list(self.q_table[state].values())

        if not all_q_values:
            return 0.3

        # Normalizar Q-value a rango 0-1
        min_q = min(all_q_values)
        max_q = max(all_q_values)

        if max_q == min_q:
            return 0.5

        normalized = (q_value - min_q) / (max_q - min_q)

        # Ajustar confianza basado en experiencia
        # 🔧 Reducido de 100 a 50 para alcanzar max confidence más rápido
        experience_factor = min(1.0, self.total_trades / 50)  # Más confianza con más trades

        return 0.3 + (normalized * 0.7 * experience_factor)

    def learn_from_trade(self, reward: float, next_state: str = None, done: bool = False):
        """
        Aprende de un trade completado

        Args:
            reward: Recompensa obtenida (% profit/loss)
            next_state: Siguiente estado (None si trade cerrado)
            done: True si episodio terminó
        """
        if self.current_state is None or self.current_action is None:
            return

        # Guardar experiencia en memoria
        self.memory.append({
            'state': self.current_state,
            'action': self.current_action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        })

        # Update Q-value
        self._update_q_value(self.current_state, self.current_action, reward, next_state, done)

        # Estadísticas
        self.total_trades += 1
        self.total_reward += reward
        if reward > 0:
            self.successful_trades += 1

        logger.info(
            f"📚 Aprendizaje: Estado='{self.current_state[:30]}...' | "
            f"Acción='{self.current_action}' | Reward={reward:.3f} | "
            f"Win Rate={self.get_success_rate():.1f}%"
        )

        # Decay exploration rate
        self.exploration_rate = max(
            self.min_exploration,
            self.exploration_rate * self.exploration_decay
        )

        # Reset current state/action si episodio terminó
        if done:
            self.episode_rewards.append(reward)
            self.current_state = None
            self.current_action = None

    def _update_q_value(self, state: str, action: str, reward: float,
                       next_state: str, done: bool):
        """
        Actualiza Q-value usando ecuación de Bellman
        Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
        """
        # Inicializar estado si no existe
        if state not in self.q_table:
            self.q_table[state] = {}

        # Q-value actual
        current_q = self.q_table[state].get(action, 0.0)

        # Max Q-value del siguiente estado
        if done or next_state is None:
            max_next_q = 0.0
        else:
            if next_state not in self.q_table:
                self.q_table[next_state] = {}
            max_next_q = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0.0

        # Ecuación de Bellman
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )

        # Update Q-table
        self.q_table[state][action] = new_q

        logger.debug(f"Q-update: Q({state[:20]}..., {action}) = {current_q:.3f} → {new_q:.3f}")

    def replay_experience(self, batch_size: int = 32):
        """
        Experience Replay: re-entrena con experiencias pasadas
        Ayuda a romper correlaciones temporales y mejorar estabilidad

        Args:
            batch_size: Tamaño del batch para replay
        """
        if len(self.memory) < batch_size:
            return

        # Sample aleatorio de experiencias
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        batch = [self.memory[i] for i in indices]

        # Re-aprender de cada experiencia
        for experience in batch:
            self._update_q_value(
                experience['state'],
                experience['action'],
                experience['reward'],
                experience['next_state'],
                experience['done']
            )

        logger.debug(f"🔄 Experience Replay: {batch_size} experiencias re-procesadas")

    def get_success_rate(self) -> float:
        """Retorna tasa de éxito de trades"""
        if self.total_trades == 0:
            return 0.0
        return (self.successful_trades / self.total_trades) * 100

    def get_average_reward(self) -> float:
        """Retorna recompensa promedio"""
        if self.total_trades == 0:
            return 0.0
        return self.total_reward / self.total_trades

    def get_statistics(self) -> Dict:
        """Retorna estadísticas completas del agente"""
        return {
            'total_trades': self.total_trades,
            'successful_trades': self.successful_trades,
            'success_rate': self.get_success_rate(),
            'total_reward': self.total_reward,
            'average_reward': self.get_average_reward(),
            'exploration_rate': self.exploration_rate,
            'q_table_size': len(self.q_table),
            'memory_size': len(self.memory),
            'episodes_completed': len(self.episode_rewards)
        }

    def save_to_dict(self) -> Dict:
        """Exporta agente a diccionario para persistencia"""
        return {
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'exploration_rate': self.exploration_rate,
            'min_exploration': self.min_exploration,
            'q_table': self.q_table,
            'statistics': self.get_statistics(),
            'episode_rewards': self.episode_rewards,
            'memory': list(self.memory),  # Experience buffer para replay
            'timestamp': datetime.now().isoformat()
        }

    def load_from_dict(self, data: Dict, merge: bool = False):
        """
        Carga agente desde diccionario

        Args:
            data: Diccionario con estado del agente
            merge: Si True, combina con datos existentes (acumula estadísticas y Q-table)
                   Si False, reemplaza completamente (comportamiento por defecto)
        """
        self.learning_rate = data.get('learning_rate', self.learning_rate)
        self.discount_factor = data.get('discount_factor', self.discount_factor)
        self.exploration_rate = data.get('exploration_rate', self.exploration_rate)
        self.min_exploration = data.get('min_exploration', self.min_exploration)

        if merge:
            # MODO MERGE: Combinar Q-table y acumular estadísticas
            loaded_q_table = data.get('q_table', {})

            # Merge Q-table: promediar Q-values si el estado ya existe
            for state, actions in loaded_q_table.items():
                if state in self.q_table:
                    # Promediar Q-values para acciones existentes
                    for action, q_value in actions.items():
                        if action in self.q_table[state]:
                            # Promediar: dar más peso a valores con más trades
                            current_weight = min(1.0, self.total_trades / 100)
                            loaded_weight = 1.0 - current_weight
                            self.q_table[state][action] = (
                                self.q_table[state][action] * current_weight +
                                q_value * loaded_weight
                            )
                        else:
                            self.q_table[state][action] = q_value
                else:
                    # Nuevo estado, agregar completamente
                    self.q_table[state] = actions

            # Acumular estadísticas
            stats = data.get('statistics', {})
            self.total_trades += stats.get('total_trades', 0)
            self.successful_trades += stats.get('successful_trades', 0)
            self.total_reward += stats.get('total_reward', 0.0)

            # Agregar episode rewards
            imported_rewards = data.get('episode_rewards', [])
            self.episode_rewards.extend(imported_rewards)

            # Merge experience buffer
            loaded_memory = data.get('memory', [])
            if loaded_memory:
                # Convertir a list de tuples para agregar al deque
                for experience in loaded_memory:
                    if len(experience) == 4:  # (state, action, reward, next_state)
                        self.memory.append(tuple(experience))
                logger.debug(f"  ✅ Experience buffer merged: +{len(loaded_memory)} experiencias")

            logger.info(
                f"🔄 RL Agent MERGED: {stats.get('total_trades', 0)} trades importados, "
                f"Total acumulado: {self.total_trades} trades, "
                f"{self.get_success_rate():.1f}% win rate, "
                f"{len(self.q_table)} estados aprendidos"
            )
        else:
            # MODO REPLACE: Reemplazar completamente (comportamiento original)
            self.q_table = data.get('q_table', {})

            stats = data.get('statistics', {})
            self.total_trades = stats.get('total_trades', 0)
            self.successful_trades = stats.get('successful_trades', 0)
            self.total_reward = stats.get('total_reward', 0.0)

            self.episode_rewards = data.get('episode_rewards', [])

            # Restaurar experience buffer
            loaded_memory = data.get('memory', [])
            if loaded_memory:
                # Convertir list a deque, preservando maxlen
                self.memory = deque(
                    [tuple(exp) if isinstance(exp, list) else exp for exp in loaded_memory],
                    maxlen=self.memory.maxlen
                )
                logger.debug(f"  ✅ Experience buffer restaurado: {len(self.memory)} experiencias")
            else:
                logger.debug("  ℹ️ No hay experience buffer en export (puede ser antiguo)")

            logger.info(
                f"✅ RL Agent cargado: {self.total_trades} trades, "
                f"{self.get_success_rate():.1f}% win rate, "
                f"{len(self.q_table)} estados aprendidos"
            )

    def get_autonomous_decision(self, market_data: Dict) -> Dict:
        """
        🤖 DECISIÓN 100% AUTÓNOMA - La IA decide TODO dinámicamente

        Restaura la autonomía de v1.0 (87.95% win rate) donde la IA tiene
        libertad total para decidir:
        - Si operar o no (SKIP)
        - Leverage (1x-20x)
        - TPs múltiples con porcentajes dinámicos
        - Tamaño de posición (3%-8%)

        Returns:
            Dict con decisión completa:
            {
                'action': 'OPEN' | 'SKIP',
                'leverage': int (1-20),
                'tp_percentages': [tp1, tp2, tp3],  # 3 TPs para scalping
                'position_size_pct': float (3.0-8.0),
                'confidence': float (0-1),
                'chosen_action': str,
                'composite_score': float
            }
        """
        import random

        # Obtener estado y decisión base del sistema existente
        state = self.get_state_representation(market_data)

        # Calcular composite score para determinar tipo de acción
        composite_score = self._calculate_composite_score(market_data)
        confidence = self._calculate_confidence_from_score(composite_score)

        # =====================================================
        # 🛡️ LEVERAGE PROGRESIVO POR EXPERIENCIA
        # =====================================================
        # CRÍTICO: Limitar leverage basado en trades realizados
        # Protege contra pérdidas excesivas en sistema sin experiencia
        total_exp = self.total_trades

        if total_exp < 10:
            max_allowed_leverage = 3    # Principiante: máximo 3x
            exp_level = "NOVATO"
        elif total_exp < 30:
            max_allowed_leverage = 5    # Básico: máximo 5x
            exp_level = "BÁSICO"
        elif total_exp < 75:
            max_allowed_leverage = 8    # Intermedio: máximo 8x
            exp_level = "INTERMEDIO"
        elif total_exp < 150:
            max_allowed_leverage = 12   # Avanzado: máximo 12x
            exp_level = "AVANZADO"
        else:
            max_allowed_leverage = 17   # Experto: máximo 17x
            exp_level = "EXPERTO"

        logger.info(f"🤖 AUTONOMÍA: Score={composite_score:.2f}, Confidence={confidence:.1%}")
        logger.info(f"📊 Experiencia: {total_exp} trades → Nivel {exp_level} → Max leverage: {max_allowed_leverage}x")

        # =====================================================
        # DECISIÓN AUTÓNOMA: Basada en composite score
        # =====================================================

        # Score < 3.0 = SKIP (señal débil)
        if composite_score < 3.0:
            logger.info(f"🚫 SKIP: Score bajo ({composite_score:.2f} < 3.0)")
            return {
                'action': 'SKIP',
                'leverage': 1,
                'tp_percentages': [0.5, 1.0, 1.5],
                'position_size_pct': 0.0,
                'confidence': confidence,
                'chosen_action': 'SKIP',
                'composite_score': composite_score
            }

        # Score 3.0-4.5 = FUTURES_LOW (conservador)
        elif composite_score < 4.5:
            raw_leverage = random.randint(1, 3)
            leverage = min(raw_leverage, max_allowed_leverage)  # 🛡️ Aplicar límite
            tp_percentages = [
                random.uniform(0.25, 0.4),   # TP1: 0.25-0.4% (scalping rápido)
                random.uniform(0.6, 0.8),    # TP2: 0.6-0.8%
                random.uniform(1.0, 1.2)     # TP3: 1.0-1.2%
            ]
            position_size_pct = random.uniform(3.0, 4.0)
            chosen_action = 'FUTURES_LOW'

            logger.info(
                f"📊 FUTURES_LOW: Lev={leverage}x (max={max_allowed_leverage}x), "
                f"TPs={[f'{tp:.2f}%' for tp in tp_percentages]}, Size={position_size_pct:.1f}%"
            )

        # Score 4.5-6.0 = FUTURES_MEDIUM (moderado)
        elif composite_score < 6.0:
            raw_leverage = random.randint(5, 10)
            leverage = min(raw_leverage, max_allowed_leverage)  # 🛡️ Aplicar límite
            tp_percentages = [
                random.uniform(0.3, 0.5),    # TP1: 0.3-0.5%
                random.uniform(0.8, 1.2),    # TP2: 0.8-1.2%
                random.uniform(1.5, 2.0)     # TP3: 1.5-2.0%
            ]
            position_size_pct = random.uniform(4.0, 6.0)
            chosen_action = 'FUTURES_MEDIUM'

            logger.info(
                f"📈 FUTURES_MEDIUM: Lev={leverage}x (max={max_allowed_leverage}x), "
                f"TPs={[f'{tp:.2f}%' for tp in tp_percentages]}, Size={position_size_pct:.1f}%"
            )

        # Score >= 6.0 = FUTURES_HIGH (agresivo - solo con alta confianza)
        else:
            # Verificar confianza mínima para ser agresivo
            if confidence < 0.7:
                # Downgrade a MEDIUM si confianza insuficiente
                raw_leverage = random.randint(5, 10)
                leverage = min(raw_leverage, max_allowed_leverage)  # 🛡️ Aplicar límite
                tp_percentages = [
                    random.uniform(0.3, 0.5),
                    random.uniform(0.8, 1.2),
                    random.uniform(1.5, 2.0)
                ]
                position_size_pct = random.uniform(4.0, 6.0)
                chosen_action = 'FUTURES_MEDIUM'
                logger.info(f"⚠️ Downgrade a MEDIUM (confidence {confidence:.1%} < 70%), Lev={leverage}x (max={max_allowed_leverage}x)")
            else:
                raw_leverage = random.randint(12, 20)
                leverage = min(raw_leverage, max_allowed_leverage)  # 🛡️ Aplicar límite
                tp_percentages = [
                    random.uniform(0.4, 0.6),    # TP1: 0.4-0.6%
                    random.uniform(1.0, 1.5),    # TP2: 1.0-1.5%
                    random.uniform(1.8, 2.5)     # TP3: 1.8-2.5%
                ]
                position_size_pct = random.uniform(5.0, 8.0)
                chosen_action = 'FUTURES_HIGH'

                logger.info(
                    f"🚀 FUTURES_HIGH: Lev={leverage}x (max={max_allowed_leverage}x), "
                    f"TPs={[f'{tp:.2f}%' for tp in tp_percentages]}, Size={position_size_pct:.1f}%"
                )

        return {
            'action': 'OPEN',
            'leverage': leverage,
            'tp_percentages': tp_percentages,
            'position_size_pct': position_size_pct,
            'confidence': confidence,
            'chosen_action': chosen_action,
            'composite_score': composite_score
        }

    def _calculate_composite_score(self, market_data: Dict) -> float:
        """
        Calcula score compuesto de oportunidad (0-18)
        Mismo cálculo que decide_trade_action pero simplificado
        """
        score = 0.0
        side = market_data.get('side', 'NEUTRAL')

        # 1. Base confidence (0-5 puntos)
        signal_confidence = market_data.get('confidence', 50)
        score += (signal_confidence / 100.0) * 5.0

        # 2. Pattern detection (0-2 puntos)
        if market_data.get('pattern_detected', False):
            pattern_conf = market_data.get('pattern_confidence', 0)
            if pattern_conf > 0.8:
                score += 2.0
            elif pattern_conf > 0.6:
                score += 1.0

        # 3. Order flow boost (0-1.5 puntos)
        order_flow_ratio = market_data.get('order_flow_ratio', 1.0)
        if order_flow_ratio >= 1.5:
            score += 1.5
        elif order_flow_ratio >= 1.2:
            score += 0.75

        # 4. Fear & Greed (−1.5 a +2 puntos)
        fg_index = market_data.get('fear_greed_index', 50)
        if fg_index < 20:
            score += 2.0
        elif fg_index < 35:
            score += 1.0
        elif fg_index > 80:
            score -= 1.5
        elif fg_index > 70:
            score -= 0.5

        # 5. ML prediction alignment (0-2.5 puntos)
        ml_prediction = market_data.get('ml_prediction', 'HOLD')
        ml_confidence = market_data.get('ml_confidence', 0)
        if ml_confidence > 0.7:
            if (ml_prediction == 'BUY' and side == 'BUY') or \
               (ml_prediction == 'SELL' and side == 'SELL'):
                score += 2.5
            elif ml_prediction != side and ml_prediction != 'HOLD':
                score -= 1.5

        # 6. Multi-layer alignment (0-3 puntos)
        multi_layer = market_data.get('multi_layer_alignment', 0)
        if multi_layer > 0.8:
            score += 3.0
        elif multi_layer > 0.6:
            score += 1.5

        # 7. Orderbook pressure (0-1 punto)
        pressure = market_data.get('market_pressure', 'NEUTRAL')
        if (pressure == 'BUY_PRESSURE' and side == 'BUY') or \
           (pressure == 'SELL_PRESSURE' and side == 'SELL'):
            score += 1.0

        # 8. Sentiment alignment (0-1.5 puntos)
        sentiment = market_data.get('overall_sentiment', 'neutral')
        sentiment_strength = market_data.get('sentiment_strength', 0)
        if (sentiment == 'positive' and side == 'BUY') or \
           (sentiment == 'negative' and side == 'SELL'):
            if sentiment_strength > 0.7:
                score += 1.5
            else:
                score += 0.5

        return score

    def _calculate_confidence_from_score(self, composite_score: float) -> float:
        """
        Convierte composite score a confianza (0-1)
        Score 0 = 30% confianza, Score 10+ = 95% confianza
        """
        # Normalizar score a rango 0-1
        normalized = min(1.0, max(0.0, composite_score / 10.0))

        # Mapear a 0.3-0.95
        confidence = 0.3 + (normalized * 0.65)

        # Bonus por experiencia del agente
        if self.total_trades > 100:
            confidence = min(0.98, confidence * 1.05)

        return confidence
