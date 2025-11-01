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
        Convierte datos de mercado a representación de estado

        Args:
            market_data: Dict con indicadores técnicos, sentiment, regime, etc.

        Returns:
            Hash único del estado
        """
        # Extraer features clave
        rsi = market_data.get('rsi', 50)
        macd_signal = market_data.get('macd_signal', 'neutral')
        trend = market_data.get('trend', 'sideways')
        regime = market_data.get('regime', 'SIDEWAYS')
        sentiment = market_data.get('sentiment', 'neutral')
        volatility = market_data.get('volatility', 'medium')
        win_rate = market_data.get('win_rate', 50)
        drawdown = market_data.get('drawdown', 0)

        # Discretizar valores continuos
        rsi_bucket = 'oversold' if rsi < 35 else ('overbought' if rsi > 65 else 'neutral')
        wr_bucket = 'high' if win_rate > 55 else ('low' if win_rate < 45 else 'medium')
        dd_bucket = 'critical' if drawdown > 15 else ('high' if drawdown > 10 else 'safe')

        # Crear estado compuesto
        state = f"{rsi_bucket}|{macd_signal}|{trend}|{regime}|{sentiment}|{volatility}|{wr_bucket}|{dd_bucket}"

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
            'timestamp': datetime.now().isoformat()
        }

    def load_from_dict(self, data: Dict):
        """Carga agente desde diccionario"""
        self.learning_rate = data.get('learning_rate', self.learning_rate)
        self.discount_factor = data.get('discount_factor', self.discount_factor)
        self.exploration_rate = data.get('exploration_rate', self.exploration_rate)
        self.min_exploration = data.get('min_exploration', self.min_exploration)
        self.q_table = data.get('q_table', {})

        stats = data.get('statistics', {})
        self.total_trades = stats.get('total_trades', 0)
        self.successful_trades = stats.get('successful_trades', 0)
        self.total_reward = stats.get('total_reward', 0.0)

        self.episode_rewards = data.get('episode_rewards', [])

        logger.info(
            f"✅ RL Agent cargado: {self.total_trades} trades, "
            f"{self.get_success_rate():.1f}% win rate, "
            f"{len(self.q_table)} estados aprendidos"
        )
