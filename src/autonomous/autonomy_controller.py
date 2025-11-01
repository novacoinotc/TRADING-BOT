"""
Autonomy Controller - Cerebro del Sistema Autónomo
Control ABSOLUTO sobre todos los parámetros y decisiones
"""
import logging
import asyncio
from typing import Dict, Optional, Any, List
from datetime import datetime, timedelta
from pathlib import Path

from .rl_agent import RLAgent
from .parameter_optimizer import ParameterOptimizer
from .learning_persistence import LearningPersistence

logger = logging.getLogger(__name__)


class AutonomyController:
    """
    Controlador Autónomo con PODER ABSOLUTO

    - Controla TODOS los parámetros sin limitaciones
    - Decide cuándo y qué cambiar
    - Aprende de cada trade
    - Optimiza configuración continuamente
    - Persiste inteligencia aprendida
    - Notifica cada cambio a Telegram

    La IA tiene control TOTAL - sin restricciones humanas
    """

    def __init__(
        self,
        telegram_notifier=None,
        auto_save_interval_minutes: int = 30,
        optimization_check_interval_hours: float = 2.0,
        min_trades_before_optimization: int = 20
    ):
        """
        Args:
            telegram_notifier: Instancia de TelegramNotifier para notificaciones
            auto_save_interval_minutes: Intervalo para auto-guardar inteligencia
            optimization_check_interval_hours: Cada cuántas horas considerar optimizar parámetros
            min_trades_before_optimization: Mínimo de trades antes de optimizar
        """
        self.telegram_notifier = telegram_notifier
        self.auto_save_interval = auto_save_interval_minutes
        self.optimization_interval = optimization_check_interval_hours
        self.min_trades_before_opt = min_trades_before_optimization

        # Componentes principales
        self.rl_agent = RLAgent(
            learning_rate=0.1,
            discount_factor=0.95,
            exploration_rate=0.3,
            exploration_decay=0.995,
            min_exploration=0.05
        )

        self.parameter_optimizer = ParameterOptimizer()

        self.persistence = LearningPersistence(storage_dir="data/autonomous")

        # Estado del sistema
        self.active = False
        self.current_parameters: Dict[str, Any] = {}
        self.performance_history: List[Dict] = []
        self.last_optimization_time = datetime.now()
        self.last_save_time = datetime.now()
        self.total_trades_processed = 0
        self.total_parameter_changes = 0

        # Control de decisiones
        self.decision_mode = "AUTONOMOUS"  # AUTONOMOUS, CONSERVATIVE, AGGRESSIVE

        logger.info("🤖 AUTONOMY CONTROLLER INICIALIZADO - MODO: CONTROL ABSOLUTO")
        logger.info(f"   Auto-save: cada {auto_save_interval} min")
        logger.info(f"   Optimization check: cada {optimization_check_interval} horas")
        logger.info(f"   Min trades antes de optimizar: {min_trades_before_optimization}")

    async def initialize(self):
        """
        Inicializa el controlador autónomo
        - Intenta cargar inteligencia previa
        - Envía notificación de inicio
        """
        logger.info("🚀 Inicializando Sistema Autónomo...")

        # Intentar cargar inteligencia guardada
        loaded_state = self.persistence.load_full_state()

        if loaded_state:
            await self._restore_from_state(loaded_state)
            await self._notify_telegram(
                "🧠 **Sistema Autónomo Iniciado**\n\n"
                "✅ Inteligencia previa CARGADA exitosamente\n"
                f"📊 Experiencia: {self.rl_agent.total_trades} trades aprendidos\n"
                f"🎯 Optimización: {self.parameter_optimizer.total_trials} trials completados\n"
                f"🏆 Mejor configuración restaurada\n\n"
                "El bot continuará aprendiendo desde donde se quedó ✨"
            )
        else:
            await self._notify_telegram(
                "🤖 **Sistema Autónomo Iniciado**\n\n"
                "🆕 Primera ejecución - iniciando aprendizaje desde cero\n"
                "🧠 RL Agent: Activo\n"
                "🎯 Parameter Optimizer: Activo\n"
                "💾 Auto-save: Habilitado\n\n"
                "El bot aprenderá y se optimizará de forma completamente autónoma 🚀"
            )

        self.active = True
        logger.info("✅ Sistema Autónomo ACTIVO - Control total habilitado")

    async def _restore_from_state(self, state: Dict):
        """Restaura estado completo desde persistencia"""
        try:
            # Restaurar RL Agent
            if 'rl_agent' in state:
                self.rl_agent.load_from_dict(state['rl_agent'])

            # Restaurar Parameter Optimizer
            if 'parameter_optimizer' in state:
                self.parameter_optimizer.load_from_dict(state['parameter_optimizer'])

            # Restaurar historial de performance
            if 'performance_history' in state:
                self.performance_history = state['performance_history']

            # Restaurar parámetros actuales
            if 'metadata' in state and 'current_parameters' in state['metadata']:
                self.current_parameters = state['metadata']['current_parameters']

            logger.info("✅ Estado completo restaurado exitosamente")

        except Exception as e:
            logger.error(f"❌ Error restaurando estado: {e}", exc_info=True)

    async def process_trade_outcome(
        self,
        trade_data: Dict,
        market_state: Dict,
        portfolio_metrics: Dict
    ):
        """
        Procesa resultado de un trade y aprende de él

        Args:
            trade_data: Datos del trade (pair, action, profit_pct, etc.)
            market_state: Estado del mercado (indicadores, regime, sentiment, etc.)
            portfolio_metrics: Métricas del portfolio (win_rate, roi, etc.)
        """
        if not self.active:
            return

        # Calcular reward basado en resultado del trade
        reward = self._calculate_reward(trade_data, portfolio_metrics)

        # Convertir estado de mercado a representación para RL
        state = self.rl_agent.get_state_representation(market_state)

        # RL Agent aprende del trade
        self.rl_agent.learn_from_trade(reward=reward, next_state=state, done=False)

        # Experience Replay periódico
        if self.rl_agent.total_trades % 10 == 0:
            self.rl_agent.replay_experience(batch_size=32)

        # Guardar en historial
        self.performance_history.append({
            'timestamp': datetime.now().isoformat(),
            'trade': trade_data,
            'market_state': market_state,
            'portfolio_metrics': portfolio_metrics,
            'reward': reward
        })

        self.total_trades_processed += 1

        # Notificar aprendizaje importante
        if reward > 2.0:  # Gran ganancia
            await self._notify_telegram(
                f"🎉 **Trade Exitoso Aprendido**\n\n"
                f"Par: {trade_data.get('pair', 'N/A')}\n"
                f"Profit: {trade_data.get('profit_pct', 0):.2f}%\n"
                f"Reward: {reward:.3f}\n"
                f"La IA aprenderá de este éxito ✨"
            )
        elif reward < -2.0:  # Gran pérdida
            await self._notify_telegram(
                f"📚 **Trade Perdedor Analizado**\n\n"
                f"Par: {trade_data.get('pair', 'N/A')}\n"
                f"Loss: {trade_data.get('profit_pct', 0):.2f}%\n"
                f"Reward: {reward:.3f}\n"
                f"La IA ajustará estrategia para evitar repetir ⚙️"
            )

        # Verificar si es momento de optimizar parámetros
        await self._check_and_optimize(portfolio_metrics)

        # Auto-save periódico
        await self._auto_save_if_needed()

    def _calculate_reward(self, trade_data: Dict, portfolio_metrics: Dict) -> float:
        """
        Calcula reward para el RL Agent

        Combina:
        - Profit/loss del trade
        - Impacto en métricas del portfolio
        - Risk-adjusted return
        """
        profit_pct = trade_data.get('profit_pct', 0)

        # Base reward: profit/loss
        reward = profit_pct

        # Bonus/penalty por métricas de portfolio
        win_rate = portfolio_metrics.get('win_rate', 50)
        if win_rate > 55:
            reward *= 1.2  # Bonus si win rate es bueno
        elif win_rate < 45:
            reward *= 0.8  # Penalty si win rate es malo

        # Penalty por drawdown alto
        drawdown = abs(portfolio_metrics.get('max_drawdown', 0))
        if drawdown > 15:
            reward -= 0.5  # Penalty adicional

        # Bonus por Sharpe ratio alto
        sharpe = portfolio_metrics.get('sharpe_ratio', 0)
        if sharpe > 1.5:
            reward += 0.3

        return reward

    async def _check_and_optimize(self, portfolio_metrics: Dict):
        """
        Verifica si es momento de optimizar parámetros
        Decisión AUTÓNOMA - la IA decide cuándo optimizar
        """
        # Verificar condiciones para optimizar
        elapsed_hours = (datetime.now() - self.last_optimization_time).total_seconds() / 3600

        should_optimize = False
        reason = ""

        # Condición 1: Ha pasado suficiente tiempo
        if elapsed_hours >= self.optimization_interval:
            should_optimize = True
            reason = f"Intervalo de tiempo alcanzado ({elapsed_hours:.1f} horas)"

        # Condición 2: Suficientes trades procesados
        if self.total_trades_processed >= self.min_trades_before_opt:
            should_optimize = True
            reason = f"Suficientes trades procesados ({self.total_trades_processed})"

        # Condición 3: Performance muy mala (intervención urgente)
        if portfolio_metrics.get('win_rate', 50) < 35:
            should_optimize = True
            reason = f"⚠️ Win rate crítico ({portfolio_metrics['win_rate']:.1f}%) - optimización urgente"

        if portfolio_metrics.get('roi', 0) < -10:
            should_optimize = True
            reason = f"⚠️ ROI crítico ({portfolio_metrics['roi']:.1f}%) - optimización urgente"

        # Condición 4: Performance muy buena (aprovechar momentum)
        if portfolio_metrics.get('win_rate', 50) > 65:
            should_optimize = True
            reason = f"🎯 Win rate excelente ({portfolio_metrics['win_rate']:.1f}%) - optimizar para maximizar"

        if should_optimize:
            await self._optimize_parameters(portfolio_metrics, reason)

    async def _optimize_parameters(self, portfolio_metrics: Dict, reason: str):
        """
        Optimiza parámetros de forma autónoma
        LA IA DECIDE QUÉ CAMBIAR - CONTROL ABSOLUTO
        """
        logger.info(f"🎯 INICIANDO OPTIMIZACIÓN AUTÓNOMA: {reason}")

        # Calcular exploration factor dinámico
        # Más exploración si performance es mala, menos si es buena
        win_rate = portfolio_metrics.get('win_rate', 50)
        if win_rate < 40:
            exploration_factor = 0.5  # Mucha exploración
        elif win_rate > 60:
            exploration_factor = 0.2  # Poca exploración (ya va bien)
        else:
            exploration_factor = 0.3  # Exploración moderada

        # Obtener sugerencias de cambios
        suggestions = self.parameter_optimizer.suggest_parameter_changes(
            current_performance=portfolio_metrics,
            exploration_factor=exploration_factor
        )

        new_config = suggestions['config']
        changes = suggestions['changes']
        strategy = suggestions['strategy']
        change_reason = suggestions['reason']

        # Aplicar cambios
        self.current_parameters.update(new_config)
        self.parameter_optimizer.current_config = new_config

        # Registrar trial
        self.parameter_optimizer.record_trial_result(new_config, portfolio_metrics)

        self.last_optimization_time = datetime.now()
        self.total_parameter_changes += 1

        # NOTIFICAR CAMBIOS A TELEGRAM
        await self._notify_parameter_changes(changes, strategy, change_reason, portfolio_metrics)

        logger.info(f"✅ Optimización completada: {len(changes)} parámetros modificados")

    async def _notify_parameter_changes(
        self,
        changes: List[Dict],
        strategy: str,
        reason: str,
        metrics: Dict
    ):
        """
        Notifica cambios de parámetros a Telegram
        Cada modificación que la IA hace es notificada
        """
        if not self.telegram_notifier:
            return

        # Construir mensaje
        message_parts = [
            "🤖 **IA MODIFICÓ PARÁMETROS AUTÓNOMAMENTE**",
            "",
            f"🎯 Estrategia: {strategy}",
            f"📊 Performance Actual:",
            f"   • Win Rate: {metrics.get('win_rate', 0):.1f}%",
            f"   • ROI: {metrics.get('roi', 0):.2f}%",
            f"   • Sharpe: {metrics.get('sharpe_ratio', 0):.2f}",
            "",
            "💡 Razón del cambio:",
            reason,
            "",
            f"🔧 Parámetros Modificados ({len(changes)}):"
        ]

        # Listar cambios (máximo 10 para no saturar mensaje)
        for i, change in enumerate(changes[:10], 1):
            param = change['parameter']
            old = change['old_value']
            new = change['new_value']
            pct = change['change_pct']

            direction = "📈" if new > old else "📉"
            message_parts.append(
                f"{i}. {direction} **{param}**: {old} → {new} "
                f"({pct:+.1f}%)" if old else f"{i}. ✨ **{param}**: → {new}"
            )

        if len(changes) > 10:
            message_parts.append(f"   ... y {len(changes) - 10} más")

        message_parts.extend([
            "",
            f"🧠 Total cambios realizados: {self.total_parameter_changes}",
            f"📚 Trades procesados: {self.total_trades_processed}",
            "",
            "La IA continúa aprendiendo y optimizando... 🚀"
        ])

        message = "\n".join(message_parts)

        try:
            await self.telegram_notifier.send_status_message(message)
        except Exception as e:
            logger.warning(f"No se pudo enviar notificación: {e}")

    async def _auto_save_if_needed(self):
        """Auto-guarda inteligencia si ha pasado suficiente tiempo"""
        elapsed_minutes = (datetime.now() - self.last_save_time).total_seconds() / 60

        if elapsed_minutes >= self.auto_save_interval:
            await self.save_intelligence()
            self.last_save_time = datetime.now()

    async def save_intelligence(self):
        """Guarda toda la inteligencia aprendida"""
        logger.info("💾 Guardando inteligencia aprendida...")

        rl_state = self.rl_agent.save_to_dict()
        optimizer_state = self.parameter_optimizer.save_to_dict()

        performance_summary = {
            'total_trades': len(self.performance_history),
            'recent_performance': self.performance_history[-100:] if self.performance_history else []
        }

        metadata = {
            'current_parameters': self.current_parameters,
            'total_trades_processed': self.total_trades_processed,
            'total_parameter_changes': self.total_parameter_changes,
            'last_optimization': self.last_optimization_time.isoformat(),
            'decision_mode': self.decision_mode
        }

        success = self.persistence.save_full_state(
            rl_agent_state=rl_state,
            optimizer_state=optimizer_state,
            performance_history=performance_summary,
            metadata=metadata
        )

        if success:
            logger.info("✅ Inteligencia guardada exitosamente")

            # Exportar para fácil importación
            export_path = self.persistence.export_for_import()
            if export_path:
                logger.info(f"📤 Archivo de exportación: {export_path}")
        else:
            logger.error("❌ Error guardando inteligencia")

    def get_current_parameters(self) -> Dict[str, Any]:
        """Retorna parámetros actuales (para aplicar en el bot)"""
        return self.current_parameters.copy()

    def get_statistics(self) -> Dict:
        """Retorna estadísticas completas del sistema autónomo"""
        rl_stats = self.rl_agent.get_statistics()
        opt_stats = self.parameter_optimizer.get_optimization_statistics()

        return {
            'active': self.active,
            'decision_mode': self.decision_mode,
            'total_trades_processed': self.total_trades_processed,
            'total_parameter_changes': self.total_parameter_changes,
            'rl_agent': rl_stats,
            'parameter_optimizer': opt_stats,
            'last_optimization': self.last_optimization_time.isoformat(),
            'last_save': self.last_save_time.isoformat(),
            'current_parameters_count': len(self.current_parameters)
        }

    async def _notify_telegram(self, message: str):
        """Envía notificación a Telegram"""
        if self.telegram_notifier:
            try:
                await self.telegram_notifier.send_status_message(message)
            except Exception as e:
                logger.warning(f"No se pudo enviar notificación: {e}")

    async def shutdown(self):
        """Apaga sistema autónomo y guarda estado final"""
        logger.info("🛑 Apagando Sistema Autónomo...")

        self.active = False

        # Guardar inteligencia final
        await self.save_intelligence()

        await self._notify_telegram(
            "🛑 **Sistema Autónomo Apagado**\n\n"
            f"📊 Resumen de sesión:\n"
            f"   • Trades procesados: {self.total_trades_processed}\n"
            f"   • Parámetros modificados: {self.total_parameter_changes} veces\n"
            f"   • Estados aprendidos: {len(self.rl_agent.q_table)}\n"
            f"   • Trials de optimización: {self.parameter_optimizer.total_trials}\n\n"
            "✅ Inteligencia guardada - lista para próximo deploy"
        )

        logger.info("✅ Sistema Autónomo apagado - Inteligencia preservada")
