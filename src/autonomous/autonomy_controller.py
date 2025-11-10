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
from .git_backup import GitBackup

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

        # Histórico de cambios con razonamiento (para memoria histórica)
        self.change_history: List[Dict] = []

        # Control de decisiones
        self.decision_mode = "AUTONOMOUS"  # AUTONOMOUS, CONSERVATIVE, AGGRESSIVE

        # Git Backup System
        self.git_backup = GitBackup(
            telegram_notifier=telegram_notifier,
            backup_interval_hours=24.0,
            backup_dir="data/autonomous"
        )

        logger.info("🤖 AUTONOMY CONTROLLER INICIALIZADO - MODO: CONTROL ABSOLUTO")
        logger.info(f"   Auto-save: cada {self.auto_save_interval} min")
        logger.info(f"   Optimization check: cada {self.optimization_interval} horas")
        logger.info(f"   Min trades antes de optimizar: {self.min_trades_before_opt}")

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

        # Iniciar Git Auto-Backup
        await self.git_backup.start_auto_backup()

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

            # Restaurar histórico de cambios
            if 'change_history' in state:
                self.change_history = state['change_history']
                logger.info(f"📚 {len(self.change_history)} cambios históricos restaurados")

            logger.info("✅ Estado completo restaurado exitosamente")

        except Exception as e:
            logger.error(f"❌ Error restaurando estado: {e}", exc_info=True)

    async def evaluate_trade_opportunity(
        self,
        pair: str,
        signal: Dict,
        market_state: Dict,
        portfolio_metrics: Dict
    ) -> Dict:
        """
        Evalúa si abrir un trade usando RL Agent ANTES de ejecutarlo

        Args:
            pair: Par de trading
            signal: Señal generada (BUY/SELL/HOLD)
            market_state: Estado del mercado actual
            portfolio_metrics: Métricas del portfolio

        Returns:
            Dict con decisión del RL Agent
        """
        if not self.active:
            # Si autonomy no está activo, permitir el trade
            return {
                'should_trade': True,
                'action': 'OPEN',
                'position_size_multiplier': 1.0,
                'confidence': 1.0,
                'chosen_action': 'AUTONOMOUS_DISABLED'
            }

        try:
            # Determinar side de la señal (BUY/SELL)
            signal_action = signal.get('action', 'HOLD')
            side = 'BUY' if signal_action == 'BUY' else ('SELL' if signal_action == 'SELL' else 'NEUTRAL')

            # Extraer regime strength del regime_data si está disponible
            regime = market_state.get('regime', 'SIDEWAYS')
            regime_strength = market_state.get('regime_strength', 'MEDIUM')

            # Si no hay regime_strength pero hay volatilidad, derivarlo
            if regime_strength == 'MEDIUM':
                volatility = market_state.get('volatility', 'medium')
                if volatility == 'high':
                    regime_strength = 'HIGH'
                elif volatility == 'low':
                    regime_strength = 'LOW'

            # Construir datos de mercado para RL Agent
            market_data = {
                'pair': pair,
                'side': side,
                'rsi': market_state.get('rsi', 50),
                'regime': regime,
                'regime_strength': regime_strength,
                'orderbook': market_state.get('orderbook', 'NEUTRAL'),
                'confidence': signal.get('confidence', 50)
            }

            # RL Agent decide si abrir trade
            decision = self.rl_agent.decide_trade_action(market_data)

            logger.info(
                f"🤖 RL Evaluation para {pair}: "
                f"{decision['chosen_action']} | "
                f"Trade: {'✅' if decision['should_trade'] else '❌'} | "
                f"Size: {decision['position_size_multiplier']:.1f}x"
            )

            return decision

        except Exception as e:
            logger.error(f"❌ Error en evaluate_trade_opportunity: {e}", exc_info=True)
            # En caso de error, permitir el trade
            return {
                'should_trade': True,
                'action': 'OPEN',
                'position_size_multiplier': 1.0,
                'confidence': 1.0,
                'chosen_action': 'ERROR_FALLBACK'
            }

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

        # GUARDAR EN HISTÓRICO DE CAMBIOS (memoria histórica para futuro)
        change_record = {
            'timestamp': datetime.now().isoformat(),
            'change_number': self.total_parameter_changes,
            'trigger_reason': reason,
            'strategy': strategy,
            'performance_before': {
                'win_rate': portfolio_metrics.get('win_rate', 0),
                'roi': portfolio_metrics.get('roi', 0),
                'sharpe_ratio': portfolio_metrics.get('sharpe_ratio', 0),
                'max_drawdown': portfolio_metrics.get('max_drawdown', 0)
            },
            'parameters_changed': [
                {
                    'parameter': c['parameter'],
                    'old_value': c['old_value'],
                    'new_value': c['new_value'],
                    'change_pct': c['change_pct']
                }
                for c in changes
            ],
            'reasoning': change_reason,
            'exploration_factor': exploration_factor,
            'total_trades_at_change': self.total_trades_processed
        }
        self.change_history.append(change_record)

        # Mantener solo últimos 100 cambios (para no saturar)
        if len(self.change_history) > 100:
            self.change_history = self.change_history[-100:]

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
        Notifica cambios de parámetros a Telegram (versión resumida)
        Cada modificación que la IA hace es notificada de forma concisa
        """
        if not self.telegram_notifier:
            return

        # VERSIÓN RESUMIDA (más concisa para no saturar)
        message_parts = [
            f"🤖 <b>IA realizó {len(changes)} cambios</b> ({strategy})",
            "",
            f"📊 Performance: Win Rate {metrics.get('win_rate', 0):.1f}% | ROI {metrics.get('roi', 0):+.2f}%",
        ]

        # Razón resumida (max 120 caracteres)
        reason_lines = reason.split('\n')
        brief_reason = reason_lines[0] if reason_lines else reason
        if len(brief_reason) > 120:
            brief_reason = brief_reason[:117] + "..."
        message_parts.append(f"💡 {brief_reason}")

        # Top 3-5 cambios más significativos solamente
        sorted_changes = sorted(changes, key=lambda x: abs(x.get('change_pct', 0)), reverse=True)
        top_changes = sorted_changes[:5]

        if top_changes:
            message_parts.append("")
            message_parts.append("🔧 <b>Principales cambios:</b>")
            for change in top_changes:
                param = change['parameter']
                new = change['new_value']
                direction = "📈" if change.get('change_pct', 0) > 0 else "📉"

                # Nombre más corto para parámetros comunes
                param_short = param.replace('_THRESHOLD', '').replace('_PCT', '').replace('_PERIOD', '')
                message_parts.append(f"  {direction} {param_short}: {new}")

        if len(changes) > 5:
            message_parts.append(f"  ... +{len(changes) - 5} parámetros más")

        message_parts.extend([
            "",
            f"🧠 Total: {self.total_parameter_changes} cambios | {self.total_trades_processed} trades"
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

    async def save_intelligence(self) -> str:
        """
        Guarda toda la inteligencia aprendida

        Returns:
            Path al archivo de exportación, o string vacío si falló
        """
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
            change_history=self.change_history,  # Histórico de cambios con razonamiento
            metadata=metadata
        )

        if success:
            logger.info("✅ Inteligencia guardada exitosamente")

            # Exportar para fácil importación
            export_path = self.persistence.export_for_import()
            if export_path:
                logger.info(f"📤 Archivo de exportación: {export_path}")
                return export_path
        else:
            logger.error("❌ Error guardando inteligencia")

        return ""

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

        # Detener Git Auto-Backup
        await self.git_backup.stop_auto_backup()

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

    async def manual_export(self) -> tuple[bool, str]:
        """
        Export manual de inteligencia (llamado por comando de Telegram)

        Returns:
            Tupla (success, export_file_path)
            - success: True si backup a Git fue exitoso
            - export_file_path: Path al archivo .json exportado
        """
        logger.info("📤 Export manual solicitado...")

        # Guardar inteligencia primero y obtener path del export
        export_path = await self.save_intelligence()

        # Realizar backup a Git
        success = await self.git_backup.perform_backup(manual=True)

        return success, export_path

    async def manual_import(self, file_path: str, merge: bool = False) -> bool:
        """
        Import manual de inteligencia desde archivo (llamado por comando de Telegram)

        Args:
            file_path: Path al archivo .json con la inteligencia
            merge: Si True, combina con datos existentes. Si False, reemplaza

        Returns:
            True si import fue exitoso
        """
        logger.info(f"📥 Import manual solicitado desde: {file_path} (merge={merge})")

        # Importar el archivo
        success = self.persistence.import_from_file(file_path)

        if not success:
            logger.error("❌ Falló la importación del archivo")
            return False

        # Recargar la inteligencia importada
        loaded = self.persistence.load_full_state()

        if not loaded:
            logger.error("❌ No se pudo cargar la inteligencia importada")
            return False

        # Restaurar todo el estado
        try:
            # Restaurar RL Agent (con o sin merge)
            self.rl_agent.load_from_dict(loaded['rl_agent'], merge=merge)
            logger.info("✅ RL Agent restaurado")

            # Restaurar Parameter Optimizer (con o sin merge)
            self.parameter_optimizer.load_from_dict(loaded['parameter_optimizer'], merge=merge)
            logger.info("✅ Parameter Optimizer restaurado")

            # Restaurar historial de cambios
            if merge:
                # En merge, agregar cambios históricos a los existentes
                imported_changes = loaded.get('change_history', [])
                self.change_history.extend(imported_changes)
                # Mantener solo últimos 100
                if len(self.change_history) > 100:
                    self.change_history = self.change_history[-100:]
                logger.info(f"✅ {len(imported_changes)} cambios históricos agregados (total: {len(self.change_history)})")
            else:
                # En replace, reemplazar completamente
                self.change_history = loaded.get('change_history', [])
                logger.info(f"✅ Histórico de cambios restaurado: {len(self.change_history)} cambios")

            # Restaurar metadata
            metadata = loaded.get('metadata', {})
            if metadata:
                if merge:
                    # En merge, solo actualizar si no existen
                    if not self.current_parameters:
                        self.current_parameters = metadata.get('current_parameters', {})
                    self.total_trades_processed += metadata.get('total_trades_processed', 0)
                    self.total_parameter_changes += metadata.get('total_parameter_changes', 0)
                    logger.info(f"✅ Metadata acumulada (trades: {self.total_trades_processed})")
                else:
                    # En replace, reemplazar completamente
                    self.current_parameters = metadata.get('current_parameters', {})
                    self.total_trades_processed = metadata.get('total_trades_processed', 0)
                    self.total_parameter_changes = metadata.get('total_parameter_changes', 0)
                    self.decision_mode = metadata.get('decision_mode', 'BALANCED')

            # Restaurar performance history
            perf_history = loaded.get('performance_history', {})
            if perf_history.get('recent_performance'):
                if merge:
                    # En merge, agregar a la historia existente
                    self.performance_history.extend(perf_history['recent_performance'])
                    # Mantener solo últimos 100
                    if len(self.performance_history) > 100:
                        self.performance_history = self.performance_history[-100:]
                else:
                    # En replace, reemplazar
                    self.performance_history = perf_history['recent_performance']

            logger.info(f"✅ Inteligencia importada y {'combinada' if merge else 'restaurada'} completamente")
            return True

        except Exception as e:
            logger.error(f"❌ Error restaurando estado importado: {e}", exc_info=True)
            return False
