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

        # Contador global de trades (nunca se resetea)
        self.total_trades_all_time = 0

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

    def _calculate_max_leverage(self) -> int:
        """
        Calcula el máximo leverage permitido basado en experiencia

        Límites escalonados:
        - 0-50 trades: máximo 5x
        - 50-100 trades: máximo 8x
        - 100-150 trades: máximo 10x
        - 150-500 trades: máximo 15x
        - 500+ trades: máximo 20x

        Returns:
            Leverage máximo permitido (1-20x)
        """
        if self.total_trades_all_time < 50:
            return 5
        elif self.total_trades_all_time < 100:
            return 8
        elif self.total_trades_all_time < 150:
            return 10
        elif self.total_trades_all_time < 500:
            return 15
        else:
            return 20

    async def _restore_from_state(self, state: Dict):
        """Restaura estado completo desde persistencia (robusto con errores)"""
        logger.debug("📥 Restaurando estado desde persistencia...")

        # Restaurar RL Agent
        try:
            if 'rl_agent' in state:
                self.rl_agent.load_from_dict(state['rl_agent'])
                logger.debug("  ✅ RL Agent restaurado")
            else:
                logger.warning("⚠️ No se encontró 'rl_agent' en el estado guardado")
        except Exception as e:
            logger.error(f"❌ Error restaurando RL Agent: {e}", exc_info=True)

        # Restaurar Parameter Optimizer
        try:
            if 'parameter_optimizer' in state:
                self.parameter_optimizer.load_from_dict(state['parameter_optimizer'])
                logger.debug("  ✅ Parameter Optimizer restaurado")
        except Exception as e:
            logger.warning(f"⚠️ Error restaurando Parameter Optimizer: {e}")

        # Restaurar historial de performance
        try:
            if 'performance_history' in state:
                perf = state['performance_history']
                if isinstance(perf, dict) and 'recent_performance' in perf:
                    self.performance_history = perf['recent_performance']
                elif isinstance(perf, list):
                    self.performance_history = perf
                logger.debug("  ✅ Performance history restaurado")
        except Exception as e:
            logger.warning(f"⚠️ Error restaurando performance history: {e}")

        # Restaurar parámetros actuales y metadata
        try:
            if 'metadata' in state:
                metadata = state['metadata']
                if 'current_parameters' in metadata:
                    self.current_parameters = metadata['current_parameters']

                # Restaurar total_trades_all_time con múltiples fallbacks
                if 'total_trades_all_time' in metadata:
                    self.total_trades_all_time = metadata['total_trades_all_time']
                    logger.debug(f"  ✅ Total trades all time: {self.total_trades_all_time}")
                else:
                    # Fallback 1: usar total_experience_trades del RL agent
                    rl_data = state.get('rl_agent', {})
                    self.total_trades_all_time = rl_data.get('total_experience_trades')
                    if self.total_trades_all_time is not None:
                        logger.debug(f"  ⚠️ total_trades_all_time no en metadata, usando RL agent: {self.total_trades_all_time}")
                    else:
                        # Fallback 2: usar total_trades del RL agent cargado
                        if hasattr(self, 'rl_agent'):
                            self.total_trades_all_time = self.rl_agent.total_trades
                            logger.debug(f"  ⚠️ Usando total_trades del RL agent cargado: {self.total_trades_all_time}")

                logger.debug("  ✅ Metadata restaurada")
        except Exception as e:
            logger.warning(f"⚠️ Error restaurando metadata: {e}")

        # Restaurar histórico de cambios
        try:
            if 'change_history' in state:
                self.change_history = state['change_history']
                logger.info(f"  ✅ {len(self.change_history)} cambios históricos restaurados")
        except Exception as e:
            logger.warning(f"⚠️ Error restaurando change history: {e}")

        logger.info("✅ Estado completo restaurado exitosamente")

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

            # Construir datos de mercado para RL Agent - INTEGRACIÓN COMPLETA DE LOS 16 SERVICIOS
            market_data = {
                # Básicos
                'pair': pair,
                'side': side,
                'rsi': market_state.get('rsi', 50),
                'regime': regime,
                'regime_strength': regime_strength,
                'orderbook': market_state.get('orderbook', 'NEUTRAL'),
                'confidence': signal.get('confidence', 50),
                'total_trades': self.total_trades_all_time,  # Para tier de experiencia

                # 3. CryptoPanic GROWTH API
                'cryptopanic_sentiment': market_state.get('cryptopanic_sentiment', 'neutral'),
                'news_volume': market_state.get('news_volume', 0),
                'news_importance': market_state.get('news_importance', 0),
                'pre_pump_score': market_state.get('pre_pump_score', 0),

                # 4. Fear & Greed Index
                'fear_greed_index': market_state.get('fear_greed_index', 50),
                'fear_greed_label': market_state.get('fear_greed_label', 'neutral'),

                # 5. Sentiment Analysis
                'overall_sentiment': market_state.get('overall_sentiment', 'neutral'),
                'sentiment_strength': market_state.get('sentiment_strength', 0),
                'social_buzz': market_state.get('social_buzz', 0),

                # 6. News-Triggered Trading
                'news_triggered': market_state.get('news_triggered', False),
                'news_trigger_confidence': market_state.get('news_trigger_confidence', 0),

                # 7. Multi-Layer Confidence System
                'confidence_5m': signal.get('confidence_5m', 0),
                'confidence_1h': signal.get('confidence_1h', 0),
                'confidence_4h': signal.get('confidence_4h', 0),
                'confidence_1d': signal.get('confidence_1d', 0),
                'multi_layer_alignment': signal.get('multi_layer_alignment', 0),

                # 8. ML System (Predictor)
                'ml_prediction': signal.get('ml_prediction', 'HOLD'),
                'ml_confidence': signal.get('ml_confidence', 0),
                'ml_features_importance': signal.get('ml_features_importance', {}),

                # 12. Order Book Analyzer
                'orderbook_imbalance': market_state.get('orderbook_imbalance', 0),
                'bid_ask_spread': market_state.get('bid_ask_spread', 0),
                'orderbook_depth_score': market_state.get('orderbook_depth_score', 0),
                'market_pressure': market_state.get('market_pressure', 'NEUTRAL'),

                # 13. Market Regime Detector
                'regime_confidence': market_state.get('regime_confidence', 0),
                'trend_strength': market_state.get('trend_strength', 0),
                'volatility_regime': market_state.get('volatility_regime', 'NORMAL'),

                # 14. Dynamic TP Manager
                'dynamic_tp_multiplier': signal.get('dynamic_tp_multiplier', 1.0),
                'volatility_adjusted': signal.get('volatility_adjusted', False)
            }

            # Calcular max leverage permitido
            max_leverage = self._calculate_max_leverage()

            # RL Agent decide si abrir trade (pasando max_leverage)
            decision = self.rl_agent.decide_trade_action(market_data, max_leverage=max_leverage)

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

        # Incrementar contador global de trades (nunca se resetea)
        self.total_trades_all_time += 1

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

        # Log de desbloqueo de leverage
        max_leverage = self._calculate_max_leverage()
        if self.total_trades_all_time in [50, 100, 150, 500]:
            await self._notify_telegram(
                f"🎉 **Nuevo Leverage Desbloqueado**\n\n"
                f"Total trades: {self.total_trades_all_time}\n"
                f"Max leverage: {max_leverage}x\n"
                f"¡El RL Agent ahora puede usar futuros con mayor leverage!"
            )

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

        Para SPOT:
        - Profit/loss del trade con ajustes por métricas del portfolio

        Para FUTURES:
        - Profit/loss * leverage (simple, sin penalizaciones artificiales)
        - El RL debe aprender por sí mismo que liquidarse es malo
        """
        profit_pct = trade_data.get('profit_pct', 0)
        trade_type = trade_data.get('trade_type', 'SPOT')
        leverage = trade_data.get('leverage', 1)
        liquidated = trade_data.get('liquidated', False)

        # FUTUROS: reward simple = profit_pct * leverage
        if trade_type == 'FUTURES':
            # Para liquidaciones, el profit_pct ya es -100%, así que el reward es muy negativo
            reward = profit_pct * leverage

            # NO agregamos penalizaciones artificiales - que aprenda de la experiencia real
            logger.debug(f"Futures reward: {profit_pct:.2f}% * {leverage}x = {reward:.2f}")

        else:
            # SPOT: Base reward con ajustes
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
            'total_trades_all_time': self.total_trades_all_time,  # Contador global nunca se resetea
            'current_max_leverage': self._calculate_max_leverage(),
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

    async def manual_import(self, file_path: str, merge: bool = False, force: bool = False) -> bool:
        """
        Import manual de inteligencia desde archivo (llamado por comando de Telegram)

        Args:
            file_path: Path al archivo .json con la inteligencia
            merge: Si True, combina con datos existentes. Si False, reemplaza
            force: Si True, ignora validación de checksum (para archivos editados)

        Returns:
            True si import fue exitoso
        """
        logger.info(f"📥 Import manual solicitado desde: {file_path} (merge={merge}, force={force})")

        # Importar el archivo (con force si se especifica)
        logger.debug(f"📥 Paso 1/3: Importando archivo con force={force}")
        success = self.persistence.import_from_file(file_path, force=force)

        if not success:
            logger.error("❌ Falló la importación del archivo")
            return False

        logger.info("✅ Paso 1/3 completado: Archivo importado exitosamente")

        # Recargar la inteligencia importada (con force para ignorar checksum)
        logger.debug(f"📥 Paso 2/3: Cargando estado con force={force}")
        loaded = self.persistence.load_full_state(force=force)

        if not loaded:
            logger.error("❌ No se pudo cargar la inteligencia importada")
            return False

        logger.info("✅ Paso 2/3 completado: Estado cargado exitosamente")

        # Restaurar todo el estado
        logger.debug("📥 Paso 3/3: Restaurando componentes del sistema")

        try:
            # Restaurar RL Agent (con o sin merge)
            logger.debug("  • Restaurando RL Agent...")
            if 'rl_agent' not in loaded:
                logger.error("❌ No se encontró 'rl_agent' en el archivo importado")
                return False

            self.rl_agent.load_from_dict(loaded['rl_agent'], merge=merge)
            logger.info("  ✅ RL Agent restaurado")

        except Exception as e:
            logger.error(f"❌ Error restaurando RL Agent: {e}", exc_info=True)
            return False

        try:
            # Restaurar Parameter Optimizer (con o sin merge)
            logger.debug("  • Restaurando Parameter Optimizer...")
            if 'parameter_optimizer' not in loaded:
                logger.warning("⚠️ No se encontró 'parameter_optimizer' en el archivo - usando estado vacío")
                # Continuar con optimizer vacío
            else:
                self.parameter_optimizer.load_from_dict(loaded['parameter_optimizer'], merge=merge)
            logger.info("  ✅ Parameter Optimizer restaurado")

        except Exception as e:
            logger.warning(f"⚠️ Error restaurando Parameter Optimizer (continuando): {e}")
            # No es crítico, continuar

        try:
            # Restaurar historial de cambios
            logger.debug("  • Restaurando historial de cambios...")
            if merge:
                # En merge, agregar cambios históricos a los existentes
                imported_changes = loaded.get('change_history', [])
                self.change_history.extend(imported_changes)
                # Mantener solo últimos 100
                if len(self.change_history) > 100:
                    self.change_history = self.change_history[-100:]
                logger.info(f"  ✅ {len(imported_changes)} cambios históricos agregados (total: {len(self.change_history)})")
            else:
                # En replace, reemplazar completamente
                self.change_history = loaded.get('change_history', [])
                logger.info(f"  ✅ Histórico de cambios restaurado: {len(self.change_history)} cambios")

        except Exception as e:
            logger.warning(f"⚠️ Error restaurando historial de cambios (continuando): {e}")
            # No es crítico, continuar

        try:
            # Restaurar metadata
            logger.debug("  • Restaurando metadata...")
            metadata = loaded.get('metadata', {})
            if metadata:
                if merge:
                    # En merge, solo actualizar si no existen
                    if not self.current_parameters:
                        self.current_parameters = metadata.get('current_parameters', {})
                    self.total_trades_processed += metadata.get('total_trades_processed', 0)

                    # total_trades_all_time SIEMPRE se acumula (nunca se resetea)
                    # Buscar en metadata primero, luego en rl_agent como fallback
                    imported_all_time = metadata.get('total_trades_all_time')
                    if imported_all_time is None:
                        # Fallback: usar total_experience_trades del RL agent si existe
                        rl_data = loaded.get('rl_agent', {})
                        imported_all_time = rl_data.get('total_experience_trades', metadata.get('total_trades_processed', 0))
                        logger.debug(f"  ⚠️ total_trades_all_time no encontrado en metadata, usando RL agent: {imported_all_time}")

                    self.total_trades_all_time += imported_all_time
                    self.total_parameter_changes += metadata.get('total_parameter_changes', 0)
                    logger.info(f"  ✅ Metadata acumulada (trades totales: {self.total_trades_all_time}, max leverage: {self._calculate_max_leverage()}x)")
                else:
                    # En replace, reemplazar completamente
                    self.current_parameters = metadata.get('current_parameters', {})
                    self.total_trades_processed = metadata.get('total_trades_processed', 0)

                    # Cargar total_trades_all_time con múltiples fallbacks
                    self.total_trades_all_time = metadata.get('total_trades_all_time')
                    if self.total_trades_all_time is None:
                        # Fallback 1: usar total_experience_trades del RL agent
                        rl_data = loaded.get('rl_agent', {})
                        self.total_trades_all_time = rl_data.get('total_experience_trades')
                        if self.total_trades_all_time is not None:
                            logger.debug(f"  ⚠️ total_trades_all_time no en metadata, usando RL agent total_experience_trades: {self.total_trades_all_time}")
                        else:
                            # Fallback 2: usar total_trades_processed
                            self.total_trades_all_time = metadata.get('total_trades_processed', 0)
                            logger.debug(f"  ⚠️ total_trades_all_time no encontrado, usando total_trades_processed: {self.total_trades_all_time}")

                    # Si aún es None o 0, usar el total_trades del RL agent que ya fue cargado
                    if self.total_trades_all_time == 0 and hasattr(self, 'rl_agent'):
                        self.total_trades_all_time = self.rl_agent.total_trades
                        logger.debug(f"  ⚠️ Usando total_trades del RL agent cargado: {self.total_trades_all_time}")

                    self.total_parameter_changes = metadata.get('total_parameter_changes', 0)
                    self.decision_mode = metadata.get('decision_mode', 'BALANCED')
                    logger.info(f"  ✅ Metadata restaurada (trades totales: {self.total_trades_all_time}, max leverage: {self._calculate_max_leverage()}x)")
            else:
                logger.warning("⚠️ No se encontró metadata en el archivo - usando valores por defecto")
                # Usar total_trades del RL agent como fallback si no hay metadata
                if hasattr(self, 'rl_agent'):
                    self.total_trades_all_time = self.rl_agent.total_trades
                    logger.debug(f"  ⚠️ Sin metadata, usando total_trades del RL agent: {self.total_trades_all_time}")

        except Exception as e:
            logger.warning(f"⚠️ Error restaurando metadata (continuando): {e}")
            # No es crítico, continuar
            # Intentar recuperar de RL agent como último recurso
            if hasattr(self, 'rl_agent'):
                self.total_trades_all_time = self.rl_agent.total_trades
                logger.debug(f"  ⚠️ Excepción en metadata, usando RL agent total_trades: {self.total_trades_all_time}")

        try:
            # Restaurar performance history
            logger.debug("  • Restaurando performance history...")
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
                logger.info(f"  ✅ Performance history restaurada ({len(self.performance_history)} entradas)")
            else:
                logger.warning("⚠️ No se encontró performance history en el archivo")

        except Exception as e:
            logger.warning(f"⚠️ Error restaurando performance history (continuando): {e}")
            # No es crítico, continuar

        # Éxito completo
        mode_str = "combinada" if merge else "restaurada"
        force_str = " (FORCE MODE)" if force else ""
        logger.info(f"✅ Paso 3/3 completado: Componentes restaurados")
        logger.info(f"🎉 Inteligencia importada y {mode_str} completamente{force_str}")

        # Resumen final
        logger.info(
            f"📊 Resumen de importación:\n"
            f"  • RL Agent: {self.rl_agent.total_trades} trades, {len(self.rl_agent.q_table)} estados\n"
            f"  • Total trades all time: {self.total_trades_all_time}\n"
            f"  • Max leverage: {self._calculate_max_leverage()}x\n"
            f"  • Parameter changes: {self.total_parameter_changes}\n"
            f"  • Change history: {len(self.change_history)} cambios"
        )

        return True
