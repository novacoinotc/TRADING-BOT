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

        # Referencia a market_monitor (se asigna desde main.py)
        # Necesaria para acceder a ml_system para export/import de training_buffer
        self.market_monitor = None

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

        Límites escalonados (MAX 10x para Binance Futures safety):
        - 0-50 trades: máximo 3x (conservador inicial)
        - 50-100 trades: máximo 5x
        - 100-200 trades: máximo 7x
        - 200+ trades: máximo 10x (límite absoluto del sistema)

        Returns:
            Leverage máximo permitido (1-10x)
        """
        if self.total_trades_all_time < 50:
            return 3
        elif self.total_trades_all_time < 100:
            return 5
        elif self.total_trades_all_time < 200:
            return 7
        else:
            return 10  # MÁXIMO ABSOLUTO - sincronizado con trading_schemas.py

    def _gpt_size_to_multiplier(self, size_recommendation: str) -> float:
        """
        Convierte la recomendación de tamaño de GPT a un multiplicador

        GPT recomienda: FULL, THREE_QUARTER, HALF, QUARTER, MINI, SKIP
        Retorna multiplicador para position sizing

        Args:
            size_recommendation: Recomendación de GPT (FULL/THREE_QUARTER/HALF/QUARTER/MINI/SKIP)

        Returns:
            Multiplicador (0.0 - 1.0)
        """
        size_map = {
            'FULL': 1.0,
            'THREE_QUARTER': 0.75,
            'HALF': 0.5,
            'QUARTER': 0.25,
            'MINI': 0.10,  # Para trades de riesgo/aprendizaje
            'SKIP': 0.0,
        }
        return size_map.get(size_recommendation.upper(), 0.5)  # Default HALF si no reconoce

    async def _restore_from_state(self, state: Dict, force: bool = False):
        """
        Restaura estado completo desde archivo de inteligencia
        CRÍTICO: Restaura paper trading ANTES de crear portfolio nuevo
        """
        logger.info("📦 Restaurando estado completo desde export...")
        logger.info(f"   Force mode: {force}")

        try:
            # ========== PASO 0: GUARDAR PAPER TRADING PARA RESTAURAR ==========
            paper_trading_to_restore = None

            if 'paper_trading' in state and state['paper_trading']:
                paper_state = state['paper_trading']

                # Verificar estructura
                if 'counters' in paper_state and 'closed_trades' in paper_state:
                    total_trades = paper_state.get('counters', {}).get('total_trades', 0)
                    closed_trades_count = len(paper_state.get('closed_trades', []))

                    if total_trades > 0:
                        logger.info(f"📥 Paper trading en export detectado:")
                        logger.info(f"   • Total trades: {total_trades}")
                        logger.info(f"   • Closed trades: {closed_trades_count}")
                        logger.info(f"   • Balance: ${paper_state.get('balance', 0):,.2f}")

                        # GUARDAR para restaurar DESPUÉS
                        paper_trading_to_restore = paper_state
                    else:
                        logger.warning("⚠️ Paper trading en export pero con 0 trades")
                else:
                    logger.warning("⚠️ Paper trading en export pero formato incorrecto (sin counters/closed_trades)")
            else:
                logger.info("ℹ️ No hay paper trading en export - se creará portfolio nuevo")

            # ========== PASO 1: Restaurar RL Agent ==========
            logger.info("📥 Paso 1/4: Restaurando RL Agent...")

            if 'rl_agent' in state:
                self.rl_agent.load_from_dict(state['rl_agent'])
                rl_stats = self.rl_agent.get_statistics()
                logger.info(f"  ✅ RL Agent restaurado")
                logger.info(f"     • {rl_stats.get('total_trades', 0)} trades")
                logger.info(f"     • {rl_stats.get('success_rate', 0):.1f}% win rate")
                logger.info(f"     • {rl_stats.get('q_table_size', 0)} estados aprendidos")

            # ========== PASO 2: Restaurar Parameter Optimizer ==========
            logger.info("📥 Paso 2/4: Restaurando Parameter Optimizer...")

            if 'parameter_optimizer' in state:
                self.parameter_optimizer.load_from_dict(state['parameter_optimizer'])
                logger.info(f"  ✅ Parameter Optimizer restaurado")

            # ========== PASO 3: Restaurar Change History ==========
            logger.info("📥 Paso 3/4: Restaurando Change History...")

            try:
                if 'change_history' in state:
                    self.change_history = state['change_history']
                    logger.info(f"  ✅ Histórico de cambios restaurado: {len(self.change_history)} cambios")
            except Exception as e:
                logger.warning(f"⚠️ Error restaurando change history: {e}")

            # ========== PASO 4: RESTAURAR PAPER TRADING (CRÍTICO) ==========
            logger.info("📥 Paso 4/4: Restaurando Paper Trading...")

            # NUEVO: Detectar si estamos en modo LIVE
            is_live_mode = False
            if hasattr(self, 'paper_trader') and self.paper_trader:
                # Verificar si es TradingSystem en modo LIVE
                if hasattr(self.paper_trader, 'is_live') and callable(self.paper_trader.is_live):
                    is_live_mode = self.paper_trader.is_live()

            if is_live_mode:
                # En modo LIVE, NO restauramos paper_trading (balance viene de Binance)
                logger.info("🔴 MODO LIVE DETECTADO - Saltando restauración de paper trading")
                logger.info("   • Balance real viene de Binance, no del export")
                logger.info("   • Solo se restauró: RL Agent, Optimizer, Change History")
                logger.info("   • La inteligencia (Q-table) SÍ fue restaurada ✅")
            elif paper_trading_to_restore:
                logger.info("🔄 Intentando restaurar paper trading desde export...")

                # VERIFICAR que paper_trader existe
                if not hasattr(self, 'paper_trader'):
                    logger.error("❌ CRÍTICO: self.paper_trader NO EXISTE")
                    logger.error("   El paper_trader debe inicializarse ANTES de importar")
                    logger.error("   Continuando sin restaurar paper trading...")
                elif not self.paper_trader:
                    logger.error("❌ CRÍTICO: self.paper_trader es None")
                elif not hasattr(self.paper_trader, 'portfolio'):
                    logger.error("❌ CRÍTICO: paper_trader.portfolio NO EXISTE")
                else:
                    # TODO LISTO - Restaurar
                    logger.info("✅ paper_trader existe - ejecutando restore_from_state()...")

                    try:
                        success = self.paper_trader.portfolio.restore_from_state(paper_trading_to_restore)

                        if success:
                            # Verificar que realmente se restauró
                            actual_trades = self.paper_trader.portfolio.total_trades
                            actual_closed = len(self.paper_trader.portfolio.closed_trades)
                            actual_balance = self.paper_trader.portfolio.balance

                            logger.info(f"✅ Paper Trading restaurado exitosamente:")
                            logger.info(f"   • Total trades: {actual_trades}")
                            logger.info(f"   • Closed trades: {actual_closed}")
                            logger.info(f"   • Balance: ${actual_balance:,.2f}")

                            # Verificación de integridad
                            expected_trades = paper_trading_to_restore['counters']['total_trades']
                            if actual_trades != expected_trades:
                                logger.warning(f"⚠️ Trades: esperado={expected_trades}, actual={actual_trades}")
                        else:
                            logger.error("❌ restore_from_state() retornó False")
                            logger.error("   Revisa logs de Portfolio para más detalles")

                    except Exception as e:
                        logger.error(f"❌ EXCEPCIÓN al ejecutar restore_from_state(): {e}", exc_info=True)
            else:
                logger.info("ℹ️ No hay paper trading para restaurar - portfolio quedará en estado inicial")

            # ========== Restaurar metadata y otros ==========
            try:
                if 'metadata' in state:
                    metadata = state['metadata']
                    self.total_trades_all_time = metadata.get('total_trades_all_time', 0)
                    self.max_leverage_unlocked = metadata.get('max_leverage_unlocked', 1)
                    logger.info(f"  ✅ Metadata restaurada (trades totales: {self.total_trades_all_time}, max leverage: {self.max_leverage_unlocked}x)")
            except Exception as e:
                logger.warning(f"⚠️ Error restaurando metadata: {e}")

            # ========== Restaurar performance history ==========
            try:
                if 'performance_history' in state:
                    perf = state['performance_history']
                    logger.info(f"  ✅ Performance history restaurada ({perf.get('total_trades', 0)} entradas)")
            except Exception as e:
                logger.warning(f"⚠️ Error restaurando performance history: {e}")

            # ========== Restaurar ML Training Buffer ==========
            logger.info("🧠 Restaurando ML Training Buffer...")

            try:
                if 'ml_training_buffer' in state and state['ml_training_buffer']:
                    buffer = state['ml_training_buffer']
                    # Verificar que tengamos acceso al ml_system via market_monitor
                    if hasattr(self, 'market_monitor') and self.market_monitor:
                        if hasattr(self.market_monitor, 'ml_system') and self.market_monitor.ml_system:
                            ml_system = self.market_monitor.ml_system
                            ml_system.training_buffer = buffer
                            # Crear mapa de features importadas para fallback
                            for record in buffer:
                                if 'trade_id' in record and 'features' in record:
                                    ml_system.imported_features[record['trade_id']] = record['features']
                            logger.info(f"  ✅ Training buffer restaurado: {len(buffer)} features")
                        else:
                            logger.warning("  ⚠️ ml_system no disponible para restaurar training buffer")
                    else:
                        logger.warning("  ⚠️ market_monitor no disponible para restaurar training buffer")
                else:
                    logger.debug("  ℹ️ No hay ml_training_buffer en el estado")
            except Exception as e:
                logger.warning(f"⚠️ Error restaurando ML buffer: {e}")

            # ========== VALIDACIÓN FINAL ==========
            logger.info("🎯 Validación final de sincronización...")

            rl_trades = self.rl_agent.get_statistics().get('total_trades', 0)
            paper_trades = self.paper_trader.portfolio.total_trades if hasattr(self, 'paper_trader') else 0

            if rl_trades != paper_trades:
                logger.warning(f"⚠️ DESINCRONIZACIÓN POST-IMPORT:")
                logger.warning(f"   RL Agent: {rl_trades} trades")
                logger.warning(f"   Paper Trading: {paper_trades} trades")
                logger.warning(f"   Diferencia: {abs(rl_trades - paper_trades)} trades")
            else:
                logger.info(f"✅ Sincronización OK: {rl_trades} trades en ambos sistemas")

            logger.info("🎉 Inteligencia importada y restaurada completamente")

            if force:
                logger.warning("⚠️ Importación en FORCE MODE - checksum no validado")

        except Exception as e:
            logger.error(f"❌ Error crítico restaurando estado: {e}", exc_info=True)
            raise

    async def evaluate_trade_opportunity(
        self,
        pair: str,
        signal: Dict,
        market_state: Dict,
        portfolio_metrics: Dict
    ) -> Dict:
        """
        Evalúa si abrir un trade usando GPT (control absoluto) + RL Agent (advisory)

        FLUJO DE DECISIÓN:
        1. RL Agent proporciona recomendación (advisory)
        2. GPT Trade Controller toma la DECISIÓN FINAL (absolute control)
        3. Se usan leverage, position size, SL/TP de GPT

        Args:
            pair: Par de trading
            signal: Señal generada (BUY/SELL/HOLD)
            market_state: Estado del mercado actual
            portfolio_metrics: Métricas del portfolio

        Returns:
            Dict con decisión (GPT si disponible, RL como fallback)
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

            # RL Agent proporciona recomendación (ADVISORY - GPT puede ignorarla)
            rl_decision = self.rl_agent.decide_trade_action(market_data, max_leverage=max_leverage)

            logger.info(
                f"🤖 RL Advisory para {pair}: "
                f"{rl_decision['chosen_action']} | "
                f"Trade: {'✅' if rl_decision['should_trade'] else '❌'} | "
                f"Size: {rl_decision['position_size_multiplier']:.1f}x"
            )

            # ========================================
            # GPT TRADE CONTROLLER - DECISIÓN FINAL
            # ========================================
            # GPT tiene CONTROL ABSOLUTO - puede aprobar, rechazar o modificar
            # la decisión del RL Agent

            if hasattr(self, 'gpt_brain') and self.gpt_brain:
                try:
                    # Verificar que market_state no sea None
                    if not market_state:
                        logger.warning("⚠️ market_state is None, skipping GPT evaluation")
                        return rl_decision

                    # Obtener lista de posiciones abiertas (con protección contra None)
                    open_positions = []
                    if hasattr(self, 'portfolio') and self.portfolio:
                        try:
                            positions = self.portfolio.get_open_positions()
                            if positions:
                                open_positions = [
                                    p.get('pair', '') for p in positions
                                    if p is not None and isinstance(p, dict)
                                ]
                        except Exception as e:
                            logger.debug(f"Could not get open positions: {e}")

                    # Preparar datos para GPT (con protección contra None)
                    indicators = {
                        'current_price': market_state.get('current_price', 0) or 0,
                        'rsi': market_state.get('rsi', 50) or 50,
                        'macd': market_state.get('macd', 0) or 0,
                        'macd_signal': market_state.get('macd_signal', 0) or 0,
                        'macd_histogram': market_state.get('macd_histogram', 0) or 0,
                        'ema_9': market_state.get('ema_9', 0) or 0,
                        'ema_21': market_state.get('ema_21', 0) or 0,
                        'ema_50': market_state.get('ema_50', 0) or 0,
                        'atr': market_state.get('atr', 0) or 0,
                        'adx': market_state.get('adx', 0) or 0,
                        'volume_ratio': market_state.get('volume_ratio', 1) or 1,
                        'bb_upper': market_state.get('bb_upper', 0) or 0,
                        'bb_lower': market_state.get('bb_lower', 0) or 0,
                        'bb_middle': market_state.get('bb_middle', 0) or 0,
                    }

                    # Fear & Greed con protección (evitar división por None)
                    fg_value = market_state.get('fear_greed', 50)
                    if fg_value is None:
                        fg_value = 50

                    sentiment_data = {
                        'fear_greed_index': fg_value / 100,
                        'fear_greed_label': market_state.get('fear_greed_label', 'Neutral') or 'Neutral',
                        'news_sentiment_overall': market_state.get('news_sentiment', 0.5) or 0.5,
                        'high_impact_news_count': market_state.get('high_impact_news', 0) or 0,
                    }

                    orderbook_data = {
                        'market_pressure': market_state.get('market_pressure', 'NEUTRAL') or 'NEUTRAL',
                        'imbalance': market_state.get('orderbook_imbalance', 0) or 0,
                        'depth_score': market_state.get('orderbook_depth_score', 50) or 50,
                    }

                    regime_data = {
                        'regime': market_state.get('regime', 'SIDEWAYS') or 'SIDEWAYS',
                        'regime_strength': market_state.get('regime_strength', 'MEDIUM') or 'MEDIUM',
                        'volatility': market_state.get('volatility_regime', 'NORMAL') or 'NORMAL',
                    }

                    # RL recommendation como input para GPT (con protección)
                    rl_recommendation = {
                        'action': rl_decision.get('chosen_action', 'HOLD') if rl_decision else 'HOLD',
                        'q_value': rl_decision.get('q_value', 0) if rl_decision else 0,
                        'is_exploration': rl_decision.get('is_exploration', False) if rl_decision else False,
                    }

                    # GPT toma la DECISIÓN FINAL
                    gpt_result = await self.gpt_brain.trade_controller.evaluate_signal(
                        pair=pair,
                        signal=signal,
                        indicators=indicators,
                        sentiment_data=sentiment_data,
                        orderbook_data=orderbook_data,
                        regime_data=regime_data,
                        ml_prediction=None,  # ML prediction si está disponible
                        rl_recommendation=rl_recommendation,
                        portfolio=portfolio_metrics,
                        open_positions=open_positions
                    )

                    if gpt_result.get('success'):
                        gpt_decision = gpt_result.get('decision', {})
                        approved = gpt_decision.get('approved', False)

                        # Construir decisión final basada en GPT
                        final_decision = {
                            'should_trade': approved,
                            'action': 'OPEN' if approved else 'SKIP',
                            'chosen_action': 'GPT_APPROVED' if approved else 'GPT_REJECTED',

                            # Position sizing de GPT
                            'position_size_multiplier': self._gpt_size_to_multiplier(
                                gpt_decision.get('position_size', {}).get('recommendation', 'HALF')
                            ),

                            # Leverage de GPT
                            'leverage': gpt_decision.get('leverage', {}).get('recommended', 3),
                            'max_leverage': gpt_decision.get('leverage', {}).get('max_safe', 5),

                            # Risk management de GPT
                            'stop_loss_pct': gpt_decision.get('risk_management', {}).get('stop_loss_pct', 1.5),
                            'take_profit_pct': gpt_decision.get('risk_management', {}).get('take_profit_pct', 2.5),
                            'trailing_stop': gpt_decision.get('risk_management', {}).get('trailing_stop', True),

                            # Metadata
                            'confidence': gpt_decision.get('confidence', 0),
                            'direction': gpt_decision.get('direction', 'LONG'),
                            'reasoning': gpt_decision.get('reasoning', ''),
                            'source': 'GPT_ABSOLUTE_CONTROL',

                            # RL info (para referencia)
                            'rl_recommendation': rl_decision.get('chosen_action'),
                            'rl_overridden': rl_decision.get('should_trade') != approved,
                        }

                        # Log decisión de GPT
                        override_msg = " (RL OVERRIDE)" if final_decision['rl_overridden'] else ""
                        logger.info(
                            f"🧠 GPT DECISION para {pair}: "
                            f"{'✅ APPROVED' if approved else '❌ REJECTED'}{override_msg} | "
                            f"Confidence: {final_decision['confidence']}% | "
                            f"Leverage: {final_decision['leverage']}x | "
                            f"Size: {final_decision['position_size_multiplier']:.0%}"
                        )

                        return final_decision

                except Exception as e:
                    logger.warning(f"⚠️ GPT evaluation failed, using RL fallback: {e}")
                    # Fallback a RL si GPT falla

            # Si GPT no está disponible o falló, usar RL decision
            return rl_decision

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

        # LOG CRÍTICO para debugging: Verificar que reward se calcula correctamente
        logger.info(
            f"🎓 RL LEARNING: {trade_data.get('pair')} | "
            f"P&L: {trade_data.get('profit_pct', 0):+.2f}% | "
            f"Leverage: {trade_data.get('leverage', 1)}x | "
            f"Reward calculado: {reward:+.3f}"
        )

        # Convertir estado de mercado a representación para RL
        state = self.rl_agent.get_state_representation(market_state)

        # Determinar si el episodio termina (grandes wins/losses)
        profit_pct = trade_data.get('profit_pct', 0)
        done = (profit_pct > 20) or (profit_pct < -10)  # Episodio termina en extremos

        # Determinar acción basada en el trade
        trade_action = trade_data.get('action', 'BUY')
        if trade_action == 'BUY':
            action_str = 'OPEN_NORMAL'
        elif trade_action == 'SELL':
            action_str = 'OPEN_NORMAL'
        else:
            action_str = 'OPEN_NORMAL'

        # RL Agent aprende del trade (ahora con estado y acción explícitos)
        self.rl_agent.learn_from_trade(
            reward=reward,
            next_state=state,
            done=done,
            state=state,  # Pasar estado explícitamente
            action=action_str  # Pasar acción explícitamente
        )

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

        # ===== GPT BRAIN LEARNING =====
        # Enviar trade cerrado a GPT Brain para aprendizaje (si está habilitado)
        if hasattr(self, 'gpt_brain') and self.gpt_brain:
            try:
                # Preparar contexto para GPT learning
                gpt_trade_data = {
                    'pair': trade_data.get('pair'),
                    'side': trade_data.get('action'),
                    'trade_type': trade_data.get('trade_type', 'FUTURES'),
                    'leverage': trade_data.get('leverage', 1),
                    'entry_price': trade_data.get('entry_price', 0),
                    'exit_price': trade_data.get('exit_price', 0),
                    'pnl': trade_data.get('profit_amount', 0),
                    'pnl_pct': trade_data.get('profit_pct', 0),
                    'duration': trade_data.get('duration', 0),
                    'reason': trade_data.get('exit_reason', 'UNKNOWN'),
                    'liquidated': trade_data.get('liquidated', False)
                }

                # Llamar al GPT Brain para aprendizaje
                if hasattr(self.gpt_brain, 'learn_from_closed_trade'):
                    await self.gpt_brain.learn_from_closed_trade(
                        trade=gpt_trade_data,
                        market_context=market_state,
                        signal_data={}  # Signal data no disponible en este punto
                    )
                    logger.debug(f"🧠 GPT Brain aprendió del trade: {trade_data.get('pair')}")
            except Exception as e:
                logger.warning(f"⚠️ Error en GPT Brain learning: {e}")
        # ===== FIN GPT BRAIN LEARNING =====

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
        trade_type = trade_data.get('trade_type', 'FUTURES')  # Default FUTURES
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

        # Validar sincronización ANTES de exportar
        sync_status = self.validate_sync()
        if not sync_status['in_sync']:
            logger.warning(
                f"⚠️ ADVERTENCIA: Exportando con desincronización\n"
                f"   Paper Trading: {sync_status['paper_trades']} trades\n"
                f"   RL Agent: {sync_status['rl_trades']} trades\n"
                f"   El export contendrá esta desincronización"
            )

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

        # Guardar estado de paper trading si existe (TODO EL HISTORIAL)
        paper_trading_state = None
        if hasattr(self, 'paper_trader') and self.paper_trader:
            paper_trading_state = self.paper_trader.portfolio.get_full_state_for_export()
            logger.debug(
                f"📤 Exportando paper trading: "
                f"{len(paper_trading_state.get('closed_trades', []))} trades, "
                f"{paper_trading_state['counters']['total_trades']} total histórico"
            )

        # Guardar training_buffer del ML System si existe
        ml_training_buffer = []
        if hasattr(self, 'market_monitor') and self.market_monitor:
            if hasattr(self.market_monitor, 'ml_system') and self.market_monitor.ml_system:
                ml_system = self.market_monitor.ml_system
                if hasattr(ml_system, 'training_buffer'):
                    ml_training_buffer = ml_system.training_buffer
                    logger.debug(f"🧠 ML Training Buffer incluido en export: {len(ml_training_buffer)} features")

        # Guardar GPT wisdom y trade memory si existe
        gpt_wisdom = {}
        gpt_trade_memory = []
        if hasattr(self, 'gpt_brain') and self.gpt_brain:
            if hasattr(self.gpt_brain, 'experience_learner'):
                learner = self.gpt_brain.experience_learner
                gpt_wisdom = learner.wisdom
                gpt_trade_memory = learner.trade_memory
                logger.debug(
                    f"🧠 GPT Wisdom incluido en export: "
                    f"{len(gpt_wisdom.get('lessons', []))} lecciones, "
                    f"{len(gpt_trade_memory)} trades en memoria"
                )

        success = self.persistence.save_full_state(
            rl_agent_state=rl_state,
            optimizer_state=optimizer_state,
            performance_history=performance_summary,
            change_history=self.change_history,  # Histórico de cambios con razonamiento
            metadata=metadata,
            paper_trading=paper_trading_state,  # incluir paper trading
            ml_training_buffer=ml_training_buffer,  # incluir training buffer
            gpt_wisdom=gpt_wisdom,  # NUEVO: incluir GPT wisdom
            gpt_trade_memory=gpt_trade_memory  # NUEVO: incluir GPT trade memory
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

    def validate_sync(self) -> Dict:
        """
        Valida sincronización entre TODOS los contadores de trades:
        - Paper Trading
        - RL Agent
        - AutonomyController (total_trades_processed y total_trades_all_time)
        - Win Rate entre Paper Trading y RL Agent
        """
        paper_trades = 0
        rl_trades = 0
        paper_win_rate = 0.0
        rl_win_rate = 0.0
        processed_trades = self.total_trades_processed
        all_time_trades = self.total_trades_all_time

        # Obtener conteos de cada sistema
        if hasattr(self, 'paper_trader') and self.paper_trader:
            paper_trades = self.paper_trader.portfolio.total_trades
            paper_stats = self.paper_trader.portfolio.get_statistics()
            paper_win_rate = paper_stats.get('win_rate', 0)

        if hasattr(self, 'rl_agent') and self.rl_agent:
            rl_stats = self.rl_agent.get_statistics()
            rl_trades = rl_stats.get('total_trades', 0)
            rl_win_rate = rl_stats.get('success_rate', 0)

        # Verificar sincronización completa (TODOS los contadores deben coincidir)
        # Win rate puede tener diferencia de hasta 1% por redondeo
        win_rate_in_sync = abs(paper_win_rate - rl_win_rate) < 1.0

        in_sync = (paper_trades == rl_trades and
                   paper_trades == processed_trades and
                   paper_trades == all_time_trades and
                   win_rate_in_sync)

        if not in_sync:
            logger.error(
                f"🚨 DESINCRONIZACIÓN DETECTADA:\n"
                f"   Paper Trading: {paper_trades} trades, {paper_win_rate:.1f}% win rate\n"
                f"   RL Agent: {rl_trades} trades, {rl_win_rate:.1f}% win rate\n"
                f"   Trades Procesados: {processed_trades}\n"
                f"   Total All Time: {all_time_trades}\n"
                f"   Usa /force_sync para sincronizar todos"
            )
        else:
            logger.debug(f"✅ Sincronización OK: {paper_trades} trades, {paper_win_rate:.1f}% win rate en TODOS los contadores")

        return {
            'in_sync': in_sync,
            'paper_trades': paper_trades,
            'rl_trades': rl_trades,
            'processed_trades': processed_trades,
            'all_time_trades': all_time_trades,
            'paper_win_rate': paper_win_rate,
            'rl_win_rate': rl_win_rate,
            'win_rate_in_sync': win_rate_in_sync,
            'differences': {
                'rl_vs_paper': abs(rl_trades - paper_trades),
                'processed_vs_paper': abs(processed_trades - paper_trades),
                'all_time_vs_paper': abs(all_time_trades - paper_trades),
                'win_rate_diff': abs(paper_win_rate - rl_win_rate)
            }
        }

    async def force_sync_from_paper(self) -> bool:
        """
        FUERZA sincronización usando Paper Trading como fuente de verdad

        ADVERTENCIA: Esto ajustará TODOS los contadores al Paper Trading,
        pero NO borrará el conocimiento aprendido (Q-table se mantiene).

        Sincroniza:
        - RL Agent total_trades y successful_trades
        - AutonomyController total_trades_processed
        - AutonomyController total_trades_all_time

        Returns:
            True si sincronización fue exitosa
        """
        if not hasattr(self, 'paper_trader') or not self.paper_trader:
            logger.error("❌ Paper trader no disponible")
            return False

        if not hasattr(self, 'rl_agent') or not self.rl_agent:
            logger.error("❌ RL Agent no disponible")
            return False

        # Obtener conteos actuales
        paper_trades = self.paper_trader.portfolio.total_trades
        rl_trades = self.rl_agent.total_trades
        processed_trades = self.total_trades_processed
        all_time_trades = self.total_trades_all_time

        # Obtener win rates
        paper_stats = self.paper_trader.portfolio.get_statistics()
        paper_win_rate = paper_stats['win_rate']
        rl_win_rate = self.rl_agent.get_success_rate()

        # Verificar si ya están sincronizados TODOS los contadores Y WIN RATE
        trades_in_sync = (paper_trades == rl_trades and
                         paper_trades == processed_trades and
                         paper_trades == all_time_trades)
        win_rate_in_sync = abs(paper_win_rate - rl_win_rate) < 1.0  # Tolerancia de 1%

        if trades_in_sync and win_rate_in_sync:
            logger.info(f"✅ Ya están sincronizados todos los contadores y win rate ({paper_win_rate:.1f}%), no se requiere acción")
            return True

        # Si solo los contadores están sincronizados pero no el win rate
        if trades_in_sync and not win_rate_in_sync:
            logger.warning(
                f"⚠️ Contadores sincronizados pero WIN RATE desincronizado:\n"
                f"   Paper Trading: {paper_trades} trades, {paper_win_rate:.1f}% WR\n"
                f"   RL Agent: {rl_trades} trades, {rl_win_rate:.1f}% WR\n"
                f"   FORZANDO SINCRONIZACIÓN DE WIN RATE..."
            )

        logger.warning(
            f"⚠️ FORZANDO SINCRONIZACIÓN COMPLETA:\n"
            f"   Paper Trading: {paper_trades} trades (FUENTE DE VERDAD)\n"
            f"   \n"
            f"   ANTES:\n"
            f"   • RL Agent: {rl_trades} trades\n"
            f"   • Trades Procesados: {processed_trades}\n"
            f"   • Total All Time: {all_time_trades}\n"
            f"   \n"
            f"   DESPUÉS (todos ajustados a {paper_trades}):"
        )

        # 1. Ajustar contador del RL Agent
        old_rl_trades = self.rl_agent.total_trades
        old_successful = self.rl_agent.successful_trades
        self.rl_agent.total_trades = paper_trades

        # 2. Ajustar successful_trades usando SIEMPRE Paper Trading como fuente de verdad
        # paper_stats y paper_win_rate ya calculados arriba en la línea 941-942
        new_successful = int(paper_trades * paper_win_rate / 100)

        logger.info(
            f"🔄 Actualizando RL Agent:\n"
            f"   ANTES: total_trades={old_rl_trades}, successful_trades={old_successful} ({(old_successful/old_rl_trades*100) if old_rl_trades > 0 else 0:.1f}% WR)\n"
            f"   DESPUÉS: total_trades={paper_trades}, successful_trades={new_successful} ({paper_win_rate:.1f}% WR)"
        )

        self.rl_agent.successful_trades = new_successful

        # Verificar que se actualizó correctamente
        actual_wr = self.rl_agent.get_success_rate()
        logger.info(f"✅ Verificación: RL Agent ahora tiene {self.rl_agent.successful_trades}/{self.rl_agent.total_trades} = {actual_wr:.1f}% WR")

        # 3. Ajustar contadores del AutonomyController
        self.total_trades_processed = paper_trades
        self.total_trades_all_time = paper_trades

        logger.info(f"✅ Sincronización forzada completada - TODOS los contadores:")
        logger.info(f"   • Paper Trading: {paper_trades} trades ✅")
        logger.info(f"   • RL Agent: {self.rl_agent.total_trades} trades ✅")
        logger.info(f"   • Trades Procesados: {self.total_trades_processed} ✅")
        logger.info(f"   • Total All Time: {self.total_trades_all_time} ✅")

        # Guardar el estado sincronizado
        await self.save_intelligence()

        return True

    def get_statistics(self) -> Dict:
        """Retorna estadísticas completas del sistema autónomo"""
        rl_stats = self.rl_agent.get_statistics()
        opt_stats = self.parameter_optimizer.get_optimization_statistics()

        # Validar sincronización
        sync_status = self.validate_sync()

        # Log warning si no están sincronizados
        if not sync_status['in_sync']:
            diffs = sync_status['differences']
            logger.warning(
                f"⚠️ Desincronización detectada:\n"
                f"   RL vs Paper: {diffs['rl_vs_paper']} trades\n"
                f"   Processed vs Paper: {diffs['processed_vs_paper']} trades\n"
                f"   All-time vs Paper: {diffs['all_time_vs_paper']} trades\n"
                f"   Win Rate diff: {diffs['win_rate_diff']:.1f}%"
            )

        return {
            'active': self.active,
            'decision_mode': self.decision_mode,
            'total_trades_processed': self.total_trades_processed,
            'total_parameter_changes': self.total_parameter_changes,
            'rl_agent': rl_stats,
            'parameter_optimizer': opt_stats,
            'sync_status': sync_status,  # ✅ NUEVO: validación de sincronización
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

        # ===== RESTAURAR PAPER TRADING (USANDO MÉTODO CORRECTO) =====
        try:
            logger.info("📥 Verificando Paper Trading en archivo...")

            # NUEVO: Detectar si estamos en modo LIVE
            is_live_mode = False
            if hasattr(self, 'paper_trader') and self.paper_trader:
                if hasattr(self.paper_trader, 'is_live') and callable(self.paper_trader.is_live):
                    is_live_mode = self.paper_trader.is_live()

            if is_live_mode:
                # En modo LIVE, NO restauramos paper_trading
                logger.info("🔴 MODO LIVE DETECTADO - Saltando restauración de paper trading")
                logger.info("   • Balance real viene de Binance, no del export")
                logger.info("   • Inteligencia restaurada: RL Agent ✅, Optimizer ✅, Change History ✅")
                logger.info("   • Q-table con 226 trades de experiencia lista para trading real")
            elif 'paper_trading' in loaded and loaded['paper_trading']:
                paper_data = loaded['paper_trading']

                # Verificar estructura del export (NUEVO FORMATO con counters)
                if 'counters' in paper_data and 'closed_trades' in paper_data:
                    total_trades = paper_data.get('counters', {}).get('total_trades', 0)
                    closed_trades_count = len(paper_data.get('closed_trades', []))

                    if total_trades > 0:
                        logger.info(f"📊 Paper trading detectado en export:")
                        logger.info(f"   • Total trades: {total_trades}")
                        logger.info(f"   • Closed trades: {closed_trades_count}")
                        logger.info(f"   • Balance: ${paper_data.get('balance', 0):,.2f}")

                        # Verificar que paper_trader existe
                        if not hasattr(self, 'paper_trader') or self.paper_trader is None:
                            logger.error("❌ CRÍTICO: self.paper_trader NO EXISTE")
                            logger.error("   El paper_trader debe asignarse desde main.py ANTES de import")
                            logger.error("   Continuando sin restaurar paper trading...")
                        elif not hasattr(self.paper_trader, 'portfolio'):
                            logger.error("❌ CRÍTICO: paper_trader.portfolio NO EXISTE")
                        else:
                            # TODO LISTO - Usar método restore_from_state()
                            logger.info("✅ paper_trader existe - ejecutando restore_from_state()...")

                            try:
                                success = self.paper_trader.portfolio.restore_from_state(paper_data)

                                if success:
                                    # Verificar que realmente se restauró
                                    actual_trades = self.paper_trader.portfolio.total_trades
                                    actual_closed = len(self.paper_trader.portfolio.closed_trades)
                                    actual_balance = self.paper_trader.portfolio.balance

                                    logger.info(f"✅ Paper Trading restaurado exitosamente:")
                                    logger.info(f"   • Total trades: {actual_trades}")
                                    logger.info(f"   • Closed trades: {actual_closed}")
                                    logger.info(f"   • Balance: ${actual_balance:,.2f}")

                                    # Verificación de integridad
                                    if actual_trades != total_trades:
                                        logger.warning(f"⚠️ Trades: esperado={total_trades}, actual={actual_trades}")
                                else:
                                    logger.error("❌ restore_from_state() retornó False")
                                    logger.error("   Revisa logs de Portfolio para más detalles")

                            except Exception as e:
                                logger.error(f"❌ EXCEPCIÓN al ejecutar restore_from_state(): {e}", exc_info=True)
                    else:
                        logger.warning("⚠️ Paper trading en export pero con 0 trades")
                else:
                    logger.warning("⚠️ Paper trading en export pero formato incorrecto (sin counters/closed_trades)")
                    logger.warning("   Puede ser un export antiguo - usa un export reciente creado con /export")
            else:
                logger.info("ℹ️ No hay paper trading en el export")

        except Exception as e:
            logger.error(f"❌ Error restaurando Paper Trading: {e}", exc_info=True)
            # No es crítico, continuar
        # ===== FIN RESTAURAR PAPER TRADING =====

        # ===== RESTAURAR ML TRAINING BUFFER =====
        # El training_buffer contiene las features necesarias para entrenar el ML
        # Sin estas features, el ML no puede reentrenarse con los trades importados
        try:
            if 'ml_training_buffer' in loaded and loaded['ml_training_buffer']:
                logger.info("🧠 Restaurando ML Training Buffer...")

                # Verificar que tengamos acceso al ml_system
                if hasattr(self, 'market_monitor') and self.market_monitor:
                    if hasattr(self.market_monitor, 'ml_system') and self.market_monitor.ml_system:
                        ml_system = self.market_monitor.ml_system
                        buffer_data = loaded['ml_training_buffer']

                        # Restaurar training_buffer
                        ml_system.training_buffer = buffer_data
                        logger.info(f"  ✅ Training buffer restaurado: {len(buffer_data)} features")

                        # NUEVO: Crear mapa trade_id → features para fallback
                        # Esto permite que _get_features_for_trades() encuentre features
                        # incluso si el training_buffer se reorganiza
                        imported_features_map = {}
                        for record in buffer_data:
                            if 'trade_id' in record and 'features' in record:
                                imported_features_map[record['trade_id']] = record['features']

                        ml_system.imported_features = imported_features_map
                        logger.info(f"  ✅ Features indexadas: {len(imported_features_map)} trade IDs")

                        # Guardar a disco
                        ml_system._save_buffer()
                        logger.debug(f"  💾 Training buffer guardado en disco")
                    else:
                        logger.warning("⚠️ ML System no disponible, no se puede restaurar training_buffer")
                else:
                    logger.warning("⚠️ Market Monitor no disponible, no se puede restaurar training_buffer")
            else:
                logger.debug("ℹ️  No se encontró 'ml_training_buffer' en el archivo importado (puede ser un export antiguo)")

        except Exception as e:
            logger.error(f"❌ Error restaurando ML Training Buffer: {e}", exc_info=True)
            # No es crítico, continuar
        # ===== FIN RESTAURAR ML TRAINING BUFFER =====

        # ===== PARCHE DIRECTO PARA total_trades_all_time =====
        # Verificar si total_trades_all_time quedó en 0
        if self.total_trades_all_time == 0:
            logger.warning("⚠️ CRÍTICO: total_trades_all_time está en 0, aplicando parche...")

            # Intento 1: Desde RL Agent ya cargado
            if hasattr(self.rl_agent, 'total_trades') and self.rl_agent.total_trades > 0:
                self.total_trades_all_time = self.rl_agent.total_trades
                logger.warning(f"⚠️ PARCHE APLICADO: total_trades_all_time = {self.total_trades_all_time} (desde RL Agent)")

            # Intento 2: Desde el archivo cargado
            elif 'rl_agent' in loaded:
                rl_data = loaded['rl_agent']
                experience_trades = rl_data.get('total_experience_trades', 0)
                if experience_trades > 0:
                    self.total_trades_all_time = experience_trades
                    logger.warning(f"⚠️ PARCHE APLICADO: total_trades_all_time = {experience_trades} (desde total_experience_trades)")
                else:
                    # Intento 3: Desde statistics del RL agent
                    stats = rl_data.get('statistics', {})
                    total_from_stats = stats.get('total_trades', 0)
                    if total_from_stats > 0:
                        self.total_trades_all_time = total_from_stats
                        logger.warning(f"⚠️ PARCHE APLICADO: total_trades_all_time = {total_from_stats} (desde statistics)")

        # Log final para confirmar
        logger.info(f"🎯 VERIFICACIÓN FINAL: total_trades_all_time = {self.total_trades_all_time}")
        logger.info(f"🎯 Max leverage desbloqueado = {self._calculate_max_leverage()}x")
        # ===== FIN DEL PARCHE =====

        # ===== PARCHE DIRECTO PARA total_parameter_changes =====
        # Verificar también total_parameter_changes
        if self.total_parameter_changes == 0:
            logger.warning("⚠️ CRÍTICO: total_parameter_changes está en 0, aplicando parche...")

            # Buscar en metadata
            metadata = loaded.get('metadata', {})
            param_changes = metadata.get('total_parameter_changes', 0)
            if param_changes > 0:
                self.total_parameter_changes = param_changes
                logger.warning(f"⚠️ PARCHE APLICADO: total_parameter_changes = {param_changes}")
            else:
                # Buscar en parameter_optimizer
                param_opt = loaded.get('parameter_optimizer', {})
                total_opts = param_opt.get('total_optimizations', 0)
                if total_opts > 0:
                    self.total_parameter_changes = total_opts
                    logger.warning(f"⚠️ PARCHE APLICADO: total_parameter_changes = {total_opts} (desde total_optimizations)")

        # Log final de verificación
        logger.info(f"🎯 VERIFICACIÓN FINAL: total_parameter_changes = {self.total_parameter_changes}")
        # ===== FIN DEL PARCHE =====

        # ===== AUTO-ENTRENAMIENTO ML SI HAY DATOS =====
        # Si tenemos suficientes trades y features, entrenar ML automáticamente
        try:
            if hasattr(self, 'market_monitor') and self.market_monitor:
                if hasattr(self.market_monitor, 'ml_system') and self.market_monitor.ml_system:
                    ml_system = self.market_monitor.ml_system

                    # Verificar si tenemos trades y features
                    if hasattr(self, 'paper_trader') and self.paper_trader:
                        paper_trader = self.paper_trader
                        stats = paper_trader.portfolio.get_statistics()
                        total_trades = stats.get('total_trades', 0)
                        buffer_size = len(ml_system.training_buffer)
                        imported_features_size = len(ml_system.imported_features)

                        logger.info(f"🤖 Verificando posibilidad de auto-entrenamiento ML:")
                        logger.info(f"   • Trades: {total_trades}")
                        logger.info(f"   • Training buffer: {buffer_size} registros")
                        logger.info(f"   • Features importadas: {imported_features_size} trades")

                        # Entrenar si tenemos 40+ trades y al menos 25 features
                        if total_trades >= 40 and (buffer_size >= 25 or imported_features_size >= 25):
                            logger.info("🚀 Iniciando auto-entrenamiento ML con datos importados...")

                            # Forzar entrenamiento con threshold reducido
                            ml_system.force_retrain(
                                min_samples_override=25,
                                external_paper_trader=paper_trader
                            )

                            # Verificar resultado
                            model_info = ml_system.trainer.get_model_info()
                            if model_info.get('available'):
                                metrics = model_info.get('metrics', {})
                                logger.info("✅ ML ENTRENADO EXITOSAMENTE con datos importados:")
                                logger.info(f"   • Accuracy: {metrics.get('test_accuracy', 0):.1%}")
                                logger.info(f"   • Precision: {metrics.get('test_precision', 0):.1%}")
                                logger.info(f"   • F1 Score: {metrics.get('test_f1', 0):.3f}")
                            else:
                                logger.warning("⚠️ Auto-entrenamiento completado pero modelo no disponible")
                        else:
                            logger.info("ℹ️  No hay suficientes datos para auto-entrenamiento ML")
                            logger.info("   El ML entrenará automáticamente cuando haya 40+ trades")

        except Exception as e:
            logger.error(f"❌ Error en auto-entrenamiento ML: {e}", exc_info=True)
            logger.info("   El ML entrenará más tarde cuando haya datos suficientes")
        # ===== FIN AUTO-ENTRENAMIENTO ML =====

        # ===== RESTAURAR GPT WISDOM =====
        try:
            if 'gpt_brain' in loaded and loaded['gpt_brain']:
                gpt_data = loaded['gpt_brain']
                logger.info("🧠 Restaurando GPT Wisdom...")

                # Verificar que tengamos acceso al gpt_brain
                if hasattr(self, 'gpt_brain') and self.gpt_brain:
                    if hasattr(self.gpt_brain, 'experience_learner'):
                        learner = self.gpt_brain.experience_learner

                        # Restaurar wisdom
                        if 'wisdom' in gpt_data and gpt_data['wisdom']:
                            if merge:
                                # En merge, combinar lessons y patterns
                                existing_lessons = learner.wisdom.get('lessons', [])
                                new_lessons = gpt_data['wisdom'].get('lessons', [])
                                for lesson in new_lessons:
                                    if lesson not in existing_lessons:
                                        existing_lessons.append(lesson)
                                learner.wisdom['lessons'] = existing_lessons

                                # Merge golden rules
                                existing_rules = learner.wisdom.get('golden_rules', [])
                                new_rules = gpt_data['wisdom'].get('golden_rules', [])
                                for rule in new_rules:
                                    if rule not in existing_rules:
                                        existing_rules.append(rule)
                                learner.wisdom['golden_rules'] = existing_rules

                                # Merge patterns
                                for pattern_type in ['winning_patterns', 'losing_patterns']:
                                    existing = learner.wisdom.get('patterns', {}).get(pattern_type, [])
                                    new_patterns = gpt_data['wisdom'].get('patterns', {}).get(pattern_type, [])
                                    for pattern in new_patterns:
                                        if pattern not in existing:
                                            existing.append(pattern)
                                    learner.wisdom['patterns'][pattern_type] = existing

                                logger.info(f"  ✅ GPT Wisdom combinada")
                            else:
                                # En replace, reemplazar completamente
                                learner.wisdom = gpt_data['wisdom']
                                logger.info(f"  ✅ GPT Wisdom restaurada")

                            # Guardar a disco
                            learner._save_wisdom()

                            # Log stats
                            lessons = len(learner.wisdom.get('lessons', []))
                            rules = len(learner.wisdom.get('golden_rules', []))
                            patterns = len(learner.wisdom.get('patterns', {}).get('winning_patterns', []))
                            logger.info(f"  📊 {lessons} lecciones, {rules} reglas de oro, {patterns} patrones")

                        # Restaurar trade memory
                        if 'trade_memory' in gpt_data and gpt_data['trade_memory']:
                            if merge:
                                learner.trade_memory.extend(gpt_data['trade_memory'])
                                # Mantener solo últimos 500
                                learner.trade_memory = learner.trade_memory[-500:]
                            else:
                                learner.trade_memory = gpt_data['trade_memory']

                            learner._save_trade_memory()
                            logger.info(f"  ✅ GPT Trade Memory restaurada: {len(learner.trade_memory)} trades")
                    else:
                        logger.warning("⚠️ GPT Experience Learner no disponible")
                else:
                    logger.warning("⚠️ GPT Brain no disponible, no se puede restaurar wisdom")
            else:
                logger.debug("ℹ️  No se encontró 'gpt_brain' en el archivo (puede ser un export antiguo)")

        except Exception as e:
            logger.error(f"❌ Error restaurando GPT Wisdom: {e}", exc_info=True)
            # No es crítico, continuar
        # ===== FIN RESTAURAR GPT WISDOM =====

        return True
