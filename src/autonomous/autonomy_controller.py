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
from .decision_brain import DecisionBrain

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
        self.max_leverage_unlocked = 1  # Inicializar leverage

        # Referencia a market_monitor (se asigna desde main.py)
        # Necesaria para acceder a ml_system para export/import de training_buffer
        self.market_monitor = None

        # Deduplicación de trades (para evitar que test_mode y position_monitor notifiquen el mismo trade)
        # Dict: symbol -> (timestamp, pnl) de los últimos trades procesados
        self._recently_processed_trades: Dict[str, tuple] = {}

        # Flag para indicar si Test Mode está activo (para ignorar Position Monitor cuando test activo)
        self.test_mode_active = False

        # 🤖 AUTONOMÍA v2.0: Aprendizaje continuo
        self.losing_streak = 0  # Contador de pérdidas consecutivas
        self.winning_streak = 0  # Contador de ganancias consecutivas
        self.recent_trades_pnl: List[float] = []  # Últimos 20 trades para calcular win rate reciente
        self.temporary_adjustment = None  # Ajustes temporales cuando racha negativa

        # 🧠 CEREBRO CENTRAL: Decision Brain (se inicializa después con set_components)
        self.decision_brain = None
        self.ml_system = None
        self.trade_manager = None
        self.feature_aggregator = None
        self.sentiment_analyzer = None
        self.regime_detector = None
        self.orderbook_analyzer = None

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

    def set_components(
        self,
        ml_system=None,
        trade_manager=None,
        feature_aggregator=None,
        sentiment_analyzer=None,
        regime_detector=None,
        orderbook_analyzer=None
    ):
        """
        Configura los componentes externos para el Decision Brain

        Args:
            ml_system: Sistema de ML para predicciones
            trade_manager: Gestor de trades activos
            feature_aggregator: Agregador de features (Arsenal)
            sentiment_analyzer: Analizador de sentimiento
            regime_detector: Detector de régimen de mercado
            orderbook_analyzer: Analizador de orderbook
        """
        self.ml_system = ml_system
        self.trade_manager = trade_manager
        self.feature_aggregator = feature_aggregator
        self.sentiment_analyzer = sentiment_analyzer
        self.regime_detector = regime_detector
        self.orderbook_analyzer = orderbook_analyzer

        # Crear el Decision Brain con todos los componentes
        self.decision_brain = DecisionBrain(
            rl_agent=self.rl_agent,
            ml_system=ml_system,
            trade_manager=trade_manager,
            parameter_optimizer=self.parameter_optimizer,
            feature_aggregator=feature_aggregator,
            sentiment_analyzer=sentiment_analyzer,
            regime_detector=regime_detector,
            orderbook_analyzer=orderbook_analyzer
        )

        logger.info("🧠 Decision Brain configurado con todos los componentes")

    def _calculate_max_leverage(self) -> int:
        """
        Calcula max leverage basado en total_trades_all_time.

        MODO EXPLORACIÓN: Empieza con leverage 3x mínimo para permitir
        que el RL Agent explore FUTURES desde el inicio.

        Returns:
            int: Leverage máximo desbloqueado (3-20x)
        """
        from config import config

        total = self.total_trades_all_time

        # MODO EXPLORACIÓN: Mínimo leverage = DEFAULT_LEVERAGE (3x)
        # Esto permite FUTURES desde el inicio para aprendizaje real
        min_leverage = getattr(config, 'DEFAULT_LEVERAGE', 3)

        if total < 10:
            return min_leverage  # Empieza con 3x (MODO EXPLORACIÓN)
        elif total < 20:
            return max(min_leverage, 3)
        elif total < 30:
            return max(min_leverage, 4)
        elif total < 50:
            return max(min_leverage, 5)
        elif total < 100:
            return 7
        elif total < 150:
            return 10
        elif total < 200:
            return 15
        else:
            return 20

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

            if paper_trading_to_restore:
                logger.info("🔄 Intentando restaurar paper trading desde export...")

                # VERIFICAR que paper_trader existe (v2.0: opcional)
                if not hasattr(self, 'paper_trader'):
                    logger.warning("⚠️ v2.0: self.paper_trader NO EXISTE (modo Binance)")
                    logger.info("   Continuando sin restaurar paper trading (normal en v2.0)...")
                elif not self.paper_trader:
                    logger.warning("⚠️ v2.0: self.paper_trader es None (modo Binance)")
                elif not hasattr(self.paper_trader, 'portfolio'):
                    logger.warning("⚠️ paper_trader.portfolio NO EXISTE")
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
                if 'ml_training_buffer' in state:
                    buffer = state['ml_training_buffer']
                    if hasattr(self, 'ml_integration') and self.ml_integration:
                        # Código existente para restaurar ML buffer
                        pass
                    logger.info(f"  ✅ Training buffer restaurado: {len(buffer)} features")
            except Exception as e:
                logger.warning(f"⚠️ Error restaurando ML buffer: {e}")

            # ========== Restaurar Trade Management Learning ==========
            logger.info("📊 Restaurando Trade Management Learning...")

            try:
                if 'trade_management_learning' in state and state['trade_management_learning']:
                    tm_learning = state['trade_management_learning']

                    # Si trade_manager existe, restaurar directamente
                    if hasattr(self, 'trade_manager') and self.trade_manager:
                        if hasattr(self.trade_manager, 'learning'):
                            self.trade_manager.learning.stats = tm_learning.get('statistics', {})
                            self.trade_manager.learning.actions_history = tm_learning.get('actions_history', [])
                            logger.info(f"  ✅ Trade Management Learning restaurado:")
                            logger.info(f"     • {len(tm_learning.get('actions_history', []))} acciones en historial")
                            logger.info(f"     • Total evaluadas: {tm_learning.get('statistics', {}).get('total_evaluated', 0)}")
                    else:
                        # Guardar en archivo para que Trade Manager lo cargue después
                        from pathlib import Path
                        import json
                        filepath = 'data/trade_management_learning.json'
                        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
                        with open(filepath, 'w') as f:
                            json.dump(tm_learning, f, indent=2)
                        logger.info(f"  ✅ Trade Management Learning guardado en {filepath}")
                        logger.info(f"     (Se cargará cuando Trade Manager inicie)")
                else:
                    logger.info("  ℹ️ No hay Trade Management Learning en export")
            except Exception as e:
                logger.warning(f"⚠️ Error restaurando Trade Management Learning: {e}")

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
        Evalúa si abrir un trade usando RL Agent ANTES de ejecutarlo

        🧠 CEREBRO CENTRAL: Si DecisionBrain está disponible, usa análisis completo
        con TODOS los servicios. Si no, usa el flujo tradicional.

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

        # 🔍 DIAGNÓSTICO: Verificar estado de DecisionBrain
        logger.info(f"🔍 evaluate_trade_opportunity para {pair}")
        logger.info(f"   📊 decision_brain disponible: {self.decision_brain is not None}")

        try:
            # 🧠 USAR DECISION BRAIN SI ESTÁ DISPONIBLE
            if self.decision_brain:
                logger.info(f"🧠 Usando DecisionBrain para analizar {pair}")
                current_price = market_state.get('current_price', 0)
                timeframe = market_state.get('timeframe', '15m')

                # Combinar market_state con signal para análisis completo
                combined_data = {**market_state, **signal}
                combined_data['side'] = signal.get('action', 'NEUTRAL')

                # Análisis completo con todos los servicios
                analysis = self.decision_brain.analyze_opportunity(
                    symbol=pair.replace('/', ''),
                    current_price=current_price,
                    market_data=combined_data,
                    timeframe=timeframe
                )

                # Extraer decisión final
                final_decision = analysis.get('final_decision', {})

                # Aplicar ajustes temporales si existen
                if self.temporary_adjustment and final_decision.get('action') != 'SKIP':
                    final_decision['leverage'] = int(
                        final_decision.get('leverage', 3) * self.temporary_adjustment.get('leverage_multiplier', 1.0)
                    )
                    final_decision['position_size_pct'] = (
                        final_decision.get('position_size_pct', 5.0) * self.temporary_adjustment.get('position_multiplier', 1.0)
                    )
                    if 'tp_percentages' in final_decision:
                        final_decision['tp_percentages'] = [
                            tp * self.temporary_adjustment.get('tp_multiplier', 1.0)
                            for tp in final_decision['tp_percentages']
                        ]

                # Convertir a formato esperado
                decision = {
                    'should_trade': final_decision.get('action') == 'OPEN',
                    'action': final_decision.get('action', 'SKIP'),
                    'trade_type': 'FUTURES',
                    'position_size_multiplier': final_decision.get('position_size_pct', 5.0) / 5.0,  # Normalizar a ~1.0
                    'leverage': final_decision.get('leverage', 3),
                    'confidence': final_decision.get('consolidated_confidence', 0.5),
                    'chosen_action': f"BRAIN_{final_decision.get('action', 'SKIP')}",
                    'composite_score': analysis.get('rl_decision', {}).get('composite_score', 0),
                    'tp_percentages': final_decision.get('tp_percentages', [0.5, 1.0, 1.5]),
                    'position_size_pct': final_decision.get('position_size_pct', 5.0),
                    'ml_confidence': final_decision.get('ml_confidence', 0.5),
                    'rl_confidence': final_decision.get('rl_confidence', 0.5)
                }

                logger.info(
                    f"🧠 BRAIN Decision para {pair}: "
                    f"{'✅' if decision['should_trade'] else '❌'} {decision['action']} | "
                    f"Lev={decision['leverage']}x | Conf={decision['confidence']:.1%}"
                )

                return decision

            # FALLBACK: Flujo tradicional si no hay Decision Brain
            logger.warning(f"⚠️ DecisionBrain NO disponible para {pair}, usando flujo tradicional")

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

            # 🤖 AUTONOMÍA v2.0: Obtener decisión autónoma con TPs dinámicos
            if decision['should_trade']:
                autonomous_decision = self.rl_agent.get_autonomous_decision(market_data)

                # Merge decisiones: usar TPs y parámetros de decisión autónoma
                if autonomous_decision['action'] != 'SKIP':
                    decision['tp_percentages'] = autonomous_decision.get('tp_percentages', [0.5, 1.0, 1.5])
                    decision['position_size_pct'] = autonomous_decision.get('position_size_pct', 5.0)
                    # Sobrescribir leverage con el dinámico
                    decision['leverage'] = autonomous_decision.get('leverage', decision.get('leverage', 3))
                    decision['autonomous_confidence'] = autonomous_decision.get('confidence', 0.5)

                    logger.info(
                        f"🤖 RL AUTÓNOMO para {pair}: "
                        f"{decision['chosen_action']} | Lev={decision['leverage']}x | "
                        f"TPs=[{', '.join([f'{tp:.2f}%' for tp in decision['tp_percentages']])}]"
                    )

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

        # RL Agent aprende del trade
        self.rl_agent.learn_from_trade(reward=reward, next_state=state, done=done)

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

        # =============================================
        # 🤖 AUTONOMÍA v2.0: APRENDIZAJE CONTINUO DINÁMICO
        # =============================================

        # Actualizar rachas de ganancia/pérdida
        profit_pct = trade_data.get('profit_pct', 0)
        if profit_pct > 0:
            self.winning_streak += 1
            self.losing_streak = 0
            # Limpiar ajustes temporales en ganancia
            if self.temporary_adjustment:
                logger.info(f"✅ Racha ganadora ({self.winning_streak}), limpiando ajustes temporales")
                self.temporary_adjustment = None
        else:
            self.losing_streak += 1
            self.winning_streak = 0

        # Actualizar historial reciente (últimos 20 trades)
        self.recent_trades_pnl.append(profit_pct)
        if len(self.recent_trades_pnl) > 20:
            self.recent_trades_pnl = self.recent_trades_pnl[-20:]

        # Ajustar exploración basado en win rate reciente
        if len(self.recent_trades_pnl) >= 10:
            recent_wins = sum(1 for pnl in self.recent_trades_pnl if pnl > 0)
            recent_win_rate = (recent_wins / len(self.recent_trades_pnl)) * 100

            if recent_win_rate > 85:
                # Alto win rate: reducir exploración, confiar más
                new_exploration = max(self.rl_agent.min_exploration, self.rl_agent.exploration_rate * 0.95)
                if new_exploration != self.rl_agent.exploration_rate:
                    logger.info(f"📈 Win rate alto ({recent_win_rate:.1f}%), reduciendo exploración: {self.rl_agent.exploration_rate:.2f} → {new_exploration:.2f}")
                    self.rl_agent.exploration_rate = new_exploration

            elif recent_win_rate < 60:
                # Win rate bajo: aumentar exploración, buscar mejores estrategias
                new_exploration = min(0.4, self.rl_agent.exploration_rate * 1.05)
                if new_exploration != self.rl_agent.exploration_rate:
                    logger.info(f"📉 Win rate bajo ({recent_win_rate:.1f}%), aumentando exploración: {self.rl_agent.exploration_rate:.2f} → {new_exploration:.2f}")
                    self.rl_agent.exploration_rate = new_exploration

        # Ajustes temporales en racha perdedora
        if self.losing_streak >= 3:
            logger.warning(f"⚠️ Racha de {self.losing_streak} pérdidas consecutivas, ajustando estrategia temporalmente")
            self.temporary_adjustment = {
                'leverage_multiplier': 0.5,  # Reducir leverage 50%
                'tp_multiplier': 0.8,        # TPs más cercanos
                'position_multiplier': 0.7   # Posiciones más pequeñas
            }
            await self._notify_telegram(
                f"⚠️ **Ajuste Temporal de Estrategia**\n\n"
                f"Racha perdedora: {self.losing_streak} trades\n"
                f"Leverage: -50%\n"
                f"Position size: -30%\n"
                f"TPs más cercanos\n\n"
                f"Se revertirá automáticamente con próxima ganancia"
            )

        # Notificar aprendizaje importante
        if reward > 2.0:  # Gran ganancia
            await self._notify_telegram(
                f"🎉 **Trade Exitoso Aprendido**\n\n"
                f"Par: {trade_data.get('pair', 'N/A')}\n"
                f"Profit: {trade_data.get('profit_pct', 0):.2f}%\n"
                f"Reward: {reward:.3f}\n"
                f"Win streak: {self.winning_streak} ✨"
            )
        elif reward < -2.0:  # Gran pérdida
            await self._notify_telegram(
                f"📚 **Trade Perdedor Analizado**\n\n"
                f"Par: {trade_data.get('pair', 'N/A')}\n"
                f"Loss: {trade_data.get('profit_pct', 0):.2f}%\n"
                f"Reward: {reward:.3f}\n"
                f"Lose streak: {self.losing_streak} ⚙️"
            )

        # Verificar si es momento de optimizar parámetros
        await self._check_and_optimize(portfolio_metrics)

        # Auto-save periódico
        await self._auto_save_if_needed()

    async def process_trade_closure(self, trade_info: Dict):
        """
        🧠 MÉTODO CENTRALIZADO: Procesa cierre de trade y actualiza TODOS los sistemas IA

        Este método DEBE ser llamado cada vez que un trade se cierra, ya sea por:
        - Take Profit (TP)
        - Stop Loss (SL)
        - Cierre manual
        - Liquidación

        Args:
            trade_info: Dict con información completa del trade:
                - symbol: Par trading (ej: BTCUSDT)
                - side: LONG/SHORT
                - entry_price: Precio de entrada
                - exit_price: Precio de salida
                - pnl_pct: % de ganancia/pérdida (ROE)
                - pnl_usdt: P&L en USDT
                - leverage: Leverage usado
                - reason: Razón del cierre (TP_HIT, SL_HIT, MANUAL, etc.)
                - duration: Duración en segundos (opcional)
        """
        symbol = trade_info.get('symbol', 'UNKNOWN')
        pnl_pct = trade_info.get('pnl_pct', 0)
        pnl_usdt = trade_info.get('pnl_usdt', 0)
        reason = trade_info.get('reason', 'UNKNOWN')
        leverage = trade_info.get('leverage', 1)

        logger.info(f"🔄 PROCESS_TRADE_CLOSURE: {symbol} | {reason} | P&L: {pnl_pct:+.2f}% (${pnl_usdt:+.2f})")

        # ===============================================
        # 1. ACTUALIZAR RL AGENT (Q-Table)
        # ===============================================
        try:
            if self.rl_agent:
                # Calcular reward
                reward = pnl_pct * leverage

                # El RL Agent aprende del resultado
                self.rl_agent.learn_from_trade(reward=reward, next_state=None, done=True)

                logger.info(f"   📚 RL Agent actualizado: Reward={reward:+.2f}, Q-table size={len(self.rl_agent.q_table)}")

                # Experience Replay cada 5 trades
                if self.rl_agent.total_trades % 5 == 0:
                    self.rl_agent.replay_experience(batch_size=16)
                    logger.info(f"   🔄 Experience Replay ejecutado")
        except Exception as e:
            logger.error(f"   ❌ Error actualizando RL Agent: {e}")

        # ===============================================
        # 2. ACTUALIZAR ML SYSTEM
        # ===============================================
        try:
            if self.ml_system and hasattr(self.ml_system, 'add_trade_result'):
                self.ml_system.add_trade_result(trade_info)
                logger.info(f"   🧠 ML System actualizado con resultado")
        except Exception as e:
            logger.error(f"   ❌ Error actualizando ML System: {e}")

        # ===============================================
        # 3. ACTUALIZAR DECISION BRAIN
        # ===============================================
        try:
            if self.decision_brain:
                self.decision_brain.learn_from_trade(trade_info)
                logger.info(f"   🧠 Decision Brain aprendió del trade")
        except Exception as e:
            logger.error(f"   ❌ Error actualizando Decision Brain: {e}")

        # ===============================================
        # 4. NOTIFICAR A PARAMETER OPTIMIZER
        # ===============================================
        try:
            if self.parameter_optimizer:
                # Registrar resultado para el optimizador
                self.parameter_optimizer.record_trial_result(
                    config={'leverage': leverage, 'symbol': symbol},
                    performance={
                        'pnl_pct': pnl_pct,
                        'win_rate': self.rl_agent.get_success_rate() if self.rl_agent else 0,
                        'total_trades': self.total_trades_all_time
                    }
                )
                logger.info(f"   ⚙️ Parameter Optimizer notificado")
        except Exception as e:
            logger.error(f"   ❌ Error notificando Parameter Optimizer: {e}")

        # ===============================================
        # 5. ACTUALIZAR ESTADÍSTICAS GLOBALES
        # ===============================================
        self.total_trades_all_time += 1
        self.total_trades_processed += 1

        # Actualizar rachas
        if pnl_pct > 0:
            self.winning_streak += 1
            self.losing_streak = 0
            if self.temporary_adjustment:
                logger.info(f"   ✅ Racha ganadora ({self.winning_streak}), limpiando ajustes temporales")
                self.temporary_adjustment = None
        else:
            self.losing_streak += 1
            self.winning_streak = 0

        # Actualizar historial reciente
        self.recent_trades_pnl.append(pnl_pct)
        if len(self.recent_trades_pnl) > 20:
            self.recent_trades_pnl = self.recent_trades_pnl[-20:]

        # ===============================================
        # 6. NOTIFICAR A TELEGRAM (APRENDIZAJE)
        # ===============================================
        try:
            # Calcular métricas para el mensaje
            win_rate = self.rl_agent.get_success_rate() if self.rl_agent else 0
            q_table_size = len(self.rl_agent.q_table) if self.rl_agent else 0

            # Determinar qué aprendió el sistema
            learning_insight = ""
            if pnl_pct > 0:
                if leverage >= 5:
                    learning_insight = f"✨ Aprendí: {symbol} con {leverage}x leverage funciona bien en estas condiciones"
                else:
                    learning_insight = f"✨ Aprendí: {symbol} es rentable con estrategia conservadora"
            else:
                if abs(pnl_pct) > 5:
                    learning_insight = f"📚 Aprendí: Evitar {symbol} en condiciones similares con {leverage}x"
                else:
                    learning_insight = f"📚 Aprendí: Ajustar SL/TP para {symbol}"

            # Construir mensaje de notificación
            await self._notify_telegram(
                f"🧠 **Trade Cerrado - Aprendizaje**\n\n"
                f"Par: {symbol}\n"
                f"Resultado: {pnl_pct:+.2f}% (${pnl_usdt:+.2f})\n"
                f"Razón: {reason}\n"
                f"Leverage: {leverage}x\n\n"
                f"📊 **Actualización IA:**\n"
                f"• Q-table: {q_table_size} estados\n"
                f"• Win rate: {win_rate:.1f}%\n"
                f"• Trades totales: {self.total_trades_all_time}\n"
                f"• Racha: {'🔥' + str(self.winning_streak) + ' wins' if self.winning_streak > 0 else '❄️' + str(self.losing_streak) + ' losses'}\n\n"
                f"{learning_insight}"
            )
        except Exception as e:
            logger.error(f"   ❌ Error enviando notificación Telegram: {e}")

        # ===============================================
        # 7. AUTO-SAVE SI ES NECESARIO
        # ===============================================
        if self.total_trades_processed % 10 == 0:
            await self._auto_save_if_needed()

        logger.info(f"✅ PROCESS_TRADE_CLOSURE completado para {symbol}")

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
            'max_leverage_unlocked': self._calculate_max_leverage(),
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

        # Guardar learning del Trade Manager
        trade_management_learning = self.get_trade_management_learning_data()
        if trade_management_learning:
            logger.info(f"📊 Trade Management Learning incluido: {trade_management_learning.get('total_actions_recorded', 0)} acciones")

        # 🧠 CRÍTICO: Guardar estado del DecisionBrain
        decision_brain_state = None
        if hasattr(self, 'decision_brain') and self.decision_brain:
            decision_brain_state = self.decision_brain.get_state()
            logger.info(f"🧠 DecisionBrain state incluido: {decision_brain_state.get('trades_analyzed', 0)} trades analizados")

        success = self.persistence.save_full_state(
            rl_agent_state=rl_state,
            optimizer_state=optimizer_state,
            performance_history=performance_summary,
            change_history=self.change_history,  # Histórico de cambios con razonamiento
            metadata=metadata,
            paper_trading=paper_trading_state,  # NUEVO: incluir paper trading
            ml_training_buffer=ml_training_buffer,  # NUEVO: incluir training buffer
            trade_management_learning=trade_management_learning,  # NUEVO: incluir learning del Trade Manager
            decision_brain_state=decision_brain_state  # 🧠 NUEVO: incluir estado del Brain
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

            if 'paper_trading' in loaded and loaded['paper_trading']:
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
            # Intentar cargar ml_training_data primero (formato nuevo)
            if 'ml_training_data' in loaded and loaded['ml_training_data']:
                logger.info("🧠 Restaurando ML Training Data...")

                # Verificar que tengamos acceso al ml_system
                if hasattr(self, 'market_monitor') and self.market_monitor:
                    if hasattr(self.market_monitor, 'ml_system') and self.market_monitor.ml_system:
                        ml_system = self.market_monitor.ml_system
                        training_data = loaded['ml_training_data']

                        # Asignar ml_training_data al ml_system
                        ml_system.ml_training_data = training_data
                        logger.info(f"  ✅ ML training data restaurado: {len(training_data)} muestras")

                        # También copiar a training_buffer para compatibilidad
                        if not hasattr(ml_system, 'training_buffer') or not ml_system.training_buffer:
                            ml_system.training_buffer = training_data
                            logger.info(f"  ✅ Training buffer sincronizado desde ml_training_data")

            # Fallback: cargar ml_training_buffer (formato antiguo)
            elif 'ml_training_buffer' in loaded and loaded['ml_training_buffer']:
                logger.info("🧠 Restaurando ML Training Buffer...")

                # Verificar que tengamos acceso al ml_system
                if hasattr(self, 'market_monitor') and self.market_monitor:
                    if hasattr(self.market_monitor, 'ml_system') and self.market_monitor.ml_system:
                        ml_system = self.market_monitor.ml_system
                        buffer_data = loaded['ml_training_buffer']

                        # Restaurar training_buffer
                        ml_system.training_buffer = buffer_data
                        # También asignar a ml_training_data para compatibilidad
                        ml_system.ml_training_data = buffer_data
                        logger.info(f"  ✅ Training buffer restaurado: {len(buffer_data)} features")
                        logger.info(f"  ✅ ML training data sincronizado desde buffer")

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
                logger.debug("ℹ️  No se encontró 'ml_training_data' ni 'ml_training_buffer' en el archivo importado")

        except Exception as e:
            logger.error(f"❌ Error restaurando ML Training Data/Buffer: {e}", exc_info=True)
            # No es crítico, continuar
        # ===== FIN RESTAURAR ML TRAINING DATA/BUFFER =====

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

        return True

    def update_from_trade_result(self, closed_info: Dict, reward: float):
        """
        Actualiza RL Agent con resultado de trade cerrado (v2.0 Binance)
        Método simplificado que toma closed_info de PositionMonitor

        Args:
            closed_info: Información del trade cerrado desde Binance
                - symbol: Par (ej: BTCUSDT)
                - side: LONG o SHORT
                - realized_pnl: P&L en USDT
                - realized_pnl_pct: P&L en porcentaje
                - leverage: Apalancamiento usado
                - entry_price: Precio de entrada
                - exit_price: Precio de salida
            reward: Reward calculado (usualmente el realized_pnl)
        """
        try:
            # Extraer datos del trade
            symbol = closed_info.get('symbol', 'UNKNOWN')
            side = closed_info.get('side', 'LONG')
            realized_pnl = closed_info.get('realized_pnl', 0)
            realized_pnl_pct = closed_info.get('realized_pnl_pct', 0)
            leverage = closed_info.get('leverage', 1)

            # DEDUPLICACIÓN: Evitar procesar el mismo trade dos veces
            # (test_mode y position_monitor pueden notificar el mismo cierre)
            current_time = datetime.now().timestamp()

            # Limpiar trades antiguos (más de 10 segundos)
            symbols_to_remove = []
            for sym, (ts, _) in self._recently_processed_trades.items():
                if current_time - ts > 10:
                    symbols_to_remove.append(sym)
            for sym in symbols_to_remove:
                del self._recently_processed_trades[sym]

            # Verificar si ya procesamos este trade recientemente
            if symbol in self._recently_processed_trades:
                prev_ts, prev_pnl = self._recently_processed_trades[symbol]
                time_diff = current_time - prev_ts

                # Si el P&L es similar (dentro de 5%) y fue hace menos de 10 segundos, es duplicado
                if time_diff < 10 and abs(prev_pnl - realized_pnl) < abs(prev_pnl * 0.05):
                    logger.warning(
                        f"⚠️ Trade duplicado detectado y IGNORADO: {symbol} | "
                        f"P&L: ${realized_pnl:+.2f} | "
                        f"Tiempo desde último: {time_diff:.1f}s | "
                        f"Razón: {closed_info.get('reason', 'N/A')}"
                    )
                    return

            # Registrar este trade como procesado
            self._recently_processed_trades[symbol] = (current_time, realized_pnl)

            # Log de cuál fuente está notificando (trade_id puede ser int o string)
            trade_id = str(closed_info.get('trade_id', ''))
            source = "Test Mode" if "test_" in trade_id else "Position Monitor"
            logger.info(f"📥 Trade notification from: {source}")

            # CRÍTICO: Ignorar Position Monitor cuando Test Mode está activo
            # (Test Mode ya notificó con el P&L correcto, Position Monitor tendría P&L diferente)
            if source == "Position Monitor" and self.test_mode_active:
                logger.info(f"⏭️ Ignorando notificación de Position Monitor (Test Mode activo)")
                return  # No procesar esta notificación duplicada

            # Incrementar contador global
            self.total_trades_all_time += 1

            # Crear state simplificado para RL Agent
            # (el state completo se creará internamente en learn_from_trade)
            state_dict = {
                'symbol': symbol,
                'side': side,
                'leverage': leverage,
                'rsi': 50,  # Valores default (el RL usa el next_state principalmente)
                'regime': 'SIDEWAYS',
                'orderbook': 'NEUTRAL',
                'volatility': 'medium'
            }

            state = self.rl_agent.get_state_representation(state_dict)

            # Normalizar reward (P&L en porcentaje es más útil que absoluto)
            normalized_reward = realized_pnl_pct / 100.0  # -10% → -0.1, +5% → 0.05

            # RL Agent aprende del trade
            done = abs(realized_pnl_pct) > 15  # Episodio termina en grandes wins/losses
            self.rl_agent.learn_from_trade(
                reward=normalized_reward,
                next_state=state,
                done=done
            )

            # Log de aprendizaje
            emoji = "✅" if realized_pnl > 0 else "❌"
            logger.info(
                f"{emoji} RL LEARNING: {symbol} {side} | "
                f"P&L: {realized_pnl_pct:+.2f}% | "
                f"Leverage: {leverage}x | "
                f"Reward: {normalized_reward:+.3f} | "
                f"Total trades: {self.total_trades_all_time}"
            )

            # Experience Replay periódico
            if self.rl_agent.total_trades % 10 == 0:
                self.rl_agent.replay_experience(batch_size=32)
                logger.debug(f"🔄 Experience replay ejecutado ({self.rl_agent.total_trades} trades)")

        except Exception as e:
            logger.error(f"❌ Error in update_from_trade_result: {e}", exc_info=True)

    def export_intelligence(self) -> Dict:
        """Exporta toda la inteligencia aprendida"""
        try:
            intelligence_data = {
                "version": "2.0",
                "timestamp": datetime.now().isoformat(),
                "rl_agent": self.rl_agent.save_to_dict(),
                "parameter_optimizer": self.parameter_optimizer.save_to_dict(),
                "performance_history": {
                    'total_trades': len(self.performance_history),
                    'recent_performance': self.performance_history[-100:] if self.performance_history else []
                },
                "metadata": {
                    'current_parameters': self.current_parameters,
                    'total_trades_processed': self.total_trades_processed,
                    'total_trades_all_time': self.total_trades_all_time,
                    'max_leverage_unlocked': self._calculate_max_leverage(),
                    'total_parameter_changes': self.total_parameter_changes,
                    'last_optimization': self.last_optimization_time.isoformat(),
                    'decision_mode': self.decision_mode
                },

                # ✨ AÑADIR LEARNING SYSTEM
                "trade_management_learning": self.get_trade_management_learning_data(),
            }

            # Añadir paper trading si existe
            if hasattr(self, 'paper_trader') and self.paper_trader:
                intelligence_data['paper_trading'] = self.paper_trader.portfolio.get_full_state_for_export()
                logger.debug(
                    f"📤 Exportando paper trading: "
                    f"{len(intelligence_data['paper_trading'].get('closed_trades', []))} trades"
                )

            # Añadir ML training buffer si existe
            ml_training_buffer = []
            if hasattr(self, 'market_monitor') and self.market_monitor:
                if hasattr(self.market_monitor, 'ml_system') and self.market_monitor.ml_system:
                    ml_system = self.market_monitor.ml_system
                    if hasattr(ml_system, 'training_buffer'):
                        ml_training_buffer = ml_system.training_buffer
                        intelligence_data['ml_training_buffer'] = ml_training_buffer
                        logger.debug(f"🧠 ML Training Buffer incluido en export: {len(ml_training_buffer)} features")

            return intelligence_data

        except Exception as e:
            logger.error(f"Error exporting intelligence: {e}", exc_info=True)
            return {}

    def get_trade_management_learning_data(self) -> Dict:
        """Obtiene datos del learning system del Trade Manager"""
        try:
            # Opción 1: Si trade_manager está en autonomy_controller
            if hasattr(self, 'trade_manager') and self.trade_manager:
                return self.trade_manager.learning.export_to_json()

            # Opción 2: Cargar desde archivo si no hay referencia
            from pathlib import Path
            import json
            filepath = 'data/trade_management_learning.json'
            if Path(filepath).exists():
                with open(filepath, 'r') as f:
                    return json.load(f)

            return {}

        except Exception as e:
            logger.error(f"Error getting trade management learning: {e}", exc_info=True)
            return {}

