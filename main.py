"""
Trading Signal Bot - Main Entry Point
Monitors cryptocurrency pairs and sends trading signals via Telegram
"""
import asyncio
import logging
import sys
import warnings
from pathlib import Path

# Suprimir warnings de NumPy sobre operaciones con NaN
# Estos son comunes en cálculos de indicadores técnicos y no afectan funcionalidad
warnings.filterwarnings('ignore', message='invalid value encountered in subtract')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.market_monitor import MarketMonitor
from config import config
from datetime import datetime

# Import autonomous AI system
if config.ENABLE_AUTONOMOUS_MODE:
    from src.autonomous.autonomy_controller import AutonomyController
    from src.telegram_commands import TelegramCommands

# Import GPT Brain for advanced AI reasoning
if hasattr(config, 'ENABLE_GPT_BRAIN') and config.ENABLE_GPT_BRAIN:
    from src.llm.gpt_brain import GPTBrain
    from src.llm.gpt_data_provider import GPTDataProvider


async def send_bot_status_message(monitor):
    """
    Envía mensaje completo del status del bot a Telegram

    Args:
        monitor: Instancia de MarketMonitor con todos los componentes
    """
    logger = logging.getLogger(__name__)

    try:
        # Verificar status de cada componente
        ml_status = "✅ Activo"
        ml_accuracy = "N/A"

        if hasattr(monitor, 'ml_system') and monitor.ml_system:
            predictor = monitor.ml_system.predictor
            if predictor:
                model_info = predictor.get_model_info()
                if model_info.get('available'):
                    ml_accuracy = f"{model_info.get('metrics', {}).get('test_accuracy', 0) * 100:.1f}%"
                else:
                    ml_status = "⚠️ Sin entrenar"
            else:
                ml_status = "❌ Inactivo"
        else:
            ml_status = "❌ Inactivo"

        sentiment_status = "✅ Activo" if config.ENABLE_SENTIMENT_ANALYSIS else "❌ Inactivo"
        paper_trading_status = "✅ Activo" if config.ENABLE_PAPER_TRADING else "❌ Inactivo"
        flash_signals_status = "✅ Activas" if config.ENABLE_FLASH_SIGNALS else "❌ Inactivas"
        autonomous_status = "✅ MODO AUTÓNOMO ACTIVO" if config.ENABLE_AUTONOMOUS_MODE else "❌ Modo manual"
        gpt_brain_status = "✅ Activo" if hasattr(config, 'ENABLE_GPT_BRAIN') and config.ENABLE_GPT_BRAIN else "❌ Inactivo"
        gpt_model = getattr(config, 'GPT_MODEL', 'N/A') if hasattr(config, 'ENABLE_GPT_BRAIN') and config.ENABLE_GPT_BRAIN else "N/A"

        # Obtener balance de paper trading
        balance = "$50,000 USDT"
        if hasattr(monitor, 'ml_system') and monitor.ml_system:
            if hasattr(monitor.ml_system, 'paper_trader') and monitor.ml_system.paper_trader:
                portfolio = monitor.ml_system.paper_trader.portfolio
                balance = f"${portfolio.get_equity():,.2f} USDT"

        # Contar pares
        total_pairs = len(config.TRADING_PAIRS)
        main_pairs = f"{config.TRADING_PAIRS[0]}, {config.TRADING_PAIRS[1]}"
        additional_pairs = total_pairs - 2

        # Construir mensaje
        message = (
            "🤖 **Bot de Señales Iniciado**\n\n"
            f"📊 Monitoreando: {main_pairs} y {additional_pairs} más\n"
            f"⏱️ Intervalo: {config.CHECK_INTERVAL}s\n"
            f"📈 Timeframe conservador: {config.TIMEFRAME} (1h/4h/1d)\n"
            f"⚡ Señales flash: {flash_signals_status} ({config.FLASH_TIMEFRAME})\n"
            f"💰 Paper Trading: {paper_trading_status} ({balance})\n"
            f"🧠 Machine Learning: {ml_status} ({ml_accuracy} accuracy)\n"
            f"📰 Sentiment Analysis: {sentiment_status}\n"
            f"📚 Order Book: ✅ Activo\n"
            f"🎯 Market Regime: ✅ Activo\n"
            f"🤖 Sistema Autónomo: {autonomous_status}\n"
            f"🧠 GPT Brain: {gpt_brain_status} ({gpt_model})\n"
            f"📍 Reporte diario: 9 PM CDMX"
        )

        # Enviar mensaje
        if monitor.notifier:
            await monitor.notifier.send_status_message(message)
            logger.info("✅ Mensaje de status enviado a Telegram")

    except Exception as e:
        logger.warning(f"No se pudo enviar mensaje de status: {e}")


def setup_logging():
    """
    Configure logging for the application
    """
    # Create logs directory if it doesn't exist
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.LOG_FILE),
            logging.StreamHandler(sys.stdout)
        ]
    )


async def run_historical_training(telegram_bot=None):
    """
    Pre-entrena modelo ML con datos históricos si no existe modelo

    Args:
        telegram_bot: Instancia de TelegramBot para notificaciones

    Returns:
        True si entrenamiento fue exitoso o no era necesario
    """
    logger = logging.getLogger(__name__)

    # Verificar si historical training está habilitado
    if not config.ENABLE_HISTORICAL_TRAINING:
        logger.info("📊 Historical training deshabilitado (ENABLE_HISTORICAL_TRAINING=false)")
        return True

    # Verificar si ya existe modelo y si debemos skipear
    model_file = Path('data/models/xgboost_model.pkl')
    if model_file.exists() and config.SKIP_HISTORICAL_IF_MODEL_EXISTS:
        logger.info("✅ Modelo ML existente encontrado - Skipping historical training")
        return True

    logger.info("\n" + "=" * 60)
    logger.info("🕐 INICIANDO ENTRENAMIENTO HISTÓRICO")
    logger.info("=" * 60)

    try:
        # Importar módulos necesarios
        from src.ml.historical_data_collector import HistoricalDataCollector
        from src.ml.backtester import Backtester
        from src.ml.backtest_analyzer import BacktestAnalyzer
        from src.ml.initial_trainer import InitialTrainer

        # 1. Descargar datos históricos
        logger.info("\n📥 FASE 1: Descargando datos históricos...")
        logger.info(f"   Periodo: {config.HISTORICAL_START_DATE} hasta {config.HISTORICAL_END_DATE}")
        logger.info(f"   Timeframes: {', '.join(config.HISTORICAL_TIMEFRAMES)}")
        logger.info(f"   Pares: {len(config.TRADING_PAIRS)}")
        logger.info("   (Esto puede tomar 10-30 minutos la primera vez...)")

        collector = HistoricalDataCollector(exchange_name=config.EXCHANGE_NAME)

        start_date = datetime.strptime(config.HISTORICAL_START_DATE, '%Y-%m-%d')
        end_date = datetime.strptime(config.HISTORICAL_END_DATE, '%Y-%m-%d')

        historical_data = collector.download_all_pairs(
            pairs=config.TRADING_PAIRS,
            timeframes=config.HISTORICAL_TIMEFRAMES,
            start_date=start_date,
            end_date=end_date,
            force_download=config.FORCE_HISTORICAL_DOWNLOAD
        )

        if not historical_data:
            logger.error("❌ No se pudieron descargar datos históricos")
            return False

        cache_info = collector.get_cache_info()
        logger.info(f"✅ Datos descargados: {cache_info['total_files']} archivos ({cache_info['total_size_mb']} MB)")

        # 2. Intentar cargar backtest guardado
        logger.info("\n🔍 Verificando si existe backtest previo...")
        date_range = (config.HISTORICAL_START_DATE, config.HISTORICAL_END_DATE)
        backtest_results = Backtester.load_backtest_results(expected_date_range=date_range)

        if backtest_results:
            logger.info("✅ Usando resultados de backtest guardado (ahorra ~5-10 min)")
        else:
            # 2b. Correr backtest si no existe guardado
            logger.info("\n🔄 FASE 2: Corriendo backtest histórico...")
            logger.info("   (Generando señales y simulando trades...)")

            backtester = Backtester(
                initial_balance=config.PAPER_TRADING_INITIAL_BALANCE,
                commission_rate=0.001,
                slippage_rate=0.0005,
                telegram_bot=telegram_bot  # NUEVO: Para notificaciones
            )

            backtest_results = await backtester.run_backtest(
                historical_data=historical_data,
                signal_type='both'  # Conservative + Flash
            )

        if len(backtest_results) < config.MIN_HISTORICAL_SAMPLES:
            logger.error(f"❌ Insuficientes señales históricas: {len(backtest_results)} (mínimo {config.MIN_HISTORICAL_SAMPLES})")
            return False

        # Guardar resultados
        backtester.save_results('backtest_results.json')

        # Analizar resultados
        logger.info("\n📊 FASE 3: Analizando resultados...")
        analyzer = BacktestAnalyzer(backtest_results)
        analyzer.print_summary()

        # 3. Entrenar modelo
        logger.info("\n🧠 FASE 4: Pre-entrenando modelo ML...")
        logger.info("   (Aplicando protecciones anti-overfitting...)")

        trainer = InitialTrainer(
            backtest_results=backtest_results,
            temporal_weight_recent=2.0,  # Datos recientes pesan 2x
            oos_months=2  # Últimos 2 meses para out-of-sample testing
        )

        training_result = trainer.train_with_validation()

        if not training_result.get('success'):
            logger.error(f"❌ Entrenamiento falló: {training_result.get('reason')}")
            return False

        # Guardar modelo
        trainer.save_model('data/models/xgboost_model.pkl')

        # Imprimir reporte
        report = trainer.get_training_report(training_result)
        logger.info(f"\n{report}")

        logger.info("\n" + "=" * 60)
        logger.info("✅ ENTRENAMIENTO HISTÓRICO COMPLETADO")
        logger.info("=" * 60 + "\n")

        return True

    except Exception as e:
        logger.error(f"❌ Error en entrenamiento histórico: {e}", exc_info=True)
        logger.info("ℹ️  El bot continuará sin modelo pre-entrenado")
        return True  # No es fatal, continuar de todos modos


async def main():
    """
    Main function to start the trading bot
    """
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("Trading Signal Bot Starting...")
    logger.info("=" * 60)

    # Verify configuration
    if not config.TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN not configured. Please set it in .env file")
        sys.exit(1)

    if not config.TELEGRAM_CHAT_ID:
        logger.error("TELEGRAM_CHAT_ID not configured. Please set it in .env file")
        sys.exit(1)

    logger.info(f"Exchange: {config.EXCHANGE_NAME}")
    logger.info(f"Trading Pairs: {', '.join(config.TRADING_PAIRS)}")
    logger.info(f"Check Interval: {config.CHECK_INTERVAL} seconds")
    logger.info(f"Timeframe: {config.TIMEFRAME}")

    # Initialize and start market monitor
    try:
        monitor = MarketMonitor()

        # Initialize Autonomous AI System if enabled
        autonomy_controller = None
        telegram_commands = None
        if config.ENABLE_AUTONOMOUS_MODE:
            logger.info("🤖 Inicializando Sistema Autónomo - CONTROL ABSOLUTO")
            autonomy_controller = AutonomyController(
                telegram_notifier=monitor.notifier,
                auto_save_interval_minutes=config.AUTONOMOUS_AUTO_SAVE_INTERVAL,
                optimization_check_interval_hours=config.AUTONOMOUS_OPTIMIZATION_INTERVAL,
                min_trades_before_optimization=config.AUTONOMOUS_MIN_TRADES_BEFORE_OPT
            )

            # CRÍTICO: Asignar paper_trader ANTES de initialize()
            # Esto permite que _restore_from_state() pueda restaurar paper trading correctamente
            if monitor.ml_system and hasattr(monitor.ml_system, 'paper_trader'):
                autonomy_controller.paper_trader = monitor.ml_system.paper_trader
                logger.info("✅ paper_trader asignado al autonomy_controller")
            else:
                logger.warning("⚠️ ml_system.paper_trader no disponible - paper trading no se restaurará")

            # Pass references ANTES de initialize (para que _restore_from_state tenga acceso)
            monitor.autonomy_controller = autonomy_controller
            autonomy_controller.market_monitor = monitor  # Para acceso a ml_system

            # Ahora sí, inicializar (esto cargará la inteligencia guardada)
            await autonomy_controller.initialize()
            logger.info("✅ Sistema Autónomo activo - IA tiene control total")

            # Initialize Telegram Commands Handler
            telegram_commands = TelegramCommands(
                autonomy_controller=autonomy_controller,
                telegram_token=config.TELEGRAM_BOT_TOKEN,
                chat_id=config.TELEGRAM_CHAT_ID,
                market_monitor=monitor  # Para acceso al ML System
            )
            monitor.telegram_commands = telegram_commands
            await telegram_commands.start_command_listener()
            logger.info("📱 Telegram Commands activos: /export, /import, /status, /stats, /params, /train_ml")

        # Initialize GPT Brain if enabled
        gpt_brain = None
        if hasattr(config, 'ENABLE_GPT_BRAIN') and config.ENABLE_GPT_BRAIN:
            if config.OPENAI_API_KEY:
                logger.info("🧠 Inicializando GPT Brain - Razonamiento Avanzado")

                # Callback para actualizar parámetros
                def update_param_callback(param: str, value) -> bool:
                    try:
                        setattr(config, param, value)
                        logger.info(f"GPT Brain updated {param} = {value}")
                        return True
                    except Exception as e:
                        logger.error(f"Failed to update {param}: {e}")
                        return False

                # Callback para notificaciones
                async def notify_callback(message: str):
                    if monitor.notifier:
                        await monitor.notifier.send_status_message(message)

                gpt_brain = GPTBrain(
                    api_key=config.OPENAI_API_KEY,
                    model=config.GPT_MODEL,
                    param_update_callback=update_param_callback,
                    notification_callback=notify_callback,
                    config=config
                )

                # Asignar GPT Brain al monitor y autonomy controller
                monitor.gpt_brain = gpt_brain
                if autonomy_controller:
                    autonomy_controller.gpt_brain = gpt_brain

                # Inicializar GPT Brain
                await gpt_brain.initialize()
                logger.info(f"✅ GPT Brain activo con modelo: {config.GPT_MODEL}")

                # Connect GPT Data Provider with ALL available data sources
                try:
                    # Get portfolio from ml_system if available
                    portfolio = None
                    ml_system = None
                    if hasattr(monitor, 'ml_system') and monitor.ml_system:
                        ml_system = monitor.ml_system
                        if hasattr(ml_system, 'paper_trader') and ml_system.paper_trader:
                            portfolio = ml_system.paper_trader.portfolio

                    # Create GPT Data Provider with all sources
                    data_provider = GPTDataProvider(
                        config=config,
                        feature_aggregator=getattr(monitor, 'feature_aggregator', None),
                        sentiment_analyzer=getattr(monitor, 'sentiment_system', None),
                        orderbook_analyzer=getattr(monitor, 'orderbook_analyzer', None),
                        news_collector=getattr(monitor.sentiment_system, 'news_collector', None) if hasattr(monitor, 'sentiment_system') else None,
                        market_monitor=monitor,
                        ml_system=ml_system,
                        rl_agent=getattr(autonomy_controller, 'rl_agent', None) if autonomy_controller else None,
                        portfolio=portfolio
                    )

                    # Connect to GPT Brain
                    gpt_brain.set_data_provider(data_provider)
                    logger.info("🔗 GPT Data Provider conectado - Arsenal completo disponible")

                except Exception as e:
                    logger.warning(f"⚠️ No se pudo conectar GPT Data Provider: {e}")

                # Agregar comandos GPT a Telegram si disponible
                if telegram_commands:
                    telegram_commands.gpt_brain = gpt_brain
            else:
                logger.warning("⚠️ GPT Brain habilitado pero OPENAI_API_KEY no configurada")

        # Run historical training if enabled (pre-train ML model)
        if config.ENABLE_PAPER_TRADING:
            success = await run_historical_training(telegram_bot=monitor.notifier)
            if not success:
                logger.warning("Historical training no completado, pero continuando...")

        # ✅ VALIDACIÓN COMPLETA DE SERVICIOS AL INICIO
        from src.startup_validator import run_startup_validation
        logger.info("")
        logger.info("🔍 Ejecutando validación completa de servicios...")
        validation_results = await run_startup_validation(monitor)
        logger.info("")

        # Iniciar monitoreo
        await monitor.start()

    except KeyboardInterrupt:
        logger.info("Received interrupt signal. Shutting down...")
        # Shutdown telegram commands if active
        if telegram_commands:
            await telegram_commands.stop_command_listener()
        # Shutdown autonomous controller if active
        if autonomy_controller:
            await autonomy_controller.shutdown()

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        # Shutdown telegram commands if active
        if telegram_commands:
            await telegram_commands.stop_command_listener()
        # Shutdown autonomous controller if active
        if autonomy_controller:
            await autonomy_controller.shutdown()
        sys.exit(1)

    logger.info("Bot stopped successfully")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
