"""
Trading Signal Bot - Main Entry Point
Monitors cryptocurrency pairs and sends trading signals via Telegram
"""
import asyncio
import logging
import sys
import warnings
from pathlib import Path
import threading
import uvicorn

# Suprimir warnings de NumPy sobre operaciones con NaN
# Estos son comunes en cálculos de indicadores técnicos y no afectan funcionalidad
warnings.filterwarnings('ignore', message='invalid value encountered in subtract')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.market_monitor import MarketMonitor
from config import config
from datetime import datetime
from src.api import app, set_bot_instances

# Import autonomous AI system
if config.ENABLE_AUTONOMOUS_MODE:
    from src.autonomous.autonomy_controller import AutonomyController
    from src.telegram_commands import TelegramCommands

# Import test mode
from test_mode import TestMode


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
        trading_mode = f"{'🧪 TESTNET' if config.BINANCE_TESTNET else '🔴 LIVE'}"
        auto_trade_status = "✅ Activo" if config.AUTO_TRADE else "❌ Inactivo"
        flash_signals_status = "✅ Activas" if config.ENABLE_FLASH_SIGNALS else "❌ Inactivas"
        autonomous_status = "✅ MODO AUTÓNOMO ACTIVO" if config.ENABLE_AUTONOMOUS_MODE else "❌ Modo manual"

        # Obtener balance real de Binance (v2.0)
        balance = "Obteniendo..."
        try:
            if hasattr(monitor, 'binance_client') and monitor.binance_client:
                balance_info = monitor.binance_client.get_balance()
                usdt_balance = next((b for b in balance_info if b['asset'] == 'USDT'), None)
                if usdt_balance:
                    balance = f"${float(usdt_balance['availableBalance']):,.2f} USDT"
        except:
            balance = "No disponible"

        # Contar pares
        total_pairs = len(config.TRADING_PAIRS)
        main_pairs = f"{config.TRADING_PAIRS[0]}, {config.TRADING_PAIRS[1]}"
        additional_pairs = total_pairs - 2

        # Construir mensaje
        message = (
            "🤖 **Bot de Trading v2.0 Iniciado**\n\n"
            f"📊 Monitoreando: {main_pairs} y {additional_pairs} más\n"
            f"⏱️ Intervalo: {config.CHECK_INTERVAL}s\n"
            f"📈 Timeframe conservador: {config.TIMEFRAME} (1h/4h/1d)\n"
            f"⚡ Señales flash: {flash_signals_status} ({config.FLASH_TIMEFRAME})\n"
            f"💰 Binance Futures: {trading_mode} | Balance: {balance}\n"
            f"🔄 Auto-Trading: {auto_trade_status} | Leverage: {config.DEFAULT_LEVERAGE}x\n"
            f"🧠 Machine Learning: {ml_status} ({ml_accuracy} accuracy)\n"
            f"📰 Sentiment Analysis: {sentiment_status}\n"
            f"📚 Order Book: ✅ Activo\n"
            f"🎯 Market Regime: ✅ Activo\n"
            f"🤖 Sistema Autónomo: {autonomous_status}\n"
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
                initial_balance=50000.0,  # Balance inicial para backtesting histórico
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
    import os

    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("Trading Signal Bot Starting...")
    logger.info("=" * 60)

    # ========== VERIFICACIÓN DE ESTADO INICIAL ==========
    logger.info("")
    logger.info("=" * 60)
    logger.info("🔍 VERIFICACIÓN DE ESTADO INICIAL")
    logger.info("=" * 60)

    # Verificar Paper Trading (si existe la variable)
    paper_trading = getattr(config, 'PAPER_TRADING', False)
    logger.info(f"📊 Paper Trading: {'HABILITADO ⚠️' if paper_trading else 'DESHABILITADO ✅'}")
    logger.info(f"💰 Auto Trade: {'HABILITADO ✅' if config.AUTO_TRADE else 'DESHABILITADO ❌'}")
    logger.info(f"🎚️ Threshold Base: {config.CONSERVATIVE_THRESHOLD}")
    logger.info(f"🌐 Binance Mode: {'TESTNET ⚠️' if config.BINANCE_TESTNET else 'LIVE ✅'}")

    # Verificar si hay datos previos de paper trading
    rl_state_exists = os.path.exists('data/autonomous/rl_agent_state.json')
    threshold_exp_exists = os.path.exists('data/threshold_experiences.json')
    trade_mgmt_exists = os.path.exists('data/trade_management_learning.json')

    if rl_state_exists or threshold_exp_exists or trade_mgmt_exists:
        logger.warning("⚠️ ENCONTRADOS DATOS PREVIOS:")
        if rl_state_exists:
            logger.warning("   - data/autonomous/rl_agent_state.json (RL Agent state)")
        if threshold_exp_exists:
            logger.warning("   - data/threshold_experiences.json (Threshold experiences)")
        if trade_mgmt_exists:
            logger.warning("   - data/trade_management_learning.json (Trade Manager learning)")
        logger.warning("   Estos datos serán usados para continuar el aprendizaje")
    else:
        logger.info("✅ ESTADO LIMPIO: Empezando desde cero (sin datos previos)")

    logger.info("=" * 60)
    logger.info("")

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

    # Initialize variables before try block (para evitar UnboundLocalError en exception handlers)
    autonomy_controller = None
    telegram_commands = None
    trade_manager = None

    # Initialize and start market monitor
    try:
        monitor = MarketMonitor()

        # v2.0: Iniciar Position Monitor si está disponible
        if hasattr(monitor, 'position_monitor') and monitor.position_monitor:
            try:
                logger.info("🚀 Iniciando Position Monitor...")

                # CRÍTICO: Verificar que hay un loop de eventos activo
                try:
                    loop = asyncio.get_running_loop()
                    logger.info(f"✅ Loop de eventos detectado: {loop}")
                except RuntimeError:
                    logger.warning("⚠️ No hay loop running, usando get_event_loop()...")
                    loop = asyncio.get_event_loop()

                # Iniciar monitoreo en background
                task = monitor.position_monitor.start_background_monitoring()

                if task:
                    logger.info("✅ Position Monitor iniciado - monitoreando posiciones cada 10s")
                    logger.info(f"   Task ID: {id(task)}")

                    # Verificar después de 2 segundos que está corriendo
                    await asyncio.sleep(2)
                    if monitor.position_monitor._running:
                        logger.info("✅ Position Monitor confirmado en ejecución")
                    else:
                        logger.error("❌ Position Monitor NO está corriendo después de iniciar")
                else:
                    logger.error("❌ start_background_monitoring() retornó None")

            except Exception as e:
                logger.error(f"❌ Error iniciando Position Monitor: {e}", exc_info=True)
                logger.error("   Position Monitor NO estará activo")
        else:
            logger.warning("⚠️ Position Monitor no disponible (verifica credenciales de Binance)")

        # 🤖 v2.1: Iniciar Trade Manager (gestión inteligente de trades)
        if (hasattr(monitor, 'position_monitor') and monitor.position_monitor and
            hasattr(monitor, 'futures_trader') and monitor.futures_trader):
            try:
                from src.autonomous.trade_manager import TradeManager

                logger.info("🤖 Inicializando Trade Manager...")
                trade_manager = TradeManager(
                    position_monitor=monitor.position_monitor,
                    futures_trader=monitor.futures_trader,
                    rl_agent=None,  # Será asignado después si está disponible
                    ml_system=monitor.ml_system if hasattr(monitor, 'ml_system') else None,
                    market_analyzer=monitor
                )

                # CRÍTICO: Iniciar monitoreo en el loop actual
                logger.info("🔄 Iniciando Trade Manager en background...")
                loop = asyncio.get_running_loop()
                trade_manager_task = loop.create_task(trade_manager.start_monitoring())

                # Esperar confirmación
                await asyncio.sleep(1)

                logger.info("✅ Trade Manager activo - Gestión inteligente de trades en tiempo real")
                logger.info("   - Breakeven protection: 1.5%")
                logger.info("   - Trailing stop: 3.0%")
                logger.info("   - Partial TP: 4.0%")
                logger.info("   - Protección adversa: -1.5%")
                logger.info(f"   Task ID: {id(trade_manager_task)}")

            except Exception as e:
                logger.error(f"❌ Error iniciando Trade Manager: {e}", exc_info=True)
                logger.error("   Trade Manager NO estará activo")
        else:
            reason = []
            if not hasattr(monitor, 'position_monitor') or not monitor.position_monitor:
                reason.append("Position Monitor no disponible")
            if not hasattr(monitor, 'futures_trader') or not monitor.futures_trader:
                reason.append("Futures Trader no disponible")
            logger.warning(f"⚠️ Trade Manager no inicializado: {', '.join(reason)}")

        # Initialize Autonomous AI System if enabled
        if config.ENABLE_AUTONOMOUS_MODE:
            logger.info("🤖 Inicializando Sistema Autónomo - CONTROL ABSOLUTO")
            autonomy_controller = AutonomyController(
                telegram_notifier=monitor.notifier,
                auto_save_interval_minutes=config.AUTONOMOUS_AUTO_SAVE_INTERVAL,
                optimization_check_interval_hours=config.AUTONOMOUS_OPTIMIZATION_INTERVAL,
                min_trades_before_optimization=config.AUTONOMOUS_MIN_TRADES_BEFORE_OPT
            )

            # Pass Binance integration to autonomy controller (v2.0)
            if hasattr(monitor, 'binance_client'):
                autonomy_controller.binance_client = monitor.binance_client
                logger.info("✅ Binance client asignado al autonomy_controller")
            if hasattr(monitor, 'futures_trader'):
                autonomy_controller.futures_trader = monitor.futures_trader
                logger.info("✅ Futures trader asignado al autonomy_controller")
            if hasattr(monitor, 'position_monitor'):
                autonomy_controller.position_monitor = monitor.position_monitor
                logger.info("✅ Position monitor asignado al autonomy_controller")
                # CRÍTICO: Asignar autonomy_controller a position_monitor para evitar duplicados
                monitor.position_monitor.autonomy_controller = autonomy_controller
                logger.info("✅ Autonomy controller asignado al position_monitor (evitar duplicados)")

            # Pass references ANTES de initialize (para que _restore_from_state tenga acceso)
            monitor.autonomy_controller = autonomy_controller
            autonomy_controller.market_monitor = monitor  # Para acceso a ml_system

            # Ahora sí, inicializar (esto cargará la inteligencia guardada)
            await autonomy_controller.initialize()
            logger.info("✅ Sistema Autónomo activo - IA tiene control total")

            # 🧠 CRÍTICO: Inicializar DecisionBrain con todos los componentes
            logger.info("🧠 Configurando DecisionBrain con componentes...")
            autonomy_controller.set_components(
                ml_system=monitor.ml_system if hasattr(monitor, 'ml_system') else None,
                trade_manager=trade_manager,
                feature_aggregator=monitor.feature_aggregator if hasattr(monitor, 'feature_aggregator') else None,
                sentiment_analyzer=monitor.sentiment_analyzer if hasattr(monitor, 'sentiment_analyzer') else None,
                regime_detector=monitor.regime_detector if hasattr(monitor, 'regime_detector') else None,
                orderbook_analyzer=monitor.orderbook_analyzer if hasattr(monitor, 'orderbook_analyzer') else None
            )
            logger.info("✅ DecisionBrain inicializado con todos los servicios")

            # Asignar RL agent al Trade Manager si existe
            if trade_manager and hasattr(autonomy_controller, 'rl_agent') and autonomy_controller.rl_agent:
                trade_manager.rl_agent = autonomy_controller.rl_agent
                logger.info("✅ RL Agent asignado al Trade Manager")
                logger.info(f"   Tipo: {type(autonomy_controller.rl_agent).__name__}")

                # RE-VERIFICAR
                logger.info("🔄 Verificando integración del RL Agent en Trade Manager...")
                if hasattr(trade_manager, 'rl_agent') and trade_manager.rl_agent:
                    logger.info("✅ RL Agent CONFIRMADO disponible en Trade Manager")
                else:
                    logger.error("❌ ERROR: RL Agent NO se asignó correctamente")

            # Asignar trade_manager al autonomy_controller para export
            if trade_manager:
                autonomy_controller.trade_manager = trade_manager
                logger.info("✅ Trade Manager asignado al autonomy_controller (para export)")

            # Initialize Test Mode
            test_mode = TestMode(
                futures_trader=monitor.futures_trader if hasattr(monitor, 'futures_trader') else None,
                position_monitor=monitor.position_monitor if hasattr(monitor, 'position_monitor') else None,
                notifier=monitor.notifier if hasattr(monitor, 'notifier') else None,
                autonomy_controller=autonomy_controller  # CRÍTICO: Para notificar P&L correcto al RL Agent
            )
            logger.info("🧪 Test Mode inicializado (con autonomy_controller para RL Agent)")

            # Initialize Telegram Commands Handler
            telegram_commands = TelegramCommands(
                autonomy_controller=autonomy_controller,
                telegram_token=config.TELEGRAM_BOT_TOKEN,
                chat_id=config.TELEGRAM_CHAT_ID,
                market_monitor=monitor,  # Para acceso al ML System
                test_mode=test_mode,  # Para comandos de prueba
                trade_manager=trade_manager if 'trade_manager' in locals() else None  # Para learning stats
            )
            monitor.telegram_commands = telegram_commands
            await telegram_commands.start_command_listener()
            logger.info("📱 Telegram Commands activos: /export, /import, /status, /stats, /params, /train_ml, /test_start, /test_stop, /test_status")

        # ===== INICIAR API PARA DASHBOARD =====
        try:
            set_bot_instances(monitor, autonomy_controller)
            api_thread = threading.Thread(
                target=lambda: uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning"),
                daemon=True
            )
            api_thread.start()
            logger.info("✅ API iniciada en puerto 8000")
        except Exception as e:
            logger.error(f"⚠️ Error iniciando API: {e}")

        # Run historical training if enabled (pre-train ML model)
        if config.ENABLE_HISTORICAL_TRAINING:
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
        # Shutdown trade manager if active
        if trade_manager:
            trade_manager.stop_monitoring()
        # Shutdown telegram commands if active
        if telegram_commands:
            await telegram_commands.stop_command_listener()
        # Shutdown autonomous controller if active
        if autonomy_controller:
            await autonomy_controller.shutdown()

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        # Shutdown trade manager if active
        if trade_manager:
            trade_manager.stop_monitoring()
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
