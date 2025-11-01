"""
Trading Signal Bot - Main Entry Point
Monitors cryptocurrency pairs and sends trading signals via Telegram
"""
import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.market_monitor import MarketMonitor
from config import config
from datetime import datetime


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


def run_historical_training():
    """
    Pre-entrena modelo ML con datos históricos si no existe modelo

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

        # 2. Correr backtest
        logger.info("\n🔄 FASE 2: Corriendo backtest histórico...")
        logger.info("   (Generando señales y simulando trades...)")

        backtester = Backtester(
            initial_balance=config.PAPER_TRADING_INITIAL_BALANCE,
            commission_rate=0.001,
            slippage_rate=0.0005
        )

        backtest_results = backtester.run_backtest(
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

    # Run historical training if enabled (pre-train ML model)
    if config.ENABLE_PAPER_TRADING:
        success = run_historical_training()
        if not success:
            logger.warning("Historical training no completado, pero continuando...")

    # Initialize and start market monitor
    try:
        monitor = MarketMonitor()
        await monitor.start()

    except KeyboardInterrupt:
        logger.info("Received interrupt signal. Shutting down...")

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

    logger.info("Bot stopped successfully")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
