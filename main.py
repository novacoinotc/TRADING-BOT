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
