"""
Market Monitor Module
Continuously monitors trading pairs and analyzes them for signals
"""
import ccxt
import pandas as pd
import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict
from config import config
from src.technical_analysis import TechnicalAnalyzer
from src.telegram_bot import TelegramNotifier
from src.signal_tracker import SignalTracker
from src.daily_reporter import DailyReporter

logger = logging.getLogger(__name__)


class MarketMonitor:
    """
    Monitors market data and generates trading signals
    """

    def __init__(self):
        self.exchange_name = config.EXCHANGE_NAME
        self.exchange = self._initialize_exchange()
        self.analyzer = TechnicalAnalyzer()

        # Initialize signal tracker
        self.tracker = SignalTracker() if config.TRACK_SIGNALS else None

        # Initialize notifier with tracker
        self.notifier = TelegramNotifier(tracker=self.tracker)

        # Initialize daily reporter
        self.reporter = DailyReporter(self.tracker, self.notifier) if self.tracker else None

        self.trading_pairs = config.TRADING_PAIRS
        self.timeframe = config.TIMEFRAME
        self.check_interval = config.CHECK_INTERVAL
        self.is_running = False
        self.current_prices = {}  # Store current prices for tracking

    def _initialize_exchange(self) -> ccxt.Exchange:
        """
        Initialize exchange connection

        Returns:
            CCXT exchange instance
        """
        try:
            exchange_class = getattr(ccxt, self.exchange_name)
            exchange = exchange_class({
                'apiKey': config.EXCHANGE_API_KEY,
                'secret': config.EXCHANGE_API_SECRET,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',
                }
            })
            logger.info(f"Connected to {self.exchange_name}")
            return exchange

        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")
            raise

    async def fetch_ohlcv(self, pair: str) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data for a trading pair

        Args:
            pair: Trading pair (e.g., 'BTC/USDT')

        Returns:
            DataFrame with OHLCV data or None
        """
        try:
            # Fetch data
            ohlcv = self.exchange.fetch_ohlcv(
                pair,
                timeframe=self.timeframe,
                limit=100  # Fetch last 100 candles
            )

            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')

            return df

        except ccxt.BadSymbol:
            logger.warning(f"Symbol {pair} not available on {self.exchange_name}")
            return None
        except Exception as e:
            logger.error(f"Error fetching data for {pair}: {e}")
            return None

    async def analyze_pair(self, pair: str):
        """
        Analyze a trading pair and send signals if found

        Args:
            pair: Trading pair to analyze
        """
        try:
            logger.info(f"Analyzing {pair}...")

            # Fetch market data
            df = await self.fetch_ohlcv(pair)

            if df is None or len(df) < 50:
                logger.warning(f"Insufficient data for {pair}")
                return

            # Perform technical analysis
            analysis = self.analyzer.analyze(df)

            if analysis:
                current_price = analysis['indicators']['current_price']

                # Store current price for tracking
                self.current_prices[pair] = current_price

                logger.info(
                    f"{pair}: {analysis['signals']['action']} "
                    f"(Strength: {analysis['signals']['strength']}) @ ${current_price:.2f}"
                )

                # Send signal via Telegram
                await self.notifier.send_signal(pair, analysis)

        except Exception as e:
            logger.error(f"Error analyzing {pair}: {e}")

    async def monitor_loop(self):
        """
        Main monitoring loop - continuously analyzes all pairs
        """
        logger.info("Starting market monitor...")

        # Get list of pairs to display (filter out unavailable ones)
        display_pairs = [p for p in self.trading_pairs if p != 'MXN/USD']
        pairs_display = ', '.join(display_pairs[:5])
        if len(display_pairs) > 5:
            pairs_display += f" y {len(display_pairs) - 5} más"

        await self.notifier.send_status_message(
            "🤖 <b>Bot de Señales Iniciado</b>\n\n"
            f"📊 Monitoreando: {pairs_display}\n"
            f"⏱️ Intervalo: {self.check_interval}s\n"
            f"📈 Timeframe: {config.TIMEFRAME}\n"
            f"📍 Reporte diario: 9 PM CDMX"
        )

        self.is_running = True
        iteration = 0

        while self.is_running:
            try:
                iteration += 1
                logger.info(f"=== Iteration {iteration} - {datetime.now()} ===")

                # Analyze all pairs
                for pair in self.trading_pairs:
                    await self.analyze_pair(pair)
                    # Small delay between pairs to avoid rate limits
                    await asyncio.sleep(2)

                # Update pending signals with current prices
                if self.tracker:
                    self.tracker.check_pending_signals(self.current_prices)

                # Check if it's time for daily report
                if self.reporter:
                    await self.reporter.check_and_send_report()

                logger.info(f"Waiting {self.check_interval} seconds until next check...")
                await asyncio.sleep(self.check_interval)

            except KeyboardInterrupt:
                logger.info("Stopping monitor...")
                self.is_running = False
                break

            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                await self.notifier.send_error_message(f"Error en el loop: {str(e)}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying

        await self.notifier.send_status_message("🛑 <b>Bot de Señales Detenido</b>")

    async def start(self):
        """
        Start the market monitor
        """
        try:
            await self.monitor_loop()
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            await self.notifier.send_error_message(f"Error fatal: {str(e)}")
            raise

    def stop(self):
        """
        Stop the market monitor
        """
        self.is_running = False
