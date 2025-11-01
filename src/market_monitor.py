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
from src.advanced_technical_analysis import AdvancedTechnicalAnalyzer
from src.flash_signal_analyzer import FlashSignalAnalyzer
from src.telegram_bot import TelegramNotifier
from src.signal_tracker import SignalTracker
from src.daily_reporter import DailyReporter
from src.ml.ml_integration import MLIntegration
from src.sentiment.sentiment_integration import SentimentIntegration
from src.orderbook.orderbook_analyzer import OrderBookAnalyzer
from src.market_regime.regime_detector import RegimeDetector

logger = logging.getLogger(__name__)


class MarketMonitor:
    """
    Monitors market data and generates trading signals with dual system:
    - Conservative signals (1h/4h/1d multi-timeframe, threshold 7+)
    - Flash signals (10m, threshold 4+, marked as RISKY)
    """

    def __init__(self):
        self.exchange_name = config.EXCHANGE_NAME
        self.exchange = self._initialize_exchange()

        # Dual analyzers
        self.analyzer = AdvancedTechnicalAnalyzer()  # Conservative
        self.flash_analyzer = FlashSignalAnalyzer()  # Flash/Risky

        # Initialize signal tracker
        self.tracker = SignalTracker() if config.TRACK_SIGNALS else None

        # Initialize notifier with tracker
        self.notifier = TelegramNotifier(tracker=self.tracker)

        # Initialize daily reporter
        self.reporter = DailyReporter(self.tracker, self.notifier) if self.tracker else None

        # Initialize ML + Paper Trading system
        self.ml_system = MLIntegration(
            initial_balance=config.PAPER_TRADING_INITIAL_BALANCE,
            enable_ml=True  # Enable ML predictions
        ) if config.ENABLE_PAPER_TRADING else None

        # Initialize Sentiment Analysis system
        self.sentiment_system = SentimentIntegration(
            cryptopanic_api_key=config.CRYPTOPANIC_API_KEY,
            update_interval_minutes=config.SENTIMENT_UPDATE_INTERVAL,
            enable_blocking=config.SENTIMENT_BLOCK_ON_EXTREME_FEAR
        ) if config.ENABLE_SENTIMENT_ANALYSIS else None

        # Initialize Order Book Analyzer
        self.orderbook_analyzer = OrderBookAnalyzer(depth_limit=100)

        # Initialize Market Regime Detector
        self.regime_detector = RegimeDetector()

        self.trading_pairs = config.TRADING_PAIRS
        self.timeframe = config.TIMEFRAME
        self.timeframes = ['1h', '4h', '1d']  # Multi-timeframe for conservative
        self.flash_timeframe = config.FLASH_TIMEFRAME  # 10m for flash signals
        self.check_interval = config.CHECK_INTERVAL
        self.is_running = False
        self.current_prices = {}  # Store current prices for tracking
        self.enable_flash = config.ENABLE_FLASH_SIGNALS
        self.enable_paper_trading = config.ENABLE_PAPER_TRADING  # New config option
        self.autonomy_controller = None  # Will be set by main.py if autonomous mode enabled

    def _initialize_exchange(self) -> ccxt.Exchange:
        """
        Initialize exchange connection with optional proxy support

        Returns:
            CCXT exchange instance
        """
        try:
            exchange_class = getattr(ccxt, self.exchange_name)

            # Base exchange configuration
            exchange_config = {
                'apiKey': config.EXCHANGE_API_KEY,
                'secret': config.EXCHANGE_API_SECRET,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',
                }
            }

            # Add proxy configuration if enabled (required for Binance on datacenter IPs)
            if config.USE_PROXY and config.PROXY_HOST and config.PROXY_PORT:
                # Build proxy URL
                if config.PROXY_USERNAME and config.PROXY_PASSWORD:
                    proxy_url = f"http://{config.PROXY_USERNAME}:{config.PROXY_PASSWORD}@{config.PROXY_HOST}:{config.PROXY_PORT}"
                else:
                    proxy_url = f"http://{config.PROXY_HOST}:{config.PROXY_PORT}"

                exchange_config['proxies'] = {
                    'http': proxy_url,
                    'https': proxy_url,
                }
                logger.info(f"Connected to {self.exchange_name} via proxy {config.PROXY_HOST}:{config.PROXY_PORT}")
            else:
                logger.info(f"Connected to {self.exchange_name} (no proxy)")

            exchange = exchange_class(exchange_config)
            return exchange

        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")
            raise

    async def fetch_ohlcv(self, pair: str, timeframe: str = None) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data for a trading pair

        Args:
            pair: Trading pair (e.g., 'BTC/USD')
            timeframe: Timeframe (default: config.TIMEFRAME)

        Returns:
            DataFrame with OHLCV data or None
        """
        if timeframe is None:
            timeframe = self.timeframe

        try:
            # Fetch data
            ohlcv = self.exchange.fetch_ohlcv(
                pair,
                timeframe=timeframe,
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
        Analyze a trading pair with dual system:
        1. Conservative multi-timeframe analysis (1h/4h/1d)
        2. Flash signals (10m timeframe) if enabled
        3. Sentiment analysis integration

        Args:
            pair: Trading pair to analyze
        """
        try:
            logger.info(f"Analyzing {pair}...")

            # UPDATE SENTIMENT DATA (every 15 minutes, cached)
            sentiment_data = None
            sentiment_features = None
            if self.sentiment_system:
                base_currency = pair.split('/')[0]
                self.sentiment_system.update(currencies=[base_currency])
                sentiment_features = self.sentiment_system.get_sentiment_features(pair)
                sentiment_data = sentiment_features  # Para telegram

            # ANALYZE ORDER BOOK (10 second cache)
            orderbook_analysis = None
            orderbook_features = None
            try:
                orderbook_analysis = self.orderbook_analyzer.analyze(
                    exchange=self.exchange,
                    pair=pair,
                    current_price=0  # Will be updated later
                )
                orderbook_features = self.orderbook_analyzer.get_orderbook_features(orderbook_analysis)
            except Exception as e:
                logger.warning(f"Order book analysis failed for {pair}: {e}")

            # DETECT MARKET REGIME (15 minute cache)
            regime_data = None
            regime_features = None
            try:
                regime_data = self.regime_detector.detect(
                    exchange=self.exchange,
                    pair=pair,
                    current_price=0  # Will be updated later
                )
                regime_features = self.regime_detector.get_regime_features(regime_data)
            except Exception as e:
                logger.warning(f"Regime detection failed for {pair}: {e}")

            # 1. CONSERVATIVE ANALYSIS - Multi-timeframe
            dfs = {}
            for tf in self.timeframes:
                df = await self.fetch_ohlcv(pair, tf)
                if df is not None and len(df) >= 50:
                    dfs[tf] = df
                await asyncio.sleep(0.5)  # Small delay between requests

            if not dfs or '1h' not in dfs:
                logger.warning(f"Insufficient data for {pair}")
                return

            # Perform advanced multi-timeframe analysis
            analysis = self.analyzer.analyze_multi_timeframe(dfs)

            if analysis:
                current_price = analysis['indicators']['current_price']
                signals = analysis['signals']

                # Store current price for tracking
                self.current_prices[pair] = current_price

                # Update order book and regime with actual current price
                if orderbook_analysis:
                    orderbook_analysis['current_price'] = current_price
                if regime_data:
                    regime_data['current_price'] = current_price

                logger.info(
                    f"{pair}: {signals['action']} "
                    f"(Score: {signals.get('score', 0):.1f}/{signals.get('max_score', 10)}) "
                    f"@ ${current_price:.2f}"
                )

                # Log regime and orderbook info
                if regime_data:
                    logger.info(f"  Regime: {regime_data['regime']} ({regime_data['regime_strength']})")
                if orderbook_analysis:
                    logger.info(f"  Order Book: {orderbook_analysis['market_pressure']} (imbalance={orderbook_analysis['imbalance']:.2f})")

                # APPLY SENTIMENT ANALYSIS TO SIGNALS
                if self.sentiment_system and signals['action'] != 'HOLD':
                    # Check if trade should be blocked by sentiment
                    if self.sentiment_system.should_block_trade(pair, signals):
                        logger.warning(f"🚫 Trade bloqueado por sentiment analysis extremo para {pair}")
                        return  # Don't process this signal

                    # Adjust signal confidence based on sentiment
                    signals = self.sentiment_system.adjust_signal_confidence(pair, signals)
                    logger.info(f"📊 Sentiment applied to {pair}: confidence={signals.get('confidence')}%")

                # Add all additional data to analysis for telegram notifications
                if sentiment_data:
                    analysis['sentiment_data'] = sentiment_data
                if orderbook_analysis:
                    analysis['orderbook_data'] = orderbook_analysis
                if regime_data:
                    analysis['regime_data'] = regime_data

                # Execute paper trade if enabled
                if self.enable_paper_trading and self.ml_system and signals['action'] != 'HOLD':
                    trade_result = self.ml_system.process_signal(
                        pair=pair,
                        signal=signals,
                        indicators=analysis['indicators'],
                        current_price=current_price,
                        mtf_indicators=analysis.get('mtf_indicators'),
                        sentiment_features=sentiment_features,  # Pass all features to ML
                        orderbook_features=orderbook_features,
                        regime_features=regime_features
                    )

                    # Enhance analysis with ML data if trade was processed
                    if trade_result and 'ml' in signals:
                        analysis['ml_data'] = signals['ml']
                        analysis['trade_result'] = trade_result

                # Update existing positions
                if self.enable_paper_trading and self.ml_system:
                    closed_trade = self.ml_system.update_position(pair, current_price)
                    if closed_trade:
                        logger.info(f"Position closed for {pair}: {closed_trade}")

                        # Send trade outcome to autonomous controller for learning
                        if self.autonomy_controller:
                            await self._process_trade_outcome_autonomous(
                                closed_trade,
                                analysis.get('indicators', {}),
                                sentiment_data,
                                orderbook_analysis,
                                regime_data
                            )

                # Send signal notification if strong enough (score >= 7)
                if signals['action'] != 'HOLD':
                    await self.notifier.send_signal(pair, analysis)
                else:
                    logger.debug(f"{pair}: No strong conservative signal detected")

            # 2. FLASH ANALYSIS - 10 minute timeframe (if enabled)
            if self.enable_flash:
                await asyncio.sleep(0.5)  # Small delay
                flash_df = await self.fetch_ohlcv(pair, self.flash_timeframe)

                if flash_df is not None and len(flash_df) >= 30:
                    flash_analysis = self.flash_analyzer.analyze_flash(flash_df)

                    if flash_analysis:
                        flash_signals = flash_analysis['signals']
                        flash_price = flash_analysis['indicators']['current_price']

                        logger.info(
                            f"{pair} FLASH: {flash_signals['action']} "
                            f"(Score: {flash_signals.get('score', 0):.1f}/{flash_signals.get('max_score', 10)}) "
                            f"@ ${flash_price:.2f}"
                        )

                        # APPLY SENTIMENT TO FLASH SIGNALS
                        if self.sentiment_system and flash_signals['action'] != 'HOLD':
                            # Check if should block
                            if self.sentiment_system.should_block_trade(pair, flash_signals):
                                logger.warning(f"🚫 Flash trade bloqueado por sentiment para {pair}")
                                return

                            # Adjust confidence
                            flash_signals = self.sentiment_system.adjust_signal_confidence(pair, flash_signals)

                        # Add all additional data to flash analysis
                        if sentiment_data:
                            flash_analysis['sentiment_data'] = sentiment_data
                        if orderbook_analysis:
                            flash_analysis['orderbook_data'] = orderbook_analysis
                        if regime_data:
                            flash_analysis['regime_data'] = regime_data

                        # Execute paper trade for flash signal if enabled
                        if self.enable_paper_trading and self.ml_system and flash_signals['action'] != 'HOLD':
                            flash_trade_result = self.ml_system.process_signal(
                                pair=pair,
                                signal=flash_signals,
                                indicators=flash_analysis['indicators'],
                                current_price=flash_price,
                                mtf_indicators=None,  # Flash signals don't use MTF
                                sentiment_features=sentiment_features,  # Pass all features to ML
                                orderbook_features=orderbook_features,
                                regime_features=regime_features
                            )

                            # Enhance analysis with ML data
                            if flash_trade_result and 'ml' in flash_signals:
                                flash_analysis['ml_data'] = flash_signals['ml']
                                flash_analysis['trade_result'] = flash_trade_result

                        # Send flash signal notification if threshold met (5+ points)
                        if flash_signals['action'] != 'HOLD':
                            await self.notifier.send_signal(pair, flash_analysis)
                        else:
                            logger.debug(f"{pair}: No flash signal detected")

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

        flash_status = "✅ Activas" if self.enable_flash else "❌ Desactivadas"
        paper_trading_status = "✅ Activo" if self.enable_paper_trading else "❌ Inactivo"

        startup_message = (
            "🤖 <b>Bot de Señales Iniciado</b>\n\n"
            f"📊 Monitoreando: {pairs_display}\n"
            f"⏱️ Intervalo: {self.check_interval}s\n"
            f"📈 Timeframe conservador: {config.TIMEFRAME} (1h/4h/1d)\n"
            f"⚡ Señales flash: {flash_status} ({self.flash_timeframe})\n"
            f"💰 Paper Trading: {paper_trading_status}"
        )

        if self.enable_paper_trading and self.ml_system:
            startup_message += f" (${config.PAPER_TRADING_INITIAL_BALANCE:,.0f} USDT)\n🧠 Machine Learning: ✅ Activo"

        startup_message += "\n📍 Reporte diario: 9 PM CDMX"

        await self.notifier.send_status_message(startup_message)

        self.is_running = True
        iteration = 0
        stats_iteration = 0  # Send stats every N iterations

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

                # Send paper trading stats every 30 iterations (every hour if interval=120s)
                if self.enable_paper_trading and self.ml_system:
                    stats_iteration += 1
                    if stats_iteration >= 30:  # ~1 hour
                        try:
                            stats = self.ml_system.get_comprehensive_stats()
                            await self.notifier.send_trading_stats(stats)
                            stats_iteration = 0
                        except Exception as e:
                            logger.error(f"Error sending paper trading stats: {e}")

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

    async def _process_trade_outcome_autonomous(
        self,
        closed_trade: Dict,
        indicators: Dict,
        sentiment_data: Optional[Dict],
        orderbook_data: Optional[Dict],
        regime_data: Optional[Dict]
    ):
        """
        Procesa resultado de trade cerrado y lo envía al autonomy controller para aprendizaje

        Args:
            closed_trade: Datos del trade cerrado
            indicators: Indicadores técnicos
            sentiment_data: Datos de sentiment
            orderbook_data: Datos de order book
            regime_data: Datos de regime
        """
        try:
            # Construir estado de mercado para el RL Agent
            market_state = {
                'rsi': indicators.get('rsi', 50),
                'macd_signal': 'bullish' if indicators.get('macd_diff', 0) > 0 else 'bearish',
                'trend': 'up' if indicators.get('ema_short', 0) > indicators.get('ema_long', 0) else 'down',
                'regime': regime_data['regime'] if regime_data else 'SIDEWAYS',
                'sentiment': sentiment_data.get('sentiment', 'neutral') if sentiment_data else 'neutral',
                'volatility': 'high' if indicators.get('atr', 0) > indicators.get('current_price', 1) * 0.02 else 'medium',
                'win_rate': 0,  # Will be updated with portfolio metrics
                'drawdown': 0   # Will be updated with portfolio metrics
            }

            # Obtener métricas del portfolio
            portfolio_metrics = {}
            if self.ml_system and hasattr(self.ml_system, 'paper_trader'):
                stats = self.ml_system.paper_trader.portfolio.get_statistics()
                market_state['win_rate'] = stats['win_rate']
                market_state['drawdown'] = stats['max_drawdown']

                portfolio_metrics = {
                    'win_rate': stats['win_rate'],
                    'roi': stats['roi'],
                    'max_drawdown': stats['max_drawdown'],
                    'sharpe_ratio': stats.get('sharpe_ratio', 0),
                    'profit_factor': stats.get('profit_factor', 1.0),
                    'total_trades': stats['total_trades']
                }

            # Preparar datos del trade
            trade_data = {
                'pair': closed_trade.get('pair', 'UNKNOWN'),
                'action': closed_trade.get('action', 'UNKNOWN'),
                'entry_price': closed_trade.get('entry_price', 0),
                'exit_price': closed_trade.get('exit_price', 0),
                'profit_pct': closed_trade.get('profit_pct', 0),
                'profit_amount': closed_trade.get('profit', 0),
                'duration': closed_trade.get('duration', 0),
                'exit_reason': closed_trade.get('exit_reason', 'UNKNOWN')
            }

            # Enviar al autonomy controller
            await self.autonomy_controller.process_trade_outcome(
                trade_data=trade_data,
                market_state=market_state,
                portfolio_metrics=portfolio_metrics
            )

            logger.debug(f"✅ Trade outcome enviado a Autonomy Controller: {trade_data['pair']}")

        except Exception as e:
            logger.error(f"❌ Error procesando trade outcome para autonomy: {e}", exc_info=True)

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
