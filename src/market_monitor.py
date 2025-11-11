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
from src.advanced.feature_aggregator import FeatureAggregator

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
            enable_ml=True,  # Enable ML predictions
            telegram_notifier=self.notifier  # Para notificaciones de trades
        ) if config.ENABLE_PAPER_TRADING else None

        # Auto-entrenar ML si hay suficientes trades históricos (40+)
        if self.ml_system:
            self._auto_train_ml_if_ready()

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

        # Initialize Feature Aggregator (Arsenal Avanzado) - Orquesta 7 módulos
        self.feature_aggregator = FeatureAggregator(config, self.exchange)
        logger.info("✅ Feature Aggregator initialized: Arsenal Avanzado loaded")

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

    def _auto_train_ml_if_ready(self):
        """
        Auto-entrena el ML System si hay suficientes trades históricos (40+)
        Se ejecuta automáticamente al inicializar MarketMonitor

        IMPORTANTE: Usa el paper_trader del autonomy_controller si está disponible
        (porque ese SÍ se restaura con /import), en lugar del paper_trader interno
        del ML System que puede estar vacío después de un import.
        """
        try:
            if not self.ml_system:
                return

            # Intentar usar el paper_trader del autonomy_controller primero
            # (porque ese SÍ se restaura en /import)
            paper_trader = None
            total_trades = 0

            if hasattr(self, 'autonomy_controller') and self.autonomy_controller:
                if hasattr(self.autonomy_controller, 'paper_trader') and self.autonomy_controller.paper_trader:
                    paper_trader = self.autonomy_controller.paper_trader
                    stats = paper_trader.portfolio.get_statistics()
                    total_trades = stats.get('total_trades', 0)
                    logger.debug(f"📊 Usando paper_trader del autonomy_controller: {total_trades} trades")

            # Fallback: usar el paper_trader interno del ML System
            if not paper_trader and hasattr(self.ml_system, 'paper_trader') and self.ml_system.paper_trader:
                paper_trader = self.ml_system.paper_trader
                stats = paper_trader.portfolio.get_statistics()
                total_trades = stats.get('total_trades', 0)
                logger.debug(f"📊 Usando paper_trader interno del ML System: {total_trades} trades")

            if not paper_trader:
                logger.debug("📊 No hay paper_trader disponible para auto-entrenamiento")
                return

            # Verificar si hay suficientes trades para entrenar (40+)
            if total_trades >= 40:
                logger.info(f"🤖 Detectados {total_trades} trades históricos - Iniciando auto-entrenamiento ML...")

                # Forzar entrenamiento con threshold reducido (25 samples mínimo)
                # Pasar el paper_trader que tiene los datos
                self.ml_system.force_retrain(
                    min_samples_override=25,
                    external_paper_trader=paper_trader
                )

                logger.info("✅ ML System entrenado automáticamente con datos históricos")
            elif total_trades > 0:
                logger.info(f"📊 ML System esperando más trades para entrenar: {total_trades}/40")
            else:
                logger.info("📊 ML System iniciado - entrenará automáticamente después de 40 trades")

        except Exception as e:
            logger.error(f"Error en auto-entrenamiento ML: {e}")
            # No es crítico, el sistema puede funcionar sin ML entrenado

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

                # ⚡ CHECK NEWS-TRIGGERED SIGNALS (CRITICAL NEWS)
                # Esto da ventaja de 5-30 min antes del mercado
                try:
                    from config import config
                    news_signals = self.sentiment_system.get_news_triggered_signals(
                        current_pairs=config.TRADING_PAIRS,
                        current_price_changes=getattr(self, 'price_changes_1h', {})
                    )

                    # Process urgent news signals for this pair
                    for news_signal in news_signals:
                        if news_signal.get('pair') == pair and news_signal.get('urgency') == 'HIGH':
                            logger.warning(
                                f"⚡ NEWS-TRIGGERED SIGNAL para {pair}: "
                                f"{news_signal.get('action')} "
                                f"(confidence={news_signal.get('confidence')}%) "
                                f"- {news_signal.get('reason', 'Critical news detected')}"
                            )

                            # Send urgent notification to Telegram
                            if self.notifier:
                                await self.notifier.send_message(
                                    f"⚡ **SEÑAL URGENTE POR NOTICIAS**\n\n"
                                    f"Par: {pair}\n"
                                    f"Acción: {news_signal.get('action')}\n"
                                    f"Confianza: {news_signal.get('confidence')}%\n"
                                    f"Urgencia: {news_signal.get('urgency')}\n"
                                    f"Razón: {news_signal.get('reason')}\n\n"
                                    f"🎯 Ventana: {news_signal.get('expected_timeframe', '5-30min')}\n"
                                    f"📈 Movimiento esperado: {news_signal.get('expected_move', 'N/A')}"
                                )

                            # Execute news-triggered trade immediately (if paper trading enabled)
                            if self.enable_paper_trading and self.ml_system:
                                # Get current price for execution
                                try:
                                    ticker = self.exchange.fetch_ticker(pair)
                                    news_price = ticker['last'] if ticker and 'last' in ticker else 0

                                    if news_price > 0:
                                        # CONSULTAR AL RL AGENT ANTES DE EJECUTAR NEWS TRADE
                                        should_execute_news = True
                                        rl_news_decision = None

                                        if self.autonomy_controller:
                                            # Derivar orderbook status de orderbook_analysis
                                            orderbook_status_news = 'NEUTRAL'
                                            if orderbook_analysis:
                                                pressure = orderbook_analysis.get('market_pressure', 'NEUTRAL')
                                                if pressure == 'BUY_PRESSURE':
                                                    orderbook_status_news = 'BUY_PRESSURE'
                                                elif pressure == 'SELL_PRESSURE':
                                                    orderbook_status_news = 'SELL_PRESSURE'

                                            # Construir market state para news trade - INTEGRACIÓN COMPLETA
                                            market_state_news = {
                                                'rsi': 50,  # No tenemos indicadores detallados aún
                                                'regime': regime_data['regime'] if regime_data else 'SIDEWAYS',
                                                'regime_strength': regime_data.get('regime_strength', 'HIGH') if regime_data else 'HIGH',  # News = high volatility
                                                'orderbook': orderbook_status_news,
                                                'volatility': 'high',  # News trades son inherentemente volátiles

                                                # Agregar sentiment data si está disponible
                                                'cryptopanic_sentiment': sentiment_data.get('sentiment', 'neutral') if sentiment_data else 'neutral',
                                                'news_volume': sentiment_data.get('news_volume', 0) if sentiment_data else 0,
                                                'news_importance': sentiment_data.get('news_importance', 0) if sentiment_data else 0,
                                                'pre_pump_score': sentiment_data.get('pre_pump_score', 0) if sentiment_data else 0,
                                                'fear_greed_index': sentiment_data.get('fear_greed_index', 50) if sentiment_data else 50,
                                                'fear_greed_label': sentiment_data.get('fear_greed_label', 'neutral') if sentiment_data else 'neutral',
                                                'overall_sentiment': sentiment_data.get('overall_sentiment', 'neutral') if sentiment_data else 'neutral',
                                                'sentiment_strength': sentiment_data.get('sentiment_strength', 0) if sentiment_data else 0,
                                                'social_buzz': sentiment_data.get('social_buzz', 0) if sentiment_data else 0,

                                                # News triggered = TRUE para news trades
                                                'news_triggered': True,
                                                'news_trigger_confidence': news_signal.get('confidence', 80),

                                                # Multi-layer alignment
                                                'multi_layer_alignment': 0,  # No disponible para news trades urgentes

                                                # Orderbook data si está disponible
                                                'orderbook_imbalance': orderbook_analysis.get('imbalance', 0) if orderbook_analysis else 0,
                                                'bid_ask_spread': orderbook_analysis.get('spread_pct', 0) if orderbook_analysis else 0,
                                                'orderbook_depth_score': orderbook_analysis.get('depth_score', 0) if orderbook_analysis else 0,
                                                'market_pressure': orderbook_analysis.get('market_pressure', 'NEUTRAL') if orderbook_analysis else 'NEUTRAL',

                                                # Regime data si está disponible
                                                'regime_confidence': regime_data.get('confidence', 0) if regime_data else 0,
                                                'trend_strength': regime_data.get('trend_strength', 0) if regime_data else 0,
                                                'volatility_regime': 'HIGH',  # News = high volatility
                                            }

                                            # Obtener portfolio metrics
                                            portfolio_metrics_news = {}
                                            if hasattr(self.ml_system, 'paper_trader'):
                                                stats = self.ml_system.paper_trader.portfolio.get_statistics()
                                                portfolio_metrics_news = {
                                                    'win_rate': stats['win_rate'],
                                                    'roi': stats['roi'],
                                                    'max_drawdown': stats['max_drawdown'],
                                                    'sharpe_ratio': stats.get('sharpe_ratio', 0),
                                                    'total_trades': stats['total_trades']
                                                }

                                            # RL Agent evalúa si ejecutar news trade
                                            rl_news_decision = await self.autonomy_controller.evaluate_trade_opportunity(
                                                pair=pair,
                                                signal=news_signal,
                                                market_state=market_state_news,
                                                portfolio_metrics=portfolio_metrics_news
                                            )

                                            should_execute_news = rl_news_decision.get('should_trade', True)

                                            # Aplicar modificador de tamaño de posición
                                            if should_execute_news and 'position_size_multiplier' in rl_news_decision:
                                                news_signal['rl_position_multiplier'] = rl_news_decision['position_size_multiplier']
                                                news_signal['rl_action'] = rl_news_decision.get('chosen_action', 'UNKNOWN')

                                        # Ejecutar news trade solo si RL Agent lo aprueba
                                        if should_execute_news:
                                            logger.info(f"⚡ Ejecutando trade por NEWS-TRIGGER en {pair} @ ${news_price}")

                                            # Process news signal as a trade
                                            news_trade_result = self.ml_system.process_signal(
                                                pair=pair,
                                                signal=news_signal,
                                                indicators={'current_price': news_price},
                                                current_price=news_price,
                                                mtf_indicators=None,
                                                sentiment_features=sentiment_features,
                                                orderbook_features=None,
                                                regime_features=None
                                            )

                                            if news_trade_result:
                                                logger.info(f"✅ News-triggered trade ejecutado: {news_trade_result}")
                                        else:
                                            logger.info(f"🤖 RL Agent bloqueó news trade en {pair}: {rl_news_decision.get('chosen_action', 'SKIP')}")

                                except Exception as e:
                                    logger.error(f"Error ejecutando news-triggered trade: {e}")

                except Exception as e:
                    logger.error(f"Error checking news-triggered signals: {e}")

            # Obtener precio actual del ticker PRIMERO (para orderbook y regime)
            current_price_ticker = 0
            try:
                ticker = self.exchange.fetch_ticker(pair)
                current_price_ticker = ticker['last'] if ticker and 'last' in ticker else 0
            except Exception as e:
                logger.warning(f"No se pudo obtener ticker para {pair}: {e}")

            # ANALYZE ORDER BOOK (10 second cache)
            orderbook_analysis = None
            orderbook_features = None
            try:
                orderbook_analysis = self.orderbook_analyzer.analyze(
                    exchange=self.exchange,
                    pair=pair,
                    current_price=current_price_ticker
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
                    current_price=current_price_ticker
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

                # ===== ARSENAL AVANZADO: ANÁLISIS COMPLETO (SIEMPRE) =====
                # Ejecutar análisis del arsenal SIEMPRE (no solo en BUY/SELL) para logging
                try:
                    # Obtener orderbook raw para arsenal
                    orderbook_raw = None
                    try:
                        orderbook_raw = self.exchange.fetch_order_book(pair)
                    except Exception as e:
                        logger.debug(f"No se pudo obtener orderbook raw para {pair}: {e}")

                    # Obtener posiciones abiertas
                    open_positions = []
                    if self.ml_system and hasattr(self.ml_system, 'paper_trader'):
                        open_positions = list(self.ml_system.paper_trader.portfolio.positions.keys())

                    # OBTENER ANÁLISIS DEL ARSENAL (preview)
                    arsenal_ml_features_preview = self.feature_aggregator.get_ml_features(
                        pair=pair,
                        current_price=current_price,
                        base_features={},  # Diccionario vacío para preview
                        ohlc_data=dfs.get('1h')
                    )
                    arsenal_rl_extensions_preview = self.feature_aggregator.get_rl_state_extensions(
                        pair=pair,
                        current_price=current_price,
                        open_positions=open_positions
                    )

                    # LOG DETALLADO DEL ARSENAL
                    logger.info(f"  📊 ARSENAL AVANZADO ({pair}):")

                    # Session Trading
                    session = arsenal_rl_extensions_preview.get('current_session', 'UNKNOWN')
                    session_mult = arsenal_rl_extensions_preview.get('session_multiplier', 1.0)
                    logger.info(f"    ⏰ Session: {session} (multiplier={session_mult:.2f}x)")

                    # Order Flow
                    flow_bias = arsenal_rl_extensions_preview.get('order_flow_bias', 'neutral')
                    flow_ratio = arsenal_rl_extensions_preview.get('order_flow_ratio', 1.0)
                    logger.info(f"    💧 Order Flow: {flow_bias} (ratio={flow_ratio:.2f})")

                    # Funding Rate
                    funding_sentiment = arsenal_ml_features_preview.get('funding_sentiment', 'neutral')
                    funding_rate = arsenal_ml_features_preview.get('funding_rate', 0)
                    if funding_rate != 0:
                        logger.info(f"    💰 Funding Rate: {funding_rate:.4f}% (sentiment={funding_sentiment})")

                    # Liquidation Heatmap
                    liq_bias = arsenal_ml_features_preview.get('liquidation_bias', 'neutral')
                    liq_conf = arsenal_ml_features_preview.get('liquidation_confidence', 0)
                    if liq_conf > 0:
                        logger.info(f"    🔥 Liquidation: {liq_bias} (confidence={liq_conf:.0%})")

                    # Volume Profile
                    near_poc = arsenal_ml_features_preview.get('near_poc', False)
                    in_value = arsenal_ml_features_preview.get('in_value_area', False)
                    if near_poc or in_value:
                        logger.info(f"    📈 Volume Profile: POC={near_poc}, ValueArea={in_value}")

                    # Pattern Recognition
                    pattern_detected = arsenal_ml_features_preview.get('pattern_detected', False)
                    if pattern_detected:
                        pattern_type = arsenal_ml_features_preview.get('pattern_type', 'NONE')
                        logger.info(f"    🎯 Pattern: {pattern_type} detected!")

                    # Correlation Risk
                    corr_risk = arsenal_rl_extensions_preview.get('correlation_risk', 0)
                    corr_positions = arsenal_rl_extensions_preview.get('correlated_positions', 0)
                    if len(open_positions) > 0:
                        logger.info(f"    🔗 Correlation: {corr_positions} correlated positions (risk={corr_risk:.2f})")

                    # Sentiment completo
                    if sentiment_features:
                        fg_index = sentiment_features.get('fear_greed_index', 0.5) * 100
                        news_sentiment = sentiment_features.get('news_sentiment_overall', 0.5)
                        social_buzz = sentiment_features.get('social_buzz', 0)
                        logger.info(f"    📰 Sentiment: F&G={fg_index:.0f}, News={news_sentiment:.2f}, SocialBuzz={social_buzz}")

                    # ML Features TOTAL
                    total_ml_features = len(sentiment_features or {}) + len(arsenal_ml_features_preview)
                    logger.info(f"    🧠 ML Features: {total_ml_features} total (Sentiment={len(sentiment_features or {})} + Arsenal={len(arsenal_ml_features_preview)})")

                    # RL State Extensions
                    logger.info(f"    🤖 RL State: 19 dimensions (12 base + 7 arsenal)")

                except Exception as e:
                    logger.error(f"  ❌ Error en análisis del arsenal: {e}")
                # ===== FIN ANÁLISIS ARSENAL =====

                # ===== ARSENAL AVANZADO: ENRICH SIGNAL =====
                # Aplicar Feature Aggregator ANTES de sentiment (para combinar TODOS los módulos)
                if signals['action'] != 'HOLD':
                    try:
                        # Obtener orderbook raw para arsenal
                        orderbook_raw = None
                        try:
                            orderbook_raw = self.exchange.fetch_order_book(pair)
                        except Exception as e:
                            logger.debug(f"No se pudo obtener orderbook raw para {pair}: {e}")

                        # Obtener posiciones abiertas para correlation matrix
                        open_positions = []
                        if self.ml_system and hasattr(self.ml_system, 'paper_trader'):
                            open_positions = list(self.ml_system.paper_trader.portfolio.positions.keys())

                        # ENRIQUECER SEÑAL CON ARSENAL AVANZADO
                        enriched_signal = self.feature_aggregator.enrich_signal(
                            pair=pair,
                            signal=signals,
                            current_price=current_price,
                            ohlc_data=dfs.get('1h'),  # Usamos 1h para análisis
                            orderbook=orderbook_raw,
                            open_positions=open_positions
                        )

                        # Aplicar boost/penalty del arsenal
                        if enriched_signal.get('blocked', False):
                            logger.warning(f"🚫 Trade bloqueado por Arsenal Avanzado: {enriched_signal.get('block_reason', 'UNKNOWN')}")
                            return

                        # Actualizar señal con datos del arsenal
                        signals['confidence'] = enriched_signal.get('final_confidence', signals.get('confidence', 50))
                        signals['arsenal_boost'] = enriched_signal.get('total_boost', 1.0)
                        signals['arsenal_analysis'] = enriched_signal.get('analysis', {})

                        logger.info(f"🚀 Arsenal Avanzado aplicado a {pair}: confidence={signals['confidence']:.1f}% (boost={enriched_signal.get('total_boost', 1.0):.2f}x)")

                    except Exception as e:
                        logger.error(f"Error applying Feature Aggregator to signal: {e}")
                # ===== FIN ARSENAL AVANZADO =====

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
                    # CONSULTAR AL RL AGENT ANTES DE ABRIR TRADE
                    should_execute_trade = True
                    rl_decision = None

                    if self.autonomy_controller:
                        # Derivar orderbook status de orderbook_analysis
                        orderbook_status = 'NEUTRAL'
                        if orderbook_analysis:
                            pressure = orderbook_analysis.get('market_pressure', 'NEUTRAL')
                            if pressure == 'BUY_PRESSURE':
                                orderbook_status = 'BUY_PRESSURE'
                            elif pressure == 'SELL_PRESSURE':
                                orderbook_status = 'SELL_PRESSURE'

                        # ===== ARSENAL AVANZADO: ML FEATURES + RL STATE EXTENSIONS =====
                        # Obtener features adicionales del arsenal para ML
                        arsenal_ml_features = {}
                        arsenal_rl_extensions = {}
                        try:
                            arsenal_ml_features = self.feature_aggregator.get_ml_features(
                                pair=pair,
                                current_price=current_price,
                                base_features={},  # Será combinado con sentiment_features después
                                ohlc_data=dfs.get('1h')
                            )
                            arsenal_rl_extensions = self.feature_aggregator.get_rl_state_extensions(
                                pair=pair,
                                current_price=current_price,
                                open_positions=open_positions
                            )
                        except Exception as e:
                            logger.warning(f"No se pudieron obtener features del arsenal: {e}")

                        # Construir market state para RL Agent - INTEGRACIÓN COMPLETA DE 24 SERVICIOS
                        market_state = {
                            # Indicadores técnicos básicos
                            'rsi': analysis['indicators'].get('rsi', 50),
                            'regime': regime_data['regime'] if regime_data else 'SIDEWAYS',
                            'regime_strength': regime_data.get('regime_strength', 'MEDIUM') if regime_data else 'MEDIUM',
                            'orderbook': orderbook_status,
                            'volatility': 'high' if analysis['indicators'].get('atr', 0) > current_price * 0.02 else 'medium',

                            # 3. CryptoPanic GROWTH API (sentiment_data)
                            'cryptopanic_sentiment': sentiment_data.get('sentiment', 'neutral') if sentiment_data else 'neutral',
                            'news_volume': sentiment_data.get('news_volume', 0) if sentiment_data else 0,
                            'news_importance': sentiment_data.get('news_importance', 0) if sentiment_data else 0,
                            'pre_pump_score': sentiment_data.get('pre_pump_score', 0) if sentiment_data else 0,

                            # 4. Fear & Greed Index (sentiment_data)
                            'fear_greed_index': sentiment_data.get('fear_greed_index', 50) if sentiment_data else 50,
                            'fear_greed_label': sentiment_data.get('fear_greed_label', 'neutral') if sentiment_data else 'neutral',

                            # 5. Sentiment Analysis (sentiment_data)
                            'overall_sentiment': sentiment_data.get('overall_sentiment', 'neutral') if sentiment_data else 'neutral',
                            'sentiment_strength': sentiment_data.get('sentiment_strength', 0) if sentiment_data else 0,
                            'social_buzz': sentiment_data.get('social_buzz', 0) if sentiment_data else 0,

                            # 6. News-Triggered Trading (signals puede tener news_triggered)
                            'news_triggered': signals.get('news_triggered', False),
                            'news_trigger_confidence': signals.get('news_trigger_confidence', 0),

                            # 7. Multi-Layer Confidence System (signals)
                            'multi_layer_alignment': signals.get('multi_layer_alignment', 0),

                            # 12. Order Book Analyzer (orderbook_analysis)
                            'orderbook_imbalance': orderbook_analysis.get('imbalance', 0) if orderbook_analysis else 0,
                            'bid_ask_spread': orderbook_analysis.get('spread_pct', 0) if orderbook_analysis else 0,
                            'orderbook_depth_score': orderbook_analysis.get('depth_score', 0) if orderbook_analysis else 0,
                            'market_pressure': orderbook_analysis.get('market_pressure', 'NEUTRAL') if orderbook_analysis else 'NEUTRAL',

                            # 13. Market Regime Detector (regime_data)
                            'regime_confidence': regime_data.get('confidence', 0) if regime_data else 0,
                            'trend_strength': regime_data.get('trend_strength', 0) if regime_data else 0,
                            'volatility_regime': regime_data.get('volatility', 'NORMAL') if regime_data else 'NORMAL',

                            # ===== ARSENAL AVANZADO (8 NUEVOS SERVICIOS) =====
                            # 17. Correlation Matrix
                            'correlation_risk': arsenal_rl_extensions.get('correlation_risk', 0),
                            'correlated_positions': arsenal_rl_extensions.get('correlated_positions', 0),

                            # 18. Liquidation Heatmap
                            'liquidation_bias': arsenal_rl_extensions.get('liquidation_bias', 'neutral'),
                            'liquidation_confidence': arsenal_rl_extensions.get('liquidation_confidence', 0),

                            # 19. Funding Rate Analyzer
                            'funding_sentiment': arsenal_rl_extensions.get('funding_sentiment', 'neutral'),
                            'funding_rate': arsenal_rl_extensions.get('funding_rate', 0),

                            # 20. Volume Profile & POC
                            'near_poc': arsenal_rl_extensions.get('near_poc', False),
                            'in_value_area': arsenal_rl_extensions.get('in_value_area', False),

                            # 21. Pattern Recognition
                            'pattern_detected': arsenal_rl_extensions.get('pattern_detected', False),
                            'pattern_type': arsenal_rl_extensions.get('pattern_type', 'NONE'),

                            # 22. Session-Based Trading
                            'current_session': arsenal_rl_extensions.get('current_session', 'UNKNOWN'),
                            'session_multiplier': arsenal_rl_extensions.get('session_multiplier', 1.0),

                            # 23. Order Flow Imbalance
                            'order_flow_bias': arsenal_rl_extensions.get('order_flow_bias', 'neutral'),
                            'order_flow_ratio': arsenal_rl_extensions.get('order_flow_ratio', 1.0),
                        }
                        # ===== FIN ARSENAL EXTENSIONS =====

                        # Obtener portfolio metrics
                        portfolio_metrics = {}
                        if hasattr(self.ml_system, 'paper_trader'):
                            stats = self.ml_system.paper_trader.portfolio.get_statistics()
                            portfolio_metrics = {
                                'win_rate': stats['win_rate'],
                                'roi': stats['roi'],
                                'max_drawdown': stats['max_drawdown'],
                                'sharpe_ratio': stats.get('sharpe_ratio', 0),
                                'total_trades': stats['total_trades']
                            }

                        # RL Agent evalúa si abrir el trade
                        rl_decision = await self.autonomy_controller.evaluate_trade_opportunity(
                            pair=pair,
                            signal=signals,
                            market_state=market_state,
                            portfolio_metrics=portfolio_metrics
                        )

                        # Decidir si ejecutar trade basado en RL Agent
                        should_execute_trade = rl_decision.get('should_trade', True)

                        # Aplicar parámetros del RL Agent (tamaño, trade_type, leverage)
                        if should_execute_trade and 'position_size_multiplier' in rl_decision:
                            # Pasar el multiplicador a la señal para que ml_system lo use
                            signals['rl_position_multiplier'] = rl_decision['position_size_multiplier']
                            signals['rl_action'] = rl_decision.get('chosen_action', 'UNKNOWN')
                            # Pasar trade_type y leverage para futuros
                            signals['trade_type'] = rl_decision.get('trade_type', 'SPOT')
                            signals['leverage'] = rl_decision.get('leverage', 1)

                    # Ejecutar trade solo si RL Agent lo aprueba
                    if should_execute_trade:
                        # ===== COMBINAR FEATURES: Sentiment + Orderbook + Regime + ARSENAL =====
                        # Merge arsenal ML features con las existentes
                        combined_sentiment_features = {**(sentiment_features or {}), **arsenal_ml_features}

                        trade_result = self.ml_system.process_signal(
                            pair=pair,
                            signal=signals,
                            indicators=analysis['indicators'],
                            current_price=current_price,
                            mtf_indicators=analysis.get('mtf_indicators'),
                            sentiment_features=combined_sentiment_features,  # ARSENAL + Sentiment combined
                            orderbook_features=orderbook_features,
                            regime_features=regime_features
                        )

                        # Enhance analysis with trade result if trade was processed
                        if trade_result:
                            analysis['trade_result'] = trade_result
                            # Add ML data if available
                            if 'ml' in signals:
                                analysis['ml_data'] = signals['ml']
                            # Add RL decision data
                            if rl_decision:
                                analysis['rl_decision'] = rl_decision
                    else:
                        logger.info(f"🤖 RL Agent bloqueó trade en {pair}: {rl_decision.get('chosen_action', 'SKIP')}")

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

                        # ===== ARSENAL AVANZADO: ANÁLISIS FLASH (SIEMPRE) =====
                        try:
                            # OBTENER ANÁLISIS DEL ARSENAL para flash
                            arsenal_flash_ml = self.feature_aggregator.get_ml_features(
                                pair=pair,
                                current_price=flash_price,
                                base_features={},  # Preview de flash
                                ohlc_data=flash_df
                            )
                            arsenal_flash_rl = self.feature_aggregator.get_rl_state_extensions(
                                pair=pair,
                                current_price=flash_price,
                                open_positions=open_positions
                            )

                            # LOG DETALLADO FLASH
                            logger.info(f"  ⚡ ARSENAL FLASH ({pair}):")
                            session_f = arsenal_flash_rl.get('current_session', 'UNKNOWN')
                            session_mult_f = arsenal_flash_rl.get('session_multiplier', 1.0)
                            flow_bias_f = arsenal_flash_rl.get('order_flow_bias', 'neutral')
                            flow_ratio_f = arsenal_flash_rl.get('order_flow_ratio', 1.0)
                            logger.info(f"    ⏰ Session: {session_f} ({session_mult_f:.2f}x) | 💧 Flow: {flow_bias_f} ({flow_ratio_f:.2f})")

                        except Exception as e:
                            logger.debug(f"  ⚡ Arsenal flash preview error: {e}")
                        # ===== FIN ANÁLISIS FLASH =====

                        # ===== ARSENAL AVANZADO: ENRICH FLASH SIGNAL =====
                        if flash_signals['action'] != 'HOLD':
                            try:
                                # Obtener orderbook raw para arsenal (reusar si ya se obtuvo)
                                orderbook_raw_flash = None
                                try:
                                    orderbook_raw_flash = self.exchange.fetch_order_book(pair)
                                except Exception as e:
                                    logger.debug(f"No se pudo obtener orderbook raw para flash {pair}: {e}")

                                # Obtener posiciones abiertas
                                open_positions_flash = []
                                if self.ml_system and hasattr(self.ml_system, 'paper_trader'):
                                    open_positions_flash = list(self.ml_system.paper_trader.portfolio.positions.keys())

                                # ENRIQUECER FLASH SIGNAL CON ARSENAL
                                enriched_flash = self.feature_aggregator.enrich_signal(
                                    pair=pair,
                                    signal=flash_signals,
                                    current_price=flash_price,
                                    ohlc_data=flash_df,  # 10m data para flash
                                    orderbook=orderbook_raw_flash,
                                    open_positions=open_positions_flash
                                )

                                # Aplicar boost/penalty del arsenal
                                if enriched_flash.get('blocked', False):
                                    logger.warning(f"🚫 Flash trade bloqueado por Arsenal: {enriched_flash.get('block_reason', 'UNKNOWN')}")
                                    return

                                # Actualizar flash signal con datos del arsenal
                                flash_signals['confidence'] = enriched_flash.get('final_confidence', flash_signals.get('confidence', 50))
                                flash_signals['arsenal_boost'] = enriched_flash.get('total_boost', 1.0)
                                flash_signals['arsenal_analysis'] = enriched_flash.get('analysis', {})

                                logger.info(f"⚡ Arsenal aplicado a FLASH {pair}: confidence={flash_signals['confidence']:.1f}% (boost={enriched_flash.get('total_boost', 1.0):.2f}x)")

                            except Exception as e:
                                logger.error(f"Error applying Arsenal to flash signal: {e}")
                        # ===== FIN ARSENAL FLASH =====

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
                            # CONSULTAR AL RL AGENT ANTES DE ABRIR FLASH TRADE
                            should_execute_flash = True
                            rl_flash_decision = None

                            if self.autonomy_controller:
                                # Derivar orderbook status de orderbook_analysis
                                orderbook_status_flash = 'NEUTRAL'
                                if orderbook_analysis:
                                    pressure = orderbook_analysis.get('market_pressure', 'NEUTRAL')
                                    if pressure == 'BUY_PRESSURE':
                                        orderbook_status_flash = 'BUY_PRESSURE'
                                    elif pressure == 'SELL_PRESSURE':
                                        orderbook_status_flash = 'SELL_PRESSURE'

                                # Construir market state para RL Agent
                                market_state_flash = {
                                    'rsi': flash_analysis['indicators'].get('rsi', 50),
                                    'regime': regime_data['regime'] if regime_data else 'SIDEWAYS',
                                    'regime_strength': regime_data.get('regime_strength', 'MEDIUM') if regime_data else 'MEDIUM',
                                    'orderbook': orderbook_status_flash,
                                    'volatility': 'high' if flash_analysis['indicators'].get('atr', 0) > flash_price * 0.02 else 'medium'
                                }

                                # Obtener portfolio metrics
                                portfolio_metrics_flash = {}
                                if hasattr(self.ml_system, 'paper_trader'):
                                    stats = self.ml_system.paper_trader.portfolio.get_statistics()
                                    portfolio_metrics_flash = {
                                        'win_rate': stats['win_rate'],
                                        'roi': stats['roi'],
                                        'max_drawdown': stats['max_drawdown'],
                                        'sharpe_ratio': stats.get('sharpe_ratio', 0),
                                        'total_trades': stats['total_trades']
                                    }

                                # RL Agent evalúa si abrir el flash trade
                                rl_flash_decision = await self.autonomy_controller.evaluate_trade_opportunity(
                                    pair=pair,
                                    signal=flash_signals,
                                    market_state=market_state_flash,
                                    portfolio_metrics=portfolio_metrics_flash
                                )

                                should_execute_flash = rl_flash_decision.get('should_trade', True)

                                # Aplicar modificador de tamaño de posición
                                if should_execute_flash and 'position_size_multiplier' in rl_flash_decision:
                                    flash_signals['rl_position_multiplier'] = rl_flash_decision['position_size_multiplier']
                                    flash_signals['rl_action'] = rl_flash_decision.get('chosen_action', 'UNKNOWN')

                            # Ejecutar flash trade solo si RL Agent lo aprueba
                            if should_execute_flash:
                                # ===== COMBINAR FEATURES PARA FLASH: Sentiment + ARSENAL =====
                                arsenal_ml_features_flash = {}
                                try:
                                    arsenal_ml_features_flash = self.feature_aggregator.get_ml_features(
                                        pair=pair,
                                        current_price=flash_price,
                                        base_features={},  # Será combinado con sentiment_features después
                                        ohlc_data=flash_df
                                    )
                                except Exception as e:
                                    logger.warning(f"No se pudieron obtener arsenal features para flash: {e}")

                                combined_flash_features = {**(sentiment_features or {}), **arsenal_ml_features_flash}

                                flash_trade_result = self.ml_system.process_signal(
                                    pair=pair,
                                    signal=flash_signals,
                                    indicators=flash_analysis['indicators'],
                                    current_price=flash_price,
                                    mtf_indicators=None,  # Flash signals don't use MTF
                                    sentiment_features=combined_flash_features,  # ARSENAL + Sentiment combined
                                    orderbook_features=orderbook_features,
                                    regime_features=regime_features
                                )

                                # Enhance analysis with trade result
                                if flash_trade_result:
                                    flash_analysis['trade_result'] = flash_trade_result
                                    # Add ML data if available
                                    if 'ml' in flash_signals:
                                        flash_analysis['ml_data'] = flash_signals['ml']
                                    # Add RL decision data
                                    if rl_flash_decision:
                                        flash_analysis['rl_decision'] = rl_flash_decision
                            else:
                                logger.info(f"🤖 RL Agent bloqueó flash trade en {pair}: {rl_flash_decision.get('chosen_action', 'SKIP')}")

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

                # 🔥 FIX CRÍTICO: Actualizar TODAS las posiciones abiertas cada ciclo
                if self.enable_paper_trading and self.ml_system:
                    paper_trader = self.ml_system.paper_trader
                    open_positions = list(paper_trader.portfolio.positions.keys())

                    if open_positions:
                        logger.info(f"🔍 Checking {len(open_positions)} open positions...")

                        for pair in open_positions:
                            # Get current price from cache or fetch fresh
                            current_price = self.current_prices.get(pair)

                            if current_price:
                                closed_trade = self.ml_system.update_position(pair, current_price)

                                if closed_trade:
                                    logger.info(f"✅ Position auto-closed: {pair} - {closed_trade.get('reason', 'UNKNOWN')}")

                                    # Send trade outcome to autonomous controller
                                    if self.autonomy_controller:
                                        # Fetch latest data for closed trade learning
                                        try:
                                            df = await self.fetch_ohlcv(pair, '1h')
                                            if df is not None and len(df) >= 50:
                                                analysis = self.analyzer.analyze_multi_timeframe({'1h': df})
                                                if analysis:
                                                    await self._process_trade_outcome_autonomous(
                                                        closed_trade,
                                                        analysis.get('indicators', {}),
                                                        None,  # sentiment_data
                                                        None,  # orderbook_analysis
                                                        None   # regime_data
                                                    )
                                        except Exception as e:
                                            logger.error(f"Error processing trade outcome for {pair}: {e}")
                            else:
                                logger.warning(f"⚠️ No current price for {pair}, skipping position check")

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
            # Derivar orderbook status de orderbook_data
            orderbook_status_outcome = 'NEUTRAL'
            if orderbook_data:
                pressure = orderbook_data.get('market_pressure', 'NEUTRAL')
                if pressure == 'BUY_PRESSURE':
                    orderbook_status_outcome = 'BUY_PRESSURE'
                elif pressure == 'SELL_PRESSURE':
                    orderbook_status_outcome = 'SELL_PRESSURE'

            # Construir estado de mercado para el RL Agent (formato actualizado)
            market_state = {
                # Nuevo formato requerido por RL Agent
                'rsi': indicators.get('rsi', 50),
                'regime': regime_data['regime'] if regime_data else 'SIDEWAYS',
                'regime_strength': regime_data.get('regime_strength', 'MEDIUM') if regime_data else 'MEDIUM',
                'orderbook': orderbook_status_outcome,
                'volatility': 'high' if indicators.get('atr', 0) > indicators.get('current_price', 1) * 0.02 else 'medium',

                # Campos adicionales para contexto (no usados en state representation)
                'sentiment_features': sentiment_data if sentiment_data else {},
                'win_rate': 0,
                'drawdown': 0
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

            # Preparar datos del trade (incluir campos de futuros)
            trade_data = {
                'pair': closed_trade.get('pair', 'UNKNOWN'),
                'action': closed_trade.get('action', 'UNKNOWN'),
                'trade_type': closed_trade.get('trade_type', 'SPOT'),
                'leverage': closed_trade.get('leverage', 1),
                'liquidated': closed_trade.get('liquidated', False),
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
