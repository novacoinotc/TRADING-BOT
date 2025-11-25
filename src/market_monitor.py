"""
Market Monitor Module
Continuously monitors trading pairs and analyzes them for signals
"""
import ccxt
import pandas as pd
import asyncio
import logging
import math
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
from src.trading.dynamic_threshold import get_dynamic_threshold_manager

# Binance Futures Integration (v2.0)
from src.binance_integration.binance_client import BinanceClient
from src.binance_integration.futures_trader import FuturesTrader
from src.binance_integration.position_monitor import PositionMonitor

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

        # ========== BINANCE FUTURES INTEGRATION (v2.0) ==========
        # Initialize Binance client and trading components
        self.binance_client = None
        self.futures_trader = None
        self.position_monitor = None

        if config.BINANCE_API_KEY and config.BINANCE_API_SECRET:
            try:
                # Build proxy config if needed
                proxies = None
                if config.USE_PROXY and config.PROXY_HOST and config.PROXY_PORT:
                    if config.PROXY_USERNAME and config.PROXY_PASSWORD:
                        proxy_url = f"http://{config.PROXY_USERNAME}:{config.PROXY_PASSWORD}@{config.PROXY_HOST}:{config.PROXY_PORT}"
                    else:
                        proxy_url = f"http://{config.PROXY_HOST}:{config.PROXY_PORT}"
                    proxies = {'http': proxy_url, 'https': proxy_url}

                # Initialize Binance client
                self.binance_client = BinanceClient(
                    api_key=config.BINANCE_API_KEY,
                    api_secret=config.BINANCE_API_SECRET,
                    base_url=config.BINANCE_BASE_URL,
                    proxies=proxies
                )
                logger.info("✅ Binance Client initialized")

                # Initialize Futures Trader
                self.futures_trader = FuturesTrader(
                    client=self.binance_client,
                    default_leverage=config.DEFAULT_LEVERAGE,
                    use_isolated_margin=config.USE_ISOLATED_MARGIN,
                    telegram_bot=self.notifier
                )
                logger.info("✅ Futures Trader initialized")

                # Initialize Position Monitor
                self.position_monitor = PositionMonitor(
                    client=self.binance_client,
                    update_interval=10,  # Update every 10 seconds (reducido de 5s para evitar rate limiting)
                    on_position_closed=self._on_position_closed
                )
                logger.info("✅ Position Monitor initialized")

                # Recuperar posiciones abiertas de Binance (si reinició)
                try:
                    open_positions = self.position_monitor.update_positions()
                    num_positions = len([p for p in open_positions.values() if float(p.get('positionAmt', 0)) != 0])
                    if num_positions > 0:
                        logger.info(f"🔄 Recuperadas {num_positions} posiciones abiertas de Binance al iniciar")
                        for symbol, pos in open_positions.items():
                            amt = float(pos.get('positionAmt', 0))
                            if amt != 0:
                                logger.info(f"   - {symbol}: {amt:+.4f} @ ${float(pos.get('entryPrice', 0)):,.2f}")
                    else:
                        logger.info("📊 No hay posiciones abiertas en Binance")
                except Exception as recover_error:
                    logger.error(f"⚠️ Error recuperando posiciones al iniciar: {recover_error}")

            except Exception as e:
                logger.error(f"❌ Failed to initialize Binance integration: {e}")
                self.binance_client = None
                self.futures_trader = None
                self.position_monitor = None
        else:
            logger.warning("⚠️ Binance API credentials not configured. Trading disabled.")

        # Initialize ML system (without paper trading)
        self.ml_system = MLIntegration(
            enable_ml=config.ENABLE_ML_SYSTEM,
            telegram_notifier=self.notifier
        ) if config.ENABLE_ML_SYSTEM else None

        # Auto-train ML if enough historical trades (40+)
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
        self.market_analysis_paused = False  # Pausa análisis pero permite cerrar trades existentes
        self.current_prices = {}  # Store current prices for tracking
        self.enable_flash = config.ENABLE_FLASH_SIGNALS
        self.auto_trade = config.AUTO_TRADE  # v2.0: Auto-trading con Binance
        self.autonomy_controller = None  # Will be set by main.py if autonomous mode enabled

        # Dynamic Threshold Manager - Experimenta con thresholds adaptativos
        self.dynamic_threshold_manager = get_dynamic_threshold_manager()

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

    def _on_position_closed(self, closed_info: Dict):
        """
        Callback cuando se cierra una posición (SL/TP alcanzado)

        Args:
            closed_info: Información del cierre desde PositionMonitor
        """
        try:
            logger.info(
                f"\n{'='*60}\n"
                f"📢 POSITION CLOSED NOTIFICATION\n"
                f"Symbol: {closed_info['symbol']}\n"
                f"Side: {closed_info['side']}\n"
                f"P&L: ${closed_info['realized_pnl']:+.2f} ({closed_info['realized_pnl_pct']:+.2f}%)\n"
                f"Reason: {closed_info['reason']}\n"
                f"{'='*60}\n"
            )

            # Notificar a Telegram
            if self.notifier:
                asyncio.create_task(self._notify_trade_closed(closed_info))

            # Actualizar RL Agent si está disponible
            if self.autonomy_controller:
                try:
                    # Calcular reward para RL Agent
                    reward = closed_info['realized_pnl']

                    # Actualizar agente con resultado
                    if hasattr(self.autonomy_controller, 'update_from_trade_result'):
                        self.autonomy_controller.update_from_trade_result(closed_info, reward)
                except Exception as e:
                    logger.error(f"❌ Error updating RL Agent: {e}")

            # ========== DYNAMIC THRESHOLD LEARNING ==========
            # Registrar resultado para que el sistema de threshold aprenda
            try:
                won = closed_info['realized_pnl'] > 0
                threshold_used = self.dynamic_threshold_manager.current_threshold

                # Construir condiciones de mercado del momento del trade
                market_conditions = {
                    'fear_greed_index': 50,  # Default, se actualizará con datos reales si están disponibles
                    'current_session': 'UNKNOWN',
                    'volatility': 'medium',
                    'confidence': 50
                }

                # Intentar obtener datos actuales del sentimiento
                if self.sentiment_system:
                    try:
                        base_currency = closed_info['symbol'].replace('USDT', '').replace('/USDT', '')
                        sentiment = self.sentiment_system.get_sentiment_features(f"{base_currency}/USDT")
                        if sentiment:
                            market_conditions['fear_greed_index'] = sentiment.get('fear_greed_index', 50)
                    except Exception:
                        pass

                # Obtener sesión actual
                if self.feature_aggregator and self.feature_aggregator.session_trading:
                    current_session, _ = self.feature_aggregator.session_trading.get_current_session()
                    market_conditions['current_session'] = current_session

                # Registrar experiencia
                self.dynamic_threshold_manager.record_trade_result(
                    won=won,
                    threshold_used=threshold_used,
                    market_conditions=market_conditions
                )

                logger.info(
                    f"🎚️ Threshold learning: {'WIN' if won else 'LOSS'} @ threshold={threshold_used:.2f} | "
                    f"Rachas: {self.dynamic_threshold_manager.consecutive_wins}W/{self.dynamic_threshold_manager.consecutive_losses}L"
                )

            except Exception as e:
                logger.warning(f"⚠️ Error registrando experiencia de threshold: {e}")

        except Exception as e:
            logger.error(f"❌ Error in _on_position_closed callback: {e}")

    async def _notify_trade_closed(self, closed_info: Dict):
        """Notifica a Telegram que se cerró un trade"""
        try:
            emoji = "✅" if closed_info['realized_pnl'] > 0 else "❌"
            message = (
                f"{emoji} **TRADE CERRADO - {'GANANCIA' if closed_info['realized_pnl'] > 0 else 'PÉRDIDA'}**\n\n"
                f"📌 Par: {closed_info['symbol']}\n"
                f"📊 Dirección: {closed_info['side']}\n"
                f"💰 Entrada: ${closed_info['entry_price']:,.2f}\n"
                f"💰 Salida: ${closed_info['exit_price']:,.2f}\n"
                f"📈 P&L: ${closed_info['realized_pnl']:+.2f} ({closed_info['realized_pnl_pct']:+.2f}%)\n"
                f"🔢 Cantidad: {closed_info['quantity']}\n"
                f"⚡ Leverage: {closed_info['leverage']}x\n"
                f"🏁 Razón: {closed_info['reason']}\n"
                f"🔖 Trade ID: {closed_info['trade_id']}"
            )
            await self.notifier.send_message(message)
        except Exception as e:
            logger.error(f"❌ Error notifying trade closed: {e}")

    async def _notify_trade_opened(self, trade_info: Dict):
        """Notifica a Telegram que se abrió un trade"""
        try:
            # Detectar si es flash trade
            is_flash = trade_info.get('is_flash', False)
            emoji = "⚡" if is_flash else "🟢"
            trade_type = "FLASH " if is_flash else ""

            message = (
                f"{emoji} **TRADE {trade_type}ABIERTO - Binance Futures**\n\n"
                f"📌 Par: {trade_info['symbol']}\n"
                f"📊 Dirección: {trade_info['side']}\n"
                f"⚡ Leverage: {trade_info['leverage']}x\n"
                f"💰 Entrada: ${trade_info['entry_price']:,.2f}\n"
                f"📊 Cantidad: {trade_info['quantity']}\n"
                f"🎯 Take Profit: ${trade_info['take_profit']:,.2f}\n"
                f"🛡️ Stop Loss: ${trade_info['stop_loss']:,.2f}\n"
                f"💵 USDT usado: ${trade_info['usdt_amount']:.2f}\n"
                f"🔖 Order ID: {trade_info['market_order_id']}"
            )
            await self.notifier.send_message(message)
        except Exception as e:
            logger.error(f"❌ Error notifying trade abierto: {e}")

    def _auto_train_ml_if_ready(self):
        """
        Auto-entrena el ML System si hay suficientes trades históricos (40+)
        v2.0: Usa historial de trades reales de Binance
        """
        try:
            if not self.ml_system:
                return

            # En v2.0, el entrenamiento se hace con datos históricos de Binance
            # o con el backtester si no hay trades reales todavía
            logger.debug("📊 ML auto-training en v2.0 se hace con historial de Binance")

            # TODO: Implementar obtención de trades históricos de Binance
            # y usar para entrenar ML

        except Exception as e:
            logger.warning(f"⚠️ Error en auto-entrenamiento ML: {e}")

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

                            # Execute news-triggered trade immediately (if auto-trading enabled)
                            if self.auto_trade and self.futures_trader:
                                # Get current price for execution
                                try:
                                    ticker = self.exchange.fetch_ticker(pair)
                                    news_price = ticker['last'] if ticker and 'last' in ticker else 0

                                    # VALIDACIÓN CRÍTICA: Verificar que el precio es válido
                                    if news_price is None or news_price <= 0 or math.isnan(news_price) or math.isinf(news_price):
                                        logger.error(
                                            f"❌ PRECIO INVÁLIDO de Binance para {pair}: {news_price}\n"
                                            f"   Rechazando trade de news para evitar corrupción del sistema"
                                        )
                                        continue  # Skip this news trade

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

                                            # Obtener portfolio metrics (v2.0: desde Binance)
                                            portfolio_metrics_news = {}
                                            if self.position_monitor:
                                                # TODO: Implementar get_statistics() en PositionMonitor
                                                # Por ahora, usar valores predeterminados
                                                portfolio_metrics_news = {
                                                    'win_rate': 0.0,
                                                    'roi': 0.0,
                                                    'max_drawdown': 0.0,
                                                    'sharpe_ratio': 0.0,
                                                    'total_trades': 0
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

                # VALIDACIÓN CRÍTICA: Verificar que el precio es válido
                if current_price_ticker is None or current_price_ticker <= 0 or math.isnan(current_price_ticker) or math.isinf(current_price_ticker):
                    logger.error(
                        f"❌ PRECIO INVÁLIDO de Binance para {pair}: {current_price_ticker}\n"
                        f"   Saltando análisis de este par para evitar corrupción del sistema"
                    )
                    return  # Skip this pair entirely
            except Exception as e:
                logger.warning(f"No se pudo obtener ticker para {pair}: {e}")
                return  # Skip this pair if we can't get price

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

            # ========== THRESHOLD DINÁMICO ==========
            # Calcular threshold adaptativo basado en condiciones de mercado
            dynamic_threshold = None
            threshold_explanation = ""
            try:
                # Obtener datos para el threshold dinámico
                # IMPORTANTE: Usar fear_greed_index_raw (0-100), NO fear_greed_index (0-1)
                fear_greed = sentiment_data.get('fear_greed_index_raw', 50) if sentiment_data else 50
                current_session = 'UNKNOWN'
                if self.feature_aggregator and self.feature_aggregator.session_trading:
                    current_session, _ = self.feature_aggregator.session_trading.get_current_session()

                # Determinar volatilidad del régimen
                volatility = 'medium'
                if regime_data:
                    vol_regime = regime_data.get('volatility', 'NORMAL')
                    if vol_regime in ['LOW', 'BAJA']:
                        volatility = 'low'
                    elif vol_regime in ['HIGH', 'ALTA', 'EXTREME', 'EXTREMA']:
                        volatility = 'high'

                # Calcular threshold dinámico
                dynamic_threshold, threshold_explanation = self.dynamic_threshold_manager.calculate_threshold(
                    fear_greed_index=fear_greed,
                    current_session=current_session,
                    volatility=volatility,
                    ml_confidence=sentiment_data.get('sentiment_strength', 50) if sentiment_data else 50,
                    recent_win_rate=self.dynamic_threshold_manager.get_recent_win_rate()
                )

                logger.info(f"🎚️ {pair}: {threshold_explanation}")

            except Exception as e:
                logger.warning(f"Error calculando threshold dinámico: {e}, usando default")
                dynamic_threshold = None  # Usará config.CONSERVATIVE_THRESHOLD

            # Perform advanced multi-timeframe analysis (con threshold dinámico)
            analysis = self.analyzer.analyze_multi_timeframe(dfs, dynamic_threshold=dynamic_threshold)

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

                    # Obtener posiciones abiertas (v2.0: desde Binance)
                    open_positions = []
                    if self.position_monitor:
                        positions = self.position_monitor.get_open_positions()

                        # 🔍 VALIDACIÓN CRÍTICA: Asegurar que positions es dict/list válido
                        if positions is None:
                            logger.warning("⚠️ get_open_positions() devolvió None, usando dict vacío")
                            positions = {}
                        elif isinstance(positions, str):
                            logger.error(f"❌ get_open_positions() devolvió STRING en lugar de dict: {positions[:200]}")
                            positions = {}
                        elif not isinstance(positions, (dict, list)):
                            logger.error(f"❌ get_open_positions() devolvió tipo inesperado: {type(positions)}")
                            positions = {}

                        # Extraer símbolos de posiciones abiertas (con validación robusta)
                        if isinstance(positions, dict):
                            open_positions = [p['symbol'] for p in positions.values() if isinstance(p, dict) and 'symbol' in p]
                        elif isinstance(positions, list):
                            open_positions = [p['symbol'] for p in positions if isinstance(p, dict) and 'symbol' in p]
                        else:
                            open_positions = []

                    # OBTENER ANÁLISIS DEL ARSENAL (preview)
                    arsenal_ml_features_preview = {}
                    arsenal_rl_extensions_preview = {}

                    try:
                        arsenal_ml_features_preview = self.feature_aggregator.get_ml_features(
                            pair=pair,
                            current_price=current_price,
                            base_features={},  # Diccionario vacío para preview
                            ohlc_data=dfs.get('1h')
                        )
                        # Verificar que es un diccionario
                        if not isinstance(arsenal_ml_features_preview, dict):
                            logger.warning(f"arsenal_ml_features_preview no es dict: {type(arsenal_ml_features_preview)}")
                            arsenal_ml_features_preview = {}
                    except Exception as e:
                        import traceback
                        logger.warning(f"Error obteniendo ML features del arsenal: {e}")
                        logger.warning(f"TRACEBACK:\n{traceback.format_exc()}")
                        arsenal_ml_features_preview = {}

                    try:
                        arsenal_rl_extensions_preview = self.feature_aggregator.get_rl_state_extensions(
                            pair=pair,
                            current_price=current_price,
                            open_positions=open_positions
                        )
                        # Verificar que es un diccionario
                        if not isinstance(arsenal_rl_extensions_preview, dict):
                            logger.warning(f"arsenal_rl_extensions_preview no es dict: {type(arsenal_rl_extensions_preview)}")
                            arsenal_rl_extensions_preview = {}
                    except Exception as e:
                        import traceback
                        logger.warning(f"Error obteniendo RL extensions del arsenal: {e}")
                        logger.warning(f"TRACEBACK:\n{traceback.format_exc()}")
                        arsenal_rl_extensions_preview = {}

                    # LOG DETALLADO DEL ARSENAL
                    logger.info(f"  📊 ARSENAL AVANZADO ({pair}):")

                    # VALIDACIÓN FINAL: Asegurar que son diccionarios antes de acceder
                    if not isinstance(arsenal_ml_features_preview, dict):
                        logger.warning(f"  ⚠️ arsenal_ml_features_preview no es dict en logging: {type(arsenal_ml_features_preview)}")
                        arsenal_ml_features_preview = {}
                    if not isinstance(arsenal_rl_extensions_preview, dict):
                        logger.warning(f"  ⚠️ arsenal_rl_extensions_preview no es dict en logging: {type(arsenal_rl_extensions_preview)}")
                        arsenal_rl_extensions_preview = {}

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

                    # Sentiment completo (VALIDAR que es dict antes de acceder)
                    if sentiment_features and isinstance(sentiment_features, dict):
                        fg_index = sentiment_features.get('fear_greed_index', 0.5) * 100
                        news_sentiment = sentiment_features.get('news_sentiment_overall', 0.5)
                        social_buzz = sentiment_features.get('social_buzz', 0)
                        logger.info(f"    📰 Sentiment: F&G={fg_index:.0f}, News={news_sentiment:.2f}, SocialBuzz={social_buzz}")
                    elif sentiment_features and not isinstance(sentiment_features, dict):
                        logger.warning(f"    ⚠️ sentiment_features no es dict: {type(sentiment_features)} - Valor: {sentiment_features}")

                    # ML Features TOTAL (VALIDAR tipo antes de len)
                    sentiment_count = len(sentiment_features) if isinstance(sentiment_features, dict) else 0
                    total_ml_features = sentiment_count + len(arsenal_ml_features_preview)
                    logger.info(f"    🧠 ML Features: {total_ml_features} total (Sentiment={sentiment_count} + Arsenal={len(arsenal_ml_features_preview)})")

                    # RL State Extensions
                    logger.info(f"    🤖 RL State: 19 dimensions (12 base + 7 arsenal)")

                except Exception as e:
                    import traceback
                    logger.error(f"  ❌ Error en análisis del arsenal: {e}")
                    logger.error(f"  📍 TRACEBACK COMPLETO:\n{traceback.format_exc()}")
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

                        # Obtener posiciones abiertas para correlation matrix (v2.0: desde Binance)
                        open_positions = []
                        if self.position_monitor:
                            positions = self.position_monitor.get_open_positions()

                            # 🔍 VALIDACIÓN: Asegurar que positions es dict/list válido
                            if positions is None:
                                positions = {}
                            elif isinstance(positions, str):
                                logger.error(f"❌ get_open_positions() devolvió STRING: {positions[:200]}")
                                positions = {}
                            elif not isinstance(positions, (dict, list)):
                                logger.error(f"❌ get_open_positions() tipo inesperado: {type(positions)}")
                                positions = {}

                            # Extraer símbolos con validación robusta
                            if isinstance(positions, dict):
                                open_positions = [p['symbol'] for p in positions.values() if isinstance(p, dict) and 'symbol' in p]
                            elif isinstance(positions, list):
                                open_positions = [p['symbol'] for p in positions if isinstance(p, dict) and 'symbol' in p]

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
                        import traceback
                        logger.error(f"Error applying Feature Aggregator to signal: {e}")
                        logger.error(f"TRACEBACK:\n{traceback.format_exc()}")
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

                # Execute trade with Binance if auto-trading enabled
                if self.auto_trade and self.futures_trader and signals['action'] != 'HOLD':
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
                            if not isinstance(arsenal_ml_features, dict):
                                logger.warning(f"arsenal_ml_features no es dict: {type(arsenal_ml_features)}")
                                arsenal_ml_features = {}

                            arsenal_rl_extensions = self.feature_aggregator.get_rl_state_extensions(
                                pair=pair,
                                current_price=current_price,
                                open_positions=open_positions
                            )
                            if not isinstance(arsenal_rl_extensions, dict):
                                logger.warning(f"arsenal_rl_extensions no es dict: {type(arsenal_rl_extensions)}")
                                arsenal_rl_extensions = {}
                        except Exception as e:
                            import traceback
                            logger.warning(f"No se pudieron obtener features del arsenal: {e}")
                            logger.warning(f"TRACEBACK:\n{traceback.format_exc()}")
                            arsenal_ml_features = {}
                            arsenal_rl_extensions = {}

                        # 🔧 FIX CRÍTICO: Copiar pattern data de ML features a RL extensions
                        # Los patterns se detectan en get_ml_features pero necesitan estar en RL extensions
                        # para que el composite score del RL Agent los use
                        if arsenal_ml_features.get('pattern_detected', False):
                            arsenal_rl_extensions['pattern_detected'] = True
                            arsenal_rl_extensions['pattern_type'] = arsenal_ml_features.get('pattern_type', 'NONE')
                            arsenal_rl_extensions['pattern_confidence'] = arsenal_ml_features.get('pattern_confidence', 0.0)
                            logger.debug(
                                f"🎯 Pattern copiado a RL extensions: {arsenal_rl_extensions['pattern_type']} "
                                f"(conf={arsenal_rl_extensions['pattern_confidence']:.2f})"
                            )

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
                            'pattern_confidence': arsenal_rl_extensions.get('pattern_confidence', 0.0),

                            # 22. Session-Based Trading
                            'current_session': arsenal_rl_extensions.get('current_session', 'UNKNOWN'),
                            'session_multiplier': arsenal_rl_extensions.get('session_multiplier', 1.0),

                            # 23. Order Flow Imbalance
                            'order_flow_bias': arsenal_rl_extensions.get('order_flow_bias', 'neutral'),
                            'order_flow_ratio': arsenal_rl_extensions.get('order_flow_ratio', 1.0),

                            # 🔧 FIX CRÍTICO: Confidence para RL Agent composite score
                            # El RL Agent usa confidence (0-100) para el base score
                            # Si signals tiene confidence, usarlo; sino, convertir score (0-10) a 0-100
                            'confidence': signals.get('confidence', signals.get('score', 50) * 10),
                        }
                        # ===== FIN ARSENAL EXTENSIONS =====

                        # Obtener portfolio metrics (v2.0: desde Binance)
                        portfolio_metrics = {}
                        if self.position_monitor:
                            # TODO: Implementar get_statistics() en PositionMonitor
                            # Por ahora, usar valores predeterminados
                            portfolio_metrics = {
                                'win_rate': 0.0,
                                'roi': 0.0,
                                'max_drawdown': 0.0,
                                'sharpe_ratio': 0.0,
                                'total_trades': 0
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

                        # 🔍 DEBUG: Log resultado de decisión del RL Agent
                        composite = rl_decision.get('composite_score', 0)
                        chosen_action = rl_decision.get('chosen_action', 'UNKNOWN')
                        if should_execute_trade:
                            logger.info(
                                f"✅ RL APROBÓ trade {pair}: {chosen_action} | "
                                f"composite={composite:.2f} | leverage={rl_decision.get('leverage', 1)}x"
                            )
                        else:
                            logger.warning(
                                f"❌ RL BLOQUEÓ trade {pair}: {chosen_action} | "
                                f"composite={composite:.2f} | SKIP - señal no suficientemente fuerte"
                            )

                        # Aplicar parámetros del RL Agent (tamaño, trade_type, leverage)
                        if should_execute_trade and 'position_size_multiplier' in rl_decision:
                            # Pasar el multiplicador a la señal para que ml_system lo use
                            signals['rl_position_multiplier'] = rl_decision['position_size_multiplier']
                            signals['rl_action'] = rl_decision.get('chosen_action', 'UNKNOWN')
                            # Pasar trade_type y leverage para futuros
                            signals['trade_type'] = rl_decision.get('trade_type', 'FUTURES')  # Default FUTURES
                            signals['leverage'] = rl_decision.get('leverage', 1)  # 1x = sin apalancamiento

                            # ✅ CRÍTICO: RE-CALCULAR TPs CON AGGRESSIVENESS CORRECTO
                            # Determinar aggressiveness basado en chosen_action del RL Agent
                            aggressiveness = 'MEDIUM'  # Default
                            if chosen_action == 'FUTURES_LOW':
                                aggressiveness = 'LOW'
                            elif chosen_action == 'FUTURES_MEDIUM':
                                aggressiveness = 'MEDIUM'
                            elif chosen_action == 'FUTURES_HIGH':
                                aggressiveness = 'HIGH'

                            logger.info(f"🎯 Recalculando TPs con aggressiveness={aggressiveness} (based on {chosen_action})")

                            # Re-calcular SL/TP con aggressiveness correcto
                            if 'atr' in analysis.get('indicators', {}):
                                atr = analysis['indicators']['atr']
                                current_price = analysis['indicators'].get('current_price', current_price)
                                action = signals.get('action', 'BUY')

                                # Re-calcular usando el analyzer correspondiente
                                if hasattr(self, 'flash_analyzer') and analysis.get('type') == 'FLASH':
                                    sl_tp = self.flash_analyzer._calculate_flash_sl_tp(
                                        current_price, action, atr, aggressiveness=aggressiveness
                                    )
                                elif hasattr(self, 'analyzer'):
                                    levels = analysis['indicators'].get('support_resistance', {})
                                    sl_tp = self.analyzer._calculate_sl_tp(
                                        current_price, action, atr, levels, aggressiveness=aggressiveness
                                    )
                                else:
                                    sl_tp = None

                                if sl_tp:
                                    signals['stop_loss'] = sl_tp['stop_loss']
                                    signals['take_profit'] = sl_tp['take_profit']
                                    signals['risk_reward'] = sl_tp['risk_reward']
                                    logger.info(f"✅ TPs recalculados: TP1={sl_tp.get('take_profit_1', 'N/A')}, "
                                              f"TP2={sl_tp.get('take_profit_2', 'N/A')}, "
                                              f"TP3={sl_tp.get('take_profit_3', 'N/A')}")

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

                        # ===== v2.0: EJECUTAR TRADE EN BINANCE FUTURES =====
                        try:
                            # Convertir símbolo para Binance (BTC/USDT → BTCUSDT)
                            binance_symbol = pair.replace('/', '')

                            # VALIDACIÓN: Verificar que no haya posición abierta en este par
                            if self.position_monitor.has_position(binance_symbol):
                                logger.warning(
                                    f"⚠️ Ya hay posición abierta en {binance_symbol}, "
                                    f"consultando RL Agent sobre qué hacer..."
                                )

                                # 🆕 v1.5: Consultar RL Agent para decisión informada sobre posición existente
                                if (hasattr(self, 'autonomy_controller') and self.autonomy_controller and
                                    hasattr(self.autonomy_controller, 'rl_agent') and self.autonomy_controller.rl_agent):
                                    try:
                                        # Obtener datos de la posición actual
                                        open_position = self.position_monitor.get_open_positions().get(binance_symbol, {})

                                        # Preparar estado actual para RL Agent
                                        state = self.autonomy_controller.rl_agent.get_state_representation(market_state)

                                        # Consultar RL Agent con todos los asistentes
                                        rl_decision = await self.autonomy_controller.rl_agent.make_informed_decision(
                                            state=state,
                                            position_data=open_position,
                                            trade_manager=self.autonomy_controller.trade_manager if hasattr(self.autonomy_controller, 'trade_manager') else None,
                                            ml_system=self.ml_system if hasattr(self, 'ml_system') else None,
                                            parameter_optimizer=self.autonomy_controller.parameter_optimizer if hasattr(self.autonomy_controller, 'parameter_optimizer') else None
                                        )

                                        # Si RL Agent decide cerrar la posición
                                        if rl_decision.get('action') == 'CLOSE_POSITION':
                                            logger.info(f"🧠 RL Agent decidió cerrar posición en {binance_symbol}")
                                            logger.info(f"   Razones: {rl_decision.get('reasons', [])}")
                                            logger.info(f"   Confianza: {rl_decision.get('confidence', 0):.1f}%")

                                            try:
                                                # Cerrar posición a través de futures_trader
                                                close_result = self.futures_trader.close_position(
                                                    symbol=binance_symbol,
                                                    reason="RL_AGENT_DECISION"
                                                )

                                                if close_result:
                                                    logger.info(f"✅ Posición cerrada por decisión del RL Agent: {binance_symbol}")
                                                else:
                                                    logger.error(f"❌ Error cerrando posición {binance_symbol}")
                                            except Exception as e:
                                                logger.error(f"❌ Error ejecutando cierre: {e}", exc_info=True)
                                        else:
                                            logger.info(f"🧠 RL Agent decidió MANTENER posición en {binance_symbol}")
                                            logger.info(f"   Acción: {rl_decision.get('action')}")

                                    except Exception as e:
                                        logger.error(f"❌ Error consultando RL Agent para posición existente: {e}", exc_info=True)

                                # Continuar al siguiente análisis sin abrir trade duplicado
                                pass  # El código continúa normalmente sin abrir trade
                            else:
                                # No hay posición abierta, proceder con el trade
                                # Determinar cantidad en USDT por trade
                                usdt_amount = getattr(config, 'TRADE_AMOUNT_USDT', 100.0)  # Default $100

                                # Calcular stop loss y take profit desde signals
                                stop_loss_pct = 2.0  # Default 2%
                                take_profit_pct = 3.0  # Default 3%

                                if 'stop_loss' in signals and signals['stop_loss']:
                                    # Manejar si es dict o float
                                    sl_price = signals['stop_loss']
                                    if isinstance(sl_price, dict):
                                        # Si es dict, tomar el valor 'price' o 'value'
                                        sl_price = sl_price.get('price', sl_price.get('value', current_price))

                                    # Validar que sl_price sea numérico
                                    if isinstance(sl_price, (int, float)) and sl_price > 0:
                                        stop_loss_pct = abs((sl_price - current_price) / current_price * 100)

                                        # 🔧 VALIDACIÓN CRÍTICA: SL debe estar a distancia mínima segura
                                        # Mínimo 1.5% para considerar: volatilidad normal + comisiones (0.04%) + spread
                                        min_sl_pct = 1.5
                                        if stop_loss_pct < min_sl_pct:
                                            logger.warning(
                                                f"⚠️ SL demasiado cerca del entry ({stop_loss_pct:.3f}% < {min_sl_pct}%), "
                                                f"ajustando a mínimo seguro de 2.5%"
                                            )
                                            stop_loss_pct = 2.5  # Usar 2.5% como mínimo seguro
                                    else:
                                        logger.warning(f"⚠️ stop_loss inválido: {sl_price}, usando default 2.5%")
                                        stop_loss_pct = 2.5

                                if 'take_profit' in signals and signals['take_profit']:
                                    tp_price = signals['take_profit']

                                    # 🔧 FIX: Extraer TP1 correctamente del dict
                                    if isinstance(tp_price, dict):
                                        # Intentar extraer tp1 (el TP principal para scalping)
                                        if 'tp1' in tp_price:
                                            tp_price = tp_price['tp1']
                                            logger.debug(f"✅ TP extraído de dict: tp1=${tp_price}")
                                        elif 'take_profit_1' in tp_price:
                                            tp_price = tp_price['take_profit_1']
                                            logger.debug(f"✅ TP extraído de dict: take_profit_1=${tp_price}")
                                        elif 'price' in tp_price:
                                            tp_price = tp_price['price']
                                            logger.debug(f"✅ TP extraído de dict: price=${tp_price}")
                                        elif 'value' in tp_price:
                                            tp_price = tp_price['value']
                                            logger.debug(f"✅ TP extraído de dict: value=${tp_price}")
                                        else:
                                            # Log para debug y usar primer valor numérico
                                            logger.warning(f"⚠️ Estructura TP no estándar: {list(tp_price.keys())}")
                                            tp_values = [v for k, v in tp_price.items() if isinstance(v, (int, float)) and v > 0]
                                            tp_price = tp_values[0] if tp_values else None

                                    if isinstance(tp_price, (int, float)) and tp_price > 0:
                                        take_profit_pct = abs((tp_price - current_price) / current_price * 100)
                                        logger.info(f"✅ TP calculado: {take_profit_pct:.2f}% (TP: ${tp_price:.2f}, Current: ${current_price:.2f})")

                                        # 🔍 VALIDACIÓN: TP debe estar a distancia mínima
                                        min_tp_pct = 0.2  # Mínimo 0.2% de diferencia
                                        if take_profit_pct < min_tp_pct:
                                            logger.warning(
                                                f"⚠️ TP demasiado cerca del entry ({take_profit_pct:.2f}% < {min_tp_pct}%), "
                                                f"usando default 3%"
                                            )
                                            take_profit_pct = 3.0
                                    else:
                                        logger.warning(f"⚠️ take_profit inválido: {tp_price}, usando default 3%")
                                        take_profit_pct = 3.0

                                # Determinar leverage desde signals o usar default
                                leverage = signals.get('leverage', config.DEFAULT_LEVERAGE)

                                # Aplicar multiplicador de RL Agent si está disponible
                                if 'rl_position_multiplier' in signals:
                                    usdt_amount *= signals['rl_position_multiplier']

                                # VALIDACIÓN: Verificar cantidad mínima después de multiplicador
                                if usdt_amount < 10.0:
                                    logger.warning(
                                        f"⚠️ Cantidad muy pequeña después de multiplicador RL: "
                                        f"${usdt_amount:.2f} (mínimo: $10), omitiendo trade"
                                    )
                                else:
                                    logger.info(
                                        f"🚀 Ejecutando trade Binance: {pair} {signals['action']}\n"
                                        f"   USDT: ${usdt_amount:.2f} | Leverage: {leverage}x\n"
                                        f"   SL: {stop_loss_pct:.2f}% | TP: {take_profit_pct:.2f}%"
                                    )

                                    # Ejecutar trade con Binance Futures
                                    binance_result = self.futures_trader.open_position(
                                        symbol=binance_symbol,
                                        side=signals['action'],  # 'BUY' o 'SELL'
                                        usdt_amount=usdt_amount,
                                        stop_loss_pct=stop_loss_pct,
                                        take_profit_pct=take_profit_pct,
                                        leverage=leverage,
                                        current_price=current_price
                                    )

                                    if binance_result:
                                        logger.info(f"✅ Trade ejecutado en Binance: {binance_result['market_order_id']}")

                                        # Notificar a Telegram
                                        if self.notifier:
                                            await self._notify_trade_opened(binance_result)

                                        # Añadir resultado de Binance al analysis
                                        analysis['binance_trade'] = binance_result

                                        # Registrar en RL Agent para aprendizaje futuro
                                        if self.autonomy_controller:
                                            try:
                                                # El RL Agent aprenderá cuando se cierre (vía _on_position_closed)
                                                logger.debug(f"📝 Trade registrado - aprendizaje al cerrar posición")
                                            except Exception as e:
                                                logger.error(f"❌ Error registrando trade en RL Agent: {e}")
                                    else:
                                        logger.error(
                                            f"❌ No se pudo ejecutar trade en Binance: "
                                            f"{binance_symbol} {signals['action']} ${usdt_amount:.2f}"
                                        )

                        except Exception as e:
                            symbol_attempted = pair.replace('/', '')
                            side_attempted = signals.get('action', 'UNKNOWN')
                            amount_attempted = getattr(config, 'TRADE_AMOUNT_USDT', 100.0)
                            logger.error(
                                f"❌ Error ejecutando trade Binance: "
                                f"{symbol_attempted} {side_attempted} ${amount_attempted:.2f} - {str(e)}",
                                exc_info=True
                            )
                        # ===== FIN EJECUCIÓN BINANCE =====
                    else:
                        logger.info(f"🤖 RL Agent bloqueó trade en {pair}: {rl_decision.get('chosen_action', 'SKIP')}")

                # v2.0: Position updates are handled automatically by PositionMonitor
                # The _on_position_closed callback will be triggered when SL/TP is hit

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
                        arsenal_flash_ml = {}
                        arsenal_flash_rl = {}
                        try:
                            # OBTENER ANÁLISIS DEL ARSENAL para flash
                            arsenal_flash_ml = self.feature_aggregator.get_ml_features(
                                pair=pair,
                                current_price=flash_price,
                                base_features={},  # Preview de flash
                                ohlc_data=flash_df
                            )
                            if not isinstance(arsenal_flash_ml, dict):
                                logger.warning(f"arsenal_flash_ml no es dict: {type(arsenal_flash_ml)}")
                                arsenal_flash_ml = {}

                            arsenal_flash_rl = self.feature_aggregator.get_rl_state_extensions(
                                pair=pair,
                                current_price=flash_price,
                                open_positions=open_positions
                            )
                            if not isinstance(arsenal_flash_rl, dict):
                                logger.warning(f"arsenal_flash_rl no es dict: {type(arsenal_flash_rl)}")
                                arsenal_flash_rl = {}

                            # LOG DETALLADO FLASH
                            logger.info(f"  ⚡ ARSENAL FLASH ({pair}):")
                            session_f = arsenal_flash_rl.get('current_session', 'UNKNOWN')
                            session_mult_f = arsenal_flash_rl.get('session_multiplier', 1.0)
                            flow_bias_f = arsenal_flash_rl.get('order_flow_bias', 'neutral')
                            flow_ratio_f = arsenal_flash_rl.get('order_flow_ratio', 1.0)
                            logger.info(f"    ⏰ Session: {session_f} ({session_mult_f:.2f}x) | 💧 Flow: {flow_bias_f} ({flow_ratio_f:.2f})")

                        except Exception as e:
                            import traceback
                            logger.debug(f"  ⚡ Arsenal flash preview error: {e}")
                            logger.debug(f"TRACEBACK:\n{traceback.format_exc()}")

                        # 🔧 FIX CRÍTICO: Copiar pattern data de ML features a RL extensions (FLASH)
                        if arsenal_flash_ml.get('pattern_detected', False):
                            arsenal_flash_rl['pattern_detected'] = True
                            arsenal_flash_rl['pattern_type'] = arsenal_flash_ml.get('pattern_type', 'NONE')
                            arsenal_flash_rl['pattern_confidence'] = arsenal_flash_ml.get('pattern_confidence', 0.0)
                            logger.debug(
                                f"🎯 Pattern copiado a RL extensions (flash): {arsenal_flash_rl['pattern_type']} "
                                f"(conf={arsenal_flash_rl['pattern_confidence']:.2f})"
                            )
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

                                # Obtener posiciones abiertas (v2.0: desde Binance)
                                open_positions_flash = []
                                if self.position_monitor:
                                    positions = self.position_monitor.get_open_positions()

                                    # 🔍 VALIDACIÓN: Asegurar que positions es dict/list válido
                                    if positions is None:
                                        positions = {}
                                    elif isinstance(positions, str):
                                        logger.error(f"❌ get_open_positions() devolvió STRING: {positions[:200]}")
                                        positions = {}
                                    elif not isinstance(positions, (dict, list)):
                                        logger.error(f"❌ get_open_positions() tipo inesperado: {type(positions)}")
                                        positions = {}

                                    # Extraer símbolos con validación robusta
                                    if isinstance(positions, dict):
                                        open_positions_flash = [p['symbol'] for p in positions.values() if isinstance(p, dict) and 'symbol' in p]
                                    elif isinstance(positions, list):
                                        open_positions_flash = [p['symbol'] for p in positions if isinstance(p, dict) and 'symbol' in p]

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
                                import traceback
                                logger.error(f"Error applying Arsenal to flash signal: {e}")
                                logger.error(f"TRACEBACK:\n{traceback.format_exc()}")
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

                        # Execute trade with Binance for flash signal if auto-trading enabled
                        if self.auto_trade and self.futures_trader and flash_signals['action'] != 'HOLD':
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
                                    'volatility': 'high' if flash_analysis['indicators'].get('atr', 0) > flash_price * 0.02 else 'medium',
                                    # 🔧 FIX CRÍTICO: Confidence para RL Agent composite score
                                    'confidence': flash_signals.get('confidence', flash_signals.get('score', 50) * 10),
                                }

                                # Obtener portfolio metrics (v2.0: desde Binance)
                                portfolio_metrics_flash = {}
                                if self.position_monitor:
                                    # TODO: Implementar get_statistics() en PositionMonitor
                                    # Por ahora, usar valores predeterminados
                                    portfolio_metrics_flash = {
                                        'win_rate': 0.0,
                                        'roi': 0.0,
                                        'max_drawdown': 0.0,
                                        'sharpe_ratio': 0.0,
                                        'total_trades': 0
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
                                    if not isinstance(arsenal_ml_features_flash, dict):
                                        logger.warning(f"arsenal_ml_features_flash no es dict: {type(arsenal_ml_features_flash)}")
                                        arsenal_ml_features_flash = {}
                                except Exception as e:
                                    import traceback
                                    logger.warning(f"No se pudieron obtener arsenal features para flash: {e}")
                                    logger.warning(f"TRACEBACK:\n{traceback.format_exc()}")
                                    arsenal_ml_features_flash = {}

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

                                # ===== v2.0: EJECUTAR FLASH TRADE EN BINANCE FUTURES =====
                                try:
                                    # Convertir símbolo para Binance
                                    flash_binance_symbol = pair.replace('/', '')

                                    # VALIDACIÓN: Verificar que no haya posición abierta en este par
                                    if self.position_monitor.has_position(flash_binance_symbol):
                                        logger.warning(
                                            f"⚠️ Ya hay posición abierta en {flash_binance_symbol}, "
                                            f"no abrir flash duplicada. Omitiendo señal {flash_signals['action']}."
                                        )
                                    else:
                                        # No hay posición, proceder con flash trade
                                        # Determinar cantidad en USDT (flash signals suelen usar menos)
                                        usdt_amount = getattr(config, 'FLASH_TRADE_AMOUNT_USDT', 50.0)  # Default $50 para flash

                                        # Calcular stop loss y take profit
                                        stop_loss_pct = 2.5  # Flash = SL más amplio
                                        take_profit_pct = 2.0  # Flash = TP más cercano

                                        if 'stop_loss' in flash_signals and flash_signals['stop_loss']:
                                            sl_price = flash_signals['stop_loss']
                                            # 🔧 FIX: Validar que sl_price es un número
                                            if isinstance(sl_price, dict):
                                                logger.warning(f"⚠️ sl_price es dict, extrayendo valor: {sl_price}")
                                                sl_price = sl_price.get('price', sl_price.get('value', 0))
                                            if isinstance(sl_price, (int, float)) and sl_price > 0:
                                                stop_loss_pct = abs((sl_price - flash_price) / flash_price * 100)

                                        if 'take_profit' in flash_signals and flash_signals['take_profit']:
                                            tp_price = flash_signals['take_profit']
                                            # 🔧 FIX: Validar que tp_price es un número
                                            if isinstance(tp_price, dict):
                                                logger.warning(f"⚠️ tp_price es dict, extrayendo valor: {tp_price}")
                                                tp_price = tp_price.get('price', tp_price.get('value', 0))
                                            if isinstance(tp_price, (int, float)) and tp_price > 0:
                                                take_profit_pct = abs((tp_price - flash_price) / flash_price * 100)

                                        # Leverage para flash (menor que señales normales)
                                        leverage = flash_signals.get('leverage', max(1, config.DEFAULT_LEVERAGE - 1))

                                        # Aplicar multiplicador de RL Agent
                                        if 'rl_position_multiplier' in flash_signals:
                                            usdt_amount *= flash_signals['rl_position_multiplier']

                                        # VALIDACIÓN: Verificar cantidad mínima
                                        if usdt_amount < 10.0:
                                            logger.warning(
                                                f"⚠️ Flash trade: cantidad muy pequeña después de multiplicador RL: "
                                                f"${usdt_amount:.2f} (mínimo: $10), omitiendo trade"
                                            )
                                        else:
                                            logger.info(
                                                f"⚡ Ejecutando FLASH trade Binance: {pair} {flash_signals['action']}\n"
                                                f"   USDT: ${usdt_amount:.2f} | Leverage: {leverage}x\n"
                                                f"   SL: {stop_loss_pct:.2f}% | TP: {take_profit_pct:.2f}%"
                                            )

                                            # Ejecutar trade con Binance Futures
                                            flash_binance_result = self.futures_trader.open_position(
                                                symbol=flash_binance_symbol,
                                                side=flash_signals['action'],
                                                usdt_amount=usdt_amount,
                                                stop_loss_pct=stop_loss_pct,
                                                take_profit_pct=take_profit_pct,
                                                leverage=leverage,
                                                current_price=flash_price
                                            )

                                            if flash_binance_result:
                                                logger.info(f"✅ Flash trade ejecutado: {flash_binance_result['market_order_id']}")

                                                # Notificar a Telegram con tag de FLASH
                                                if self.notifier:
                                                    flash_binance_result['is_flash'] = True
                                                    await self._notify_trade_opened(flash_binance_result)

                                                # Añadir resultado al analysis
                                                flash_analysis['binance_trade'] = flash_binance_result
                                            else:
                                                logger.error(
                                                    f"❌ No se pudo ejecutar flash trade en Binance: "
                                                    f"{flash_binance_symbol} {flash_signals['action']} ${usdt_amount:.2f}"
                                                )

                                except Exception as e:
                                    flash_symbol_attempted = pair.replace('/', '')
                                    flash_side_attempted = flash_signals.get('action', 'UNKNOWN')
                                    flash_amount_attempted = getattr(config, 'FLASH_TRADE_AMOUNT_USDT', 50.0)
                                    logger.error(
                                        f"❌ Error ejecutando flash trade Binance: "
                                        f"{flash_symbol_attempted} {flash_side_attempted} ${flash_amount_attempted:.2f} - {str(e)}",
                                        exc_info=True
                                    )
                                # ===== FIN EJECUCIÓN FLASH BINANCE =====
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
        auto_trade_status = "✅ Activo" if self.auto_trade else "❌ Inactivo"
        trading_mode = f"{'🧪 TESTNET' if config.BINANCE_TESTNET else '🔴 LIVE'}"

        startup_message = (
            "🤖 <b>Bot de Señales Iniciado</b>\n\n"
            f"📊 Monitoreando: {pairs_display}\n"
            f"⏱️ Intervalo: {self.check_interval}s\n"
            f"📈 Timeframe conservador: {config.TIMEFRAME} (1h/4h/1d)\n"
            f"⚡ Señales flash: {flash_status} ({self.flash_timeframe})\n"
            f"💰 Binance Futures: {trading_mode}\n"
            f"🔄 Auto-Trading: {auto_trade_status}"
        )

        if self.ml_system:
            startup_message += "\n🧠 Machine Learning: ✅ Activo"

        startup_message += "\n📍 Reporte diario: 9 PM CDMX"

        await self.notifier.send_status_message(startup_message)

        self.is_running = True
        iteration = 0
        stats_iteration = 0  # Send stats every N iterations

        while self.is_running:
            try:
                iteration += 1
                logger.info(f"=== Iteration {iteration} - {datetime.now()} ===")

                # Si está pausado, solo actualizar posiciones existentes
                if self.market_analysis_paused:
                    logger.info("⏸️  Análisis de mercado PAUSADO - Solo monitoreando trades abiertos")
                else:
                    # Analyze all pairs
                    for pair in self.trading_pairs:
                        await self.analyze_pair(pair)
                        # Small delay between pairs to avoid rate limits
                        await asyncio.sleep(2)

                # v2.0: Position monitoring is handled automatically by PositionMonitor
                # The _on_position_closed callback will be triggered when SL/TP is hit
                # No need to manually check positions here

                # Update pending signals with current prices
                if self.tracker:
                    self.tracker.check_pending_signals(self.current_prices)

                # Check if it's time for daily report
                if self.reporter:
                    await self.reporter.check_and_send_report()

                # TODO v2.0: Implement trading stats from Binance PositionMonitor
                # Stats should come from real trade history on Binance
                # if self.auto_trade and self.position_monitor:
                #     stats_iteration += 1
                #     if stats_iteration >= 30:  # ~1 hour
                #         try:
                #             stats = self.position_monitor.get_trading_stats()
                #             await self.notifier.send_trading_stats(stats)
                #             stats_iteration = 0
                #         except Exception as e:
                #             logger.error(f"Error sending trading stats: {e}")

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

            # Obtener métricas del portfolio (v2.0: desde Binance)
            portfolio_metrics = {}
            if self.position_monitor:
                # TODO: Implementar get_statistics() en PositionMonitor
                # Por ahora, usar valores predeterminados
                market_state['win_rate'] = 0.0
                market_state['drawdown'] = 0.0

                portfolio_metrics = {
                    'win_rate': 0.0,
                    'roi': 0.0,
                    'max_drawdown': 0.0,
                    'sharpe_ratio': 0.0,
                    'profit_factor': 1.0,
                    'total_trades': 0
                }

            # Preparar datos del trade (incluir campos de futuros)
            # CRÍTICO: Mapear campos correctos de closed_trade
            trade_data = {
                'pair': closed_trade.get('pair', 'UNKNOWN'),
                'action': closed_trade.get('side', 'UNKNOWN'),  # FIX: side, no action
                'trade_type': closed_trade.get('trade_type', 'SPOT'),
                'leverage': closed_trade.get('leverage', 1),
                'liquidated': closed_trade.get('liquidated', False),
                'entry_price': closed_trade.get('entry_price', 0),
                'exit_price': closed_trade.get('exit_price', 0),
                'profit_pct': closed_trade.get('pnl_pct', 0),  # FIX CRÍTICO: pnl_pct, no profit_pct
                'profit_amount': closed_trade.get('pnl', 0),  # FIX: pnl, no profit
                'duration': closed_trade.get('duration', 0),
                'exit_reason': closed_trade.get('reason', 'UNKNOWN')  # FIX: reason, no exit_reason
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

    def pause_analysis(self):
        """
        Pausa el análisis de nuevos pares, pero sigue monitoreando posiciones abiertas
        Útil antes de hacer export para evitar discrepancias
        """
        self.market_analysis_paused = True
        logger.info("⏸️  Análisis de mercado PAUSADO - Solo monitoreando trades abiertos")

    def resume_analysis(self):
        """
        Reanuda el análisis de mercado normal
        """
        self.market_analysis_paused = False
        logger.info("▶️  Análisis de mercado RESUMIDO - Analizando todos los pares")
