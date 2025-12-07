"""
GPT Data Provider - Comprehensive Data Integration for GPT

This module collects ALL available data sources and formats them
for GPT consumption. GPT needs the complete picture to make optimal decisions.

Data Sources Integrated:
1. Technical Indicators (RSI, MACD, EMA, BB, ATR, ADX, Volume)
2. Sentiment Analysis (Fear & Greed, News Sentiment, CryptoPanic)
3. Order Book Analysis (Bid/Ask Imbalance, Market Pressure)
4. Arsenal Advanced Modules:
   - Correlation Matrix (diversification analysis)
   - Liquidation Heatmap (liquidation zones)
   - Funding Rate (contrarian signals)
   - Volume Profile (POC, Value Area)
   - Pattern Recognition (chart patterns)
   - Session Trading (market sessions)
   - Order Flow (bid/ask imbalance)
5. Market Regime Detection
6. Multi-Timeframe Analysis
7. ML/RL Predictions (as advisory input)
8. Recent Trade History & Performance
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class GPTDataProvider:
    """
    Central data provider that collects and formats ALL available data
    for GPT to make informed trading decisions.

    This is the "eyes and ears" of GPT - it sees everything.
    """

    def __init__(
        self,
        config: Any,
        feature_aggregator: Optional[Any] = None,
        sentiment_analyzer: Optional[Any] = None,
        orderbook_analyzer: Optional[Any] = None,
        news_collector: Optional[Any] = None,
        market_monitor: Optional[Any] = None,
        ml_system: Optional[Any] = None,
        rl_agent: Optional[Any] = None,
        portfolio: Optional[Any] = None
    ):
        """
        Initialize GPT Data Provider with all available data sources.

        Args:
            config: Configuration object
            feature_aggregator: Arsenal advanced modules aggregator
            sentiment_analyzer: Sentiment analysis system
            orderbook_analyzer: Order book analyzer
            news_collector: CryptoPanic and news collector
            market_monitor: Market monitoring system
            ml_system: ML prediction system
            rl_agent: Reinforcement learning agent
            portfolio: Portfolio/position manager
        """
        self.config = config
        self.feature_aggregator = feature_aggregator
        self.sentiment_analyzer = sentiment_analyzer
        self.orderbook_analyzer = orderbook_analyzer
        self.news_collector = news_collector
        self.market_monitor = market_monitor
        self.ml_system = ml_system
        self.rl_agent = rl_agent
        self.portfolio = portfolio

        # Data cache to avoid redundant API calls
        self.cache: Dict[str, Any] = {}
        self.cache_ttl_seconds = 30  # Cache TTL

        logger.info("🧠 GPT Data Provider initialized - ALL data sources connected")
        self._log_available_sources()

    def _log_available_sources(self):
        """Log which data sources are available"""
        sources = {
            "Feature Aggregator (Arsenal)": self.feature_aggregator is not None,
            "Sentiment Analyzer": self.sentiment_analyzer is not None,
            "Order Book Analyzer": self.orderbook_analyzer is not None,
            "News Collector (CryptoPanic)": self.news_collector is not None,
            "Market Monitor": self.market_monitor is not None,
            "ML System": self.ml_system is not None,
            "RL Agent": self.rl_agent is not None,
            "Portfolio": self.portfolio is not None
        }

        active = sum(1 for v in sources.values() if v)
        total = len(sources)

        logger.info(f"📊 Data sources available: {active}/{total}")
        for source, available in sources.items():
            status = "✅" if available else "❌"
            logger.debug(f"   {status} {source}")

    async def get_complete_market_data(
        self,
        pair: str,
        indicators: Dict,
        ohlc_data: Optional[Any] = None,
        include_news: bool = True
    ) -> Dict[str, Any]:
        """
        Collect ALL available data for a trading pair.

        This is the primary method GPT uses to "see" the market.

        Args:
            pair: Trading pair (e.g., 'BTC/USDT')
            indicators: Technical indicators dict
            ohlc_data: OHLCV data for advanced analysis
            include_news: Whether to include news data

        Returns:
            Comprehensive market data dict
        """
        data = {
            "pair": pair,
            "timestamp": datetime.now().isoformat(),
            "technical": {},
            "sentiment": {},
            "orderbook": {},
            "arsenal": {},
            "regime": {},
            "ml_prediction": {},
            "rl_recommendation": {},
            "portfolio": {},
            "news": {},
            "summary": {}
        }

        current_price = indicators.get('current_price', 0)

        # 1. TECHNICAL INDICATORS
        data["technical"] = self._format_technical_indicators(indicators)

        # 2. SENTIMENT DATA
        data["sentiment"] = await self._get_sentiment_data(pair, include_news)

        # 3. ORDER BOOK DATA
        data["orderbook"] = await self._get_orderbook_data(pair)

        # 4. ARSENAL ADVANCED MODULES
        data["arsenal"] = self._get_arsenal_data(pair, current_price, ohlc_data)

        # 5. MARKET REGIME
        data["regime"] = self._get_regime_data(pair, indicators)

        # 6. ML PREDICTION
        data["ml_prediction"] = await self._get_ml_prediction(pair, indicators)

        # 7. RL RECOMMENDATION
        data["rl_recommendation"] = self._get_rl_recommendation(pair, indicators)

        # 8. PORTFOLIO STATUS
        data["portfolio"] = self._get_portfolio_status()

        # 9. NEWS (if requested)
        if include_news:
            data["news"] = await self._get_news_summary(pair)

        # 10. GENERATE SUMMARY
        data["summary"] = self._generate_summary(data)

        return data

    def _format_technical_indicators(self, indicators: Dict) -> Dict:
        """Format technical indicators for GPT"""
        return {
            "price": {
                "current": indicators.get('current_price', 0),
                "ema_9": indicators.get('ema_9', 0),
                "ema_21": indicators.get('ema_21', 0),
                "ema_50": indicators.get('ema_50', 0),
                "bb_upper": indicators.get('bb_upper', 0),
                "bb_lower": indicators.get('bb_lower', 0),
                "bb_middle": indicators.get('bb_middle', 0)
            },
            "momentum": {
                "rsi": indicators.get('rsi', 50),
                "rsi_interpretation": self._interpret_rsi(indicators.get('rsi', 50)),
                "macd": indicators.get('macd', 0),
                "macd_signal": indicators.get('macd_signal', 0),
                "macd_histogram": indicators.get('macd_histogram', 0),
                "macd_crossover": self._detect_macd_crossover(indicators)
            },
            "volatility": {
                "atr": indicators.get('atr', 0),
                "atr_pct": indicators.get('atr', 0) / max(indicators.get('current_price', 1), 1) * 100,
                "bb_width": self._calculate_bb_width(indicators)
            },
            "trend": {
                "adx": indicators.get('adx', 0),
                "trend_strength": self._interpret_adx(indicators.get('adx', 0)),
                "ema_alignment": self._check_ema_alignment(indicators)
            },
            "volume": {
                "volume_ratio": indicators.get('volume_ratio', 1),
                "volume_interpretation": self._interpret_volume(indicators.get('volume_ratio', 1))
            }
        }

    def _interpret_rsi(self, rsi: float) -> str:
        """Interpret RSI value"""
        if rsi < 20:
            return "EXTREMELY_OVERSOLD"
        elif rsi < 30:
            return "OVERSOLD"
        elif rsi < 45:
            return "SLIGHTLY_OVERSOLD"
        elif rsi <= 55:
            return "NEUTRAL"
        elif rsi <= 70:
            return "SLIGHTLY_OVERBOUGHT"
        elif rsi <= 80:
            return "OVERBOUGHT"
        else:
            return "EXTREMELY_OVERBOUGHT"

    def _interpret_adx(self, adx: float) -> str:
        """Interpret ADX for trend strength"""
        if adx < 20:
            return "WEAK_TREND"
        elif adx < 40:
            return "MODERATE_TREND"
        elif adx < 60:
            return "STRONG_TREND"
        else:
            return "VERY_STRONG_TREND"

    def _interpret_volume(self, volume_ratio: float) -> str:
        """Interpret volume ratio"""
        if volume_ratio < 0.5:
            return "VERY_LOW"
        elif volume_ratio < 0.8:
            return "LOW"
        elif volume_ratio <= 1.2:
            return "NORMAL"
        elif volume_ratio <= 2.0:
            return "HIGH"
        else:
            return "VERY_HIGH"

    def _detect_macd_crossover(self, indicators: Dict) -> str:
        """Detect MACD crossover"""
        macd = indicators.get('macd', 0)
        signal = indicators.get('macd_signal', 0)
        hist = indicators.get('macd_histogram', 0)

        if macd > signal and hist > 0:
            if abs(hist) < 0.01 * abs(macd):
                return "FRESH_BULLISH_CROSSOVER"
            return "BULLISH"
        elif macd < signal and hist < 0:
            if abs(hist) < 0.01 * abs(macd):
                return "FRESH_BEARISH_CROSSOVER"
            return "BEARISH"
        else:
            return "NEUTRAL"

    def _calculate_bb_width(self, indicators: Dict) -> float:
        """Calculate Bollinger Band width percentage"""
        upper = indicators.get('bb_upper', 0)
        lower = indicators.get('bb_lower', 0)
        middle = indicators.get('bb_middle', 1)

        if middle > 0:
            return ((upper - lower) / middle) * 100
        return 0

    def _check_ema_alignment(self, indicators: Dict) -> str:
        """Check EMA alignment for trend"""
        ema_9 = indicators.get('ema_9', 0)
        ema_21 = indicators.get('ema_21', 0)
        ema_50 = indicators.get('ema_50', 0)

        if ema_9 > ema_21 > ema_50:
            return "BULLISH_ALIGNED"
        elif ema_9 < ema_21 < ema_50:
            return "BEARISH_ALIGNED"
        elif ema_9 > ema_21:
            return "SHORT_TERM_BULLISH"
        elif ema_9 < ema_21:
            return "SHORT_TERM_BEARISH"
        else:
            return "MIXED"

    async def _get_sentiment_data(self, pair: str, include_news: bool) -> Dict:
        """Get sentiment analysis data"""
        sentiment_data = {
            "fear_greed_index": 50,
            "fear_greed_label": "Neutral",
            "news_sentiment_overall": 0.5,
            "high_impact_news_count": 0,
            "social_buzz": 0,
            "sentiment_interpretation": "NEUTRAL"
        }

        if self.sentiment_analyzer:
            try:
                # Get Fear & Greed
                fg_data = await self._safe_call(
                    self.sentiment_analyzer.get_fear_greed_index
                )
                if fg_data:
                    sentiment_data["fear_greed_index"] = fg_data.get('value', 50)
                    sentiment_data["fear_greed_label"] = fg_data.get('value_classification', 'Neutral')

                # Get pair-specific sentiment
                pair_sentiment = await self._safe_call(
                    self.sentiment_analyzer.analyze_pair_sentiment,
                    pair
                )
                if pair_sentiment:
                    sentiment_data["news_sentiment_overall"] = pair_sentiment.get('score', 0.5)
                    sentiment_data["high_impact_news_count"] = pair_sentiment.get('high_impact', 0)

                # Interpret sentiment
                fg = sentiment_data["fear_greed_index"]
                if fg < 25:
                    sentiment_data["sentiment_interpretation"] = "EXTREME_FEAR"
                elif fg < 40:
                    sentiment_data["sentiment_interpretation"] = "FEAR"
                elif fg <= 60:
                    sentiment_data["sentiment_interpretation"] = "NEUTRAL"
                elif fg <= 75:
                    sentiment_data["sentiment_interpretation"] = "GREED"
                else:
                    sentiment_data["sentiment_interpretation"] = "EXTREME_GREED"

            except Exception as e:
                logger.debug(f"Sentiment data error: {e}")

        return sentiment_data

    async def _get_orderbook_data(self, pair: str) -> Dict:
        """Get order book analysis data"""
        orderbook_data = {
            "market_pressure": "NEUTRAL",
            "imbalance": 0.0,
            "bid_strength": 50,
            "ask_strength": 50,
            "spread_pct": 0.1,
            "depth_score": 50,
            "interpretation": "Neutral order book - no clear dominance"
        }

        if self.orderbook_analyzer:
            try:
                analysis = await self._safe_call(
                    self.orderbook_analyzer.analyze,
                    pair
                )
                if analysis:
                    orderbook_data.update({
                        "market_pressure": analysis.get('pressure', 'NEUTRAL'),
                        "imbalance": analysis.get('imbalance', 0),
                        "bid_strength": analysis.get('bid_strength', 50),
                        "ask_strength": analysis.get('ask_strength', 50),
                        "spread_pct": analysis.get('spread_pct', 0.1),
                        "depth_score": analysis.get('depth_score', 50)
                    })

                    # Generate interpretation
                    imb = orderbook_data["imbalance"]
                    if imb > 0.3:
                        orderbook_data["interpretation"] = "Strong buying pressure in order book"
                    elif imb > 0.1:
                        orderbook_data["interpretation"] = "Moderate buying pressure"
                    elif imb < -0.3:
                        orderbook_data["interpretation"] = "Strong selling pressure in order book"
                    elif imb < -0.1:
                        orderbook_data["interpretation"] = "Moderate selling pressure"
                    else:
                        orderbook_data["interpretation"] = "Balanced order book"

            except Exception as e:
                logger.debug(f"Order book data error: {e}")

        return orderbook_data

    def _get_arsenal_data(
        self,
        pair: str,
        current_price: float,
        ohlc_data: Optional[Any]
    ) -> Dict:
        """Get data from all arsenal advanced modules"""
        arsenal_data = {
            "correlation": {},
            "liquidation": {},
            "funding_rate": {},
            "volume_profile": {},
            "patterns": {},
            "session": {},
            "order_flow": {},
            "modules_status": {}
        }

        if not self.feature_aggregator:
            arsenal_data["modules_status"] = {"error": "Feature aggregator not available"}
            return arsenal_data

        try:
            fa = self.feature_aggregator

            # Module status
            arsenal_data["modules_status"] = fa.is_everything_enabled()

            # 1. Correlation Analysis
            if hasattr(fa, 'correlation_matrix'):
                arsenal_data["correlation"] = {
                    "diversification_score": fa.correlation_matrix.get_diversification_score(
                        self._get_open_positions()
                    ),
                    "correlated_pairs": self._get_correlated_pairs(pair)
                }

            # 2. Liquidation Analysis
            if hasattr(fa, 'liquidation_heatmap'):
                is_near, liq_details = fa.liquidation_heatmap.is_near_liquidation_zone(
                    pair, current_price
                )
                bias, confidence = fa.liquidation_heatmap.get_liquidation_bias(
                    pair, current_price
                )
                arsenal_data["liquidation"] = {
                    "is_near_zone": is_near,
                    "zone_details": liq_details,
                    "bias": bias,
                    "confidence": confidence,
                    "interpretation": self._interpret_liquidation(is_near, bias, confidence)
                }

            # 3. Funding Rate Analysis
            if hasattr(fa, 'funding_rate_analyzer'):
                funding_rate = fa.funding_rate_analyzer.fetch_funding_rate(pair)
                sentiment, strength, signal = fa.funding_rate_analyzer.get_funding_sentiment(pair)
                arsenal_data["funding_rate"] = {
                    "current_rate": funding_rate,
                    "sentiment": sentiment,
                    "strength": strength,
                    "signal": signal,
                    "is_extreme": abs(funding_rate or 0) > 0.1,
                    "interpretation": self._interpret_funding(funding_rate, sentiment)
                }

            # 4. Volume Profile (if OHLC data available)
            if hasattr(fa, 'volume_profile') and ohlc_data is not None:
                try:
                    is_near_poc, poc_distance = fa.volume_profile.is_near_poc(pair, current_price)
                    in_value_area = fa.volume_profile.is_in_value_area(pair, current_price)
                    arsenal_data["volume_profile"] = {
                        "is_near_poc": is_near_poc,
                        "poc_distance_pct": poc_distance,
                        "in_value_area": in_value_area,
                        "interpretation": self._interpret_volume_profile(
                            is_near_poc, in_value_area, poc_distance
                        )
                    }
                except Exception as e:
                    logger.debug(f"Volume profile error: {e}")

            # 5. Pattern Recognition (if OHLC data available)
            if hasattr(fa, 'pattern_recognition') and ohlc_data is not None:
                try:
                    patterns = fa.pattern_recognition.detect_all_patterns(ohlc_data)
                    arsenal_data["patterns"] = {
                        "detected_count": len(patterns),
                        "patterns": [
                            {
                                "name": p.get('pattern', 'Unknown'),
                                "type": p.get('type', 'Unknown'),
                                "confidence": p.get('confidence', 0),
                                "signal": p.get('signal', 'NEUTRAL')
                            }
                            for p in patterns[:5]  # Top 5 patterns
                        ],
                        "has_bullish_pattern": any(p.get('type') == 'bullish' for p in patterns),
                        "has_bearish_pattern": any(p.get('type') == 'bearish' for p in patterns)
                    }
                except Exception as e:
                    logger.debug(f"Pattern recognition error: {e}")

            # 6. Session-Based Trading
            if hasattr(fa, 'session_trading'):
                session, multiplier = fa.session_trading.get_current_session()
                arsenal_data["session"] = {
                    "current_session": session,
                    "position_size_multiplier": multiplier,
                    "interpretation": self._interpret_session(session, multiplier)
                }

            # 7. Order Flow (needs real orderbook data, basic info here)
            if hasattr(fa, 'order_flow'):
                arsenal_data["order_flow"] = {
                    "available": True,
                    "note": "Order flow data integrated via orderbook analysis"
                }

        except Exception as e:
            logger.error(f"Arsenal data error: {e}")
            arsenal_data["error"] = str(e)

        return arsenal_data

    def _get_regime_data(self, pair: str, indicators: Dict) -> Dict:
        """Determine market regime"""
        regime_data = {
            "regime": "SIDEWAYS",
            "regime_strength": "MEDIUM",
            "volatility": "NORMAL",
            "confidence": 50
        }

        # Calculate regime from indicators
        adx = indicators.get('adx', 25)
        ema_9 = indicators.get('ema_9', 0)
        ema_21 = indicators.get('ema_21', 0)
        ema_50 = indicators.get('ema_50', 0)
        atr_pct = (indicators.get('atr', 0) / max(indicators.get('current_price', 1), 1)) * 100

        # Determine trend
        if ema_9 > ema_21 > ema_50 and adx > 25:
            regime_data["regime"] = "BULLISH_TRENDING"
        elif ema_9 < ema_21 < ema_50 and adx > 25:
            regime_data["regime"] = "BEARISH_TRENDING"
        elif adx < 20:
            regime_data["regime"] = "RANGING"
        else:
            regime_data["regime"] = "SIDEWAYS"

        # Determine strength
        if adx > 40:
            regime_data["regime_strength"] = "STRONG"
        elif adx > 25:
            regime_data["regime_strength"] = "MEDIUM"
        else:
            regime_data["regime_strength"] = "WEAK"

        # Determine volatility
        if atr_pct > 3:
            regime_data["volatility"] = "HIGH"
        elif atr_pct > 1.5:
            regime_data["volatility"] = "NORMAL"
        else:
            regime_data["volatility"] = "LOW"

        regime_data["confidence"] = min(90, adx + 30)

        return regime_data

    async def _get_ml_prediction(self, pair: str, indicators: Dict) -> Dict:
        """Get ML system prediction (advisory only)"""
        ml_data = {
            "available": False,
            "win_probability": 0.5,
            "recommendation": "HOLD",
            "confidence": 0,
            "note": "ML provides advisory input only - GPT makes final decision"
        }

        if self.ml_system:
            try:
                prediction = await self._safe_call(
                    self.ml_system.predict,
                    pair,
                    indicators
                )
                if prediction:
                    ml_data.update({
                        "available": True,
                        "win_probability": prediction.get('win_probability', 0.5),
                        "recommendation": prediction.get('recommendation', 'HOLD'),
                        "confidence": prediction.get('confidence', 0)
                    })
            except Exception as e:
                logger.debug(f"ML prediction error: {e}")

        return ml_data

    def _get_rl_recommendation(self, pair: str, indicators: Dict) -> Dict:
        """Get RL agent recommendation (advisory only)"""
        rl_data = {
            "available": False,
            "action": "HOLD",
            "q_value": 0,
            "is_exploration": False,
            "note": "RL provides advisory input only - GPT makes final decision"
        }

        if self.rl_agent:
            try:
                # Get RL state and recommendation
                state = self.rl_agent.get_state_key(indicators)
                action = self.rl_agent.get_action(state)
                q_value = self.rl_agent.get_q_value(state, action)

                rl_data.update({
                    "available": True,
                    "action": action,
                    "q_value": q_value,
                    "is_exploration": self.rl_agent.is_exploring()
                })
            except Exception as e:
                logger.debug(f"RL recommendation error: {e}")

        return rl_data

    def _get_portfolio_status(self) -> Dict:
        """Get current portfolio status"""
        portfolio_data = {
            "equity": 50000,
            "available_margin": 50000,
            "open_positions_count": 0,
            "open_positions": [],
            "daily_pnl": 0,
            "win_rate": 0,
            "current_drawdown": 0,
            "risk_level": "LOW"
        }

        if self.portfolio:
            try:
                status = self.portfolio.get_status()
                if status:
                    portfolio_data.update({
                        "equity": status.get('equity', 50000),
                        "available_margin": status.get('available_margin', 50000),
                        "open_positions_count": len(status.get('positions', [])),
                        "open_positions": [p.get('pair') for p in status.get('positions', [])],
                        "daily_pnl": status.get('daily_pnl', 0),
                        "win_rate": status.get('win_rate', 0),
                        "current_drawdown": status.get('drawdown', 0)
                    })

                    # Determine risk level
                    dd = portfolio_data["current_drawdown"]
                    if dd > 8:
                        portfolio_data["risk_level"] = "HIGH"
                    elif dd > 4:
                        portfolio_data["risk_level"] = "MEDIUM"
                    else:
                        portfolio_data["risk_level"] = "LOW"

            except Exception as e:
                logger.debug(f"Portfolio status error: {e}")

        return portfolio_data

    async def _get_news_summary(self, pair: str) -> Dict:
        """Get relevant news summary"""
        news_data = {
            "available": False,
            "recent_news_count": 0,
            "bullish_news": 0,
            "bearish_news": 0,
            "high_impact_news": [],
            "summary": "No news data available"
        }

        if self.news_collector:
            try:
                # Extract currency from pair (e.g., BTC from BTC/USDT)
                currency = pair.split('/')[0] if '/' in pair else pair.replace('USDT', '')

                news = await self._safe_call(
                    self.news_collector.get_cryptopanic_news,
                    currencies=[currency],
                    limit=20
                )

                if news:
                    news_data["available"] = True
                    news_data["recent_news_count"] = len(news)

                    # Count sentiment
                    bullish = sum(1 for n in news if n.get('sentiment') == 'positive')
                    bearish = sum(1 for n in news if n.get('sentiment') == 'negative')

                    news_data["bullish_news"] = bullish
                    news_data["bearish_news"] = bearish

                    # Get high impact news
                    high_impact = [
                        {
                            "title": n.get('title', '')[:100],
                            "sentiment": n.get('sentiment', 'neutral'),
                            "source": n.get('source', {}).get('title', 'Unknown')
                        }
                        for n in news
                        if n.get('votes', {}).get('positive', 0) > 5
                    ][:3]  # Top 3

                    news_data["high_impact_news"] = high_impact

                    # Generate summary
                    if bullish > bearish * 2:
                        news_data["summary"] = f"Strongly bullish news flow ({bullish} positive vs {bearish} negative)"
                    elif bearish > bullish * 2:
                        news_data["summary"] = f"Strongly bearish news flow ({bearish} negative vs {bullish} positive)"
                    elif bullish > bearish:
                        news_data["summary"] = f"Slightly bullish news flow ({bullish}+ vs {bearish}-)"
                    elif bearish > bullish:
                        news_data["summary"] = f"Slightly bearish news flow ({bearish}- vs {bullish}+)"
                    else:
                        news_data["summary"] = "Mixed or neutral news flow"

            except Exception as e:
                logger.debug(f"News data error: {e}")

        return news_data

    def _generate_summary(self, data: Dict) -> Dict:
        """Generate overall market summary from all data"""
        bullish_factors = []
        bearish_factors = []
        neutral_factors = []
        warnings = []

        # Technical
        tech = data.get("technical", {})
        if tech.get("momentum", {}).get("rsi_interpretation", "").startswith("OVERSOLD"):
            bullish_factors.append("RSI oversold")
        elif tech.get("momentum", {}).get("rsi_interpretation", "").startswith("OVERBOUGHT"):
            bearish_factors.append("RSI overbought")

        if tech.get("trend", {}).get("ema_alignment") == "BULLISH_ALIGNED":
            bullish_factors.append("EMAs bullish aligned")
        elif tech.get("trend", {}).get("ema_alignment") == "BEARISH_ALIGNED":
            bearish_factors.append("EMAs bearish aligned")

        # Sentiment
        sent = data.get("sentiment", {})
        if sent.get("sentiment_interpretation") == "EXTREME_FEAR":
            bullish_factors.append("Extreme fear (contrarian bullish)")
        elif sent.get("sentiment_interpretation") == "EXTREME_GREED":
            bearish_factors.append("Extreme greed (contrarian bearish)")

        # Orderbook
        ob = data.get("orderbook", {})
        if ob.get("imbalance", 0) > 0.2:
            bullish_factors.append("Order book buying pressure")
        elif ob.get("imbalance", 0) < -0.2:
            bearish_factors.append("Order book selling pressure")

        # Arsenal - Funding
        funding = data.get("arsenal", {}).get("funding_rate", {})
        if funding.get("sentiment") == "bullish" and funding.get("strength") == "strong":
            bullish_factors.append("Funding rate contrarian bullish")
        elif funding.get("sentiment") == "bearish" and funding.get("strength") == "strong":
            bearish_factors.append("Funding rate contrarian bearish")

        # Patterns
        patterns = data.get("arsenal", {}).get("patterns", {})
        if patterns.get("has_bullish_pattern"):
            bullish_factors.append("Bullish chart pattern detected")
        if patterns.get("has_bearish_pattern"):
            bearish_factors.append("Bearish chart pattern detected")

        # Portfolio warnings
        portfolio = data.get("portfolio", {})
        if portfolio.get("risk_level") == "HIGH":
            warnings.append("High portfolio risk - reduce position size")
        if portfolio.get("open_positions_count", 0) >= 4:
            warnings.append("Many open positions - be selective")

        # Generate bias
        bullish_score = len(bullish_factors)
        bearish_score = len(bearish_factors)

        if bullish_score > bearish_score + 2:
            overall_bias = "STRONGLY_BULLISH"
        elif bullish_score > bearish_score:
            overall_bias = "BULLISH"
        elif bearish_score > bullish_score + 2:
            overall_bias = "STRONGLY_BEARISH"
        elif bearish_score > bullish_score:
            overall_bias = "BEARISH"
        else:
            overall_bias = "NEUTRAL"

        return {
            "overall_bias": overall_bias,
            "bullish_factors": bullish_factors,
            "bearish_factors": bearish_factors,
            "neutral_factors": neutral_factors,
            "warnings": warnings,
            "confidence": min(90, (bullish_score + bearish_score) * 15),
            "trade_recommended": abs(bullish_score - bearish_score) >= 2
        }

    # Helper methods for interpretations
    def _interpret_liquidation(self, is_near: bool, bias: str, confidence: float) -> str:
        if is_near:
            return f"Near liquidation zone - potential volatility. Bias: {bias}"
        return f"No immediate liquidation pressure. Market bias: {bias}"

    def _interpret_funding(self, rate: Optional[float], sentiment: str) -> str:
        if rate is None:
            return "Funding rate unavailable"
        if abs(rate) > 0.1:
            direction = "longs paying shorts" if rate > 0 else "shorts paying longs"
            return f"Extreme funding ({rate:.4f}%) - {direction}. Contrarian signal: {sentiment}"
        return f"Normal funding rate ({rate:.4f}%). Market sentiment: {sentiment}"

    def _interpret_volume_profile(
        self,
        is_near_poc: bool,
        in_value_area: bool,
        poc_distance: float
    ) -> str:
        if is_near_poc:
            return "Price near Point of Control (POC) - high volume zone, expect consolidation or reversal"
        elif in_value_area:
            return "Price within Value Area - normal trading zone"
        else:
            return f"Price outside Value Area ({poc_distance:.1f}% from POC) - potential return to POC"

    def _interpret_session(self, session: str, multiplier: float) -> str:
        sessions = {
            "ASIA": "Asian session - typically lower volatility",
            "EUROPE": "European session - increasing volume",
            "US": "US session - highest volatility and volume",
            "OVERLAP": "Session overlap - high activity period"
        }
        base = sessions.get(session, "Unknown session")
        if multiplier != 1.0:
            base += f" (position size adjusted to {multiplier*100:.0f}%)"
        return base

    def _get_open_positions(self) -> List[str]:
        """Get list of open position pairs"""
        if self.portfolio:
            try:
                status = self.portfolio.get_status()
                return [p.get('pair') for p in status.get('positions', [])]
            except:
                pass
        return []

    def _get_correlated_pairs(self, pair: str) -> List[Dict]:
        """Get highly correlated pairs"""
        if self.feature_aggregator and hasattr(self.feature_aggregator, 'correlation_matrix'):
            try:
                return self.feature_aggregator.correlation_matrix.get_highly_correlated(pair)
            except:
                pass
        return []

    async def _safe_call(self, func, *args, **kwargs):
        """Safely call a function (sync or async)"""
        try:
            import asyncio
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        except Exception as e:
            logger.debug(f"Safe call error: {e}")
            return None

    def format_for_gpt_prompt(self, data: Dict) -> str:
        """
        Format all collected data into a comprehensive prompt for GPT.

        Args:
            data: Complete market data dict

        Returns:
            Formatted string for GPT prompt
        """
        pair = data.get("pair", "UNKNOWN")
        tech = data.get("technical", {})
        sent = data.get("sentiment", {})
        ob = data.get("orderbook", {})
        arsenal = data.get("arsenal", {})
        regime = data.get("regime", {})
        ml = data.get("ml_prediction", {})
        rl = data.get("rl_recommendation", {})
        portfolio = data.get("portfolio", {})
        news = data.get("news", {})
        summary = data.get("summary", {})

        prompt = f"""
=== COMPLETE MARKET DATA FOR {pair} ===
Timestamp: {data.get('timestamp', 'N/A')}

== TECHNICAL INDICATORS ==
Price: ${tech.get('price', {}).get('current', 0):,.2f}
EMAs: 9=${tech.get('price', {}).get('ema_9', 0):,.2f} | 21=${tech.get('price', {}).get('ema_21', 0):,.2f} | 50=${tech.get('price', {}).get('ema_50', 0):,.2f}
EMA Alignment: {tech.get('trend', {}).get('ema_alignment', 'N/A')}

RSI: {tech.get('momentum', {}).get('rsi', 50):.1f} ({tech.get('momentum', {}).get('rsi_interpretation', 'N/A')})
MACD: {tech.get('momentum', {}).get('macd', 0):.4f} (Signal: {tech.get('momentum', {}).get('macd_signal', 0):.4f})
MACD Status: {tech.get('momentum', {}).get('macd_crossover', 'N/A')}

ADX: {tech.get('trend', {}).get('adx', 0):.1f} ({tech.get('trend', {}).get('trend_strength', 'N/A')})
ATR%: {tech.get('volatility', {}).get('atr_pct', 0):.2f}%
Volume: {tech.get('volume', {}).get('volume_ratio', 1):.1f}x ({tech.get('volume', {}).get('volume_interpretation', 'N/A')})

== SENTIMENT ==
Fear & Greed: {sent.get('fear_greed_index', 50)} ({sent.get('fear_greed_label', 'N/A')})
Interpretation: {sent.get('sentiment_interpretation', 'N/A')}
News Sentiment: {sent.get('news_sentiment_overall', 0.5):.2f}
High Impact News: {sent.get('high_impact_news_count', 0)}

== ORDER BOOK ==
Pressure: {ob.get('market_pressure', 'NEUTRAL')}
Imbalance: {ob.get('imbalance', 0):+.2f}
Interpretation: {ob.get('interpretation', 'N/A')}

== MARKET REGIME ==
Regime: {regime.get('regime', 'SIDEWAYS')}
Strength: {regime.get('regime_strength', 'MEDIUM')}
Volatility: {regime.get('volatility', 'NORMAL')}

== ARSENAL ADVANCED ANALYSIS ==
"""

        # Funding Rate
        funding = arsenal.get('funding_rate', {})
        if funding:
            prompt += f"""
Funding Rate: {funding.get('current_rate', 'N/A')}
Funding Sentiment: {funding.get('sentiment', 'N/A')} ({funding.get('strength', 'N/A')})
Funding Signal: {funding.get('signal', 'N/A')}
"""

        # Liquidation
        liq = arsenal.get('liquidation', {})
        if liq:
            prompt += f"""
Near Liquidation Zone: {'YES' if liq.get('is_near_zone') else 'NO'}
Liquidation Bias: {liq.get('bias', 'N/A')}
"""

        # Patterns
        patterns = arsenal.get('patterns', {})
        if patterns and patterns.get('detected_count', 0) > 0:
            prompt += f"""
Patterns Detected: {patterns.get('detected_count', 0)}
"""
            for p in patterns.get('patterns', [])[:3]:
                prompt += f"  - {p.get('name')}: {p.get('signal')} ({p.get('confidence', 0):.0f}%)\n"

        # Session
        session = arsenal.get('session', {})
        if session:
            prompt += f"""
Current Session: {session.get('current_session', 'N/A')}
Position Multiplier: {session.get('position_size_multiplier', 1.0):.2f}
"""

        # ML/RL Advisory
        prompt += f"""
== ADVISORY SYSTEMS (GPT CAN OVERRIDE) ==
ML Prediction: {ml.get('recommendation', 'N/A')} (Win Prob: {ml.get('win_probability', 0.5)*100:.0f}%)
RL Recommendation: {rl.get('action', 'N/A')} (Q-Value: {rl.get('q_value', 0):.3f})
Note: These are advisory only - GPT makes final decisions

== PORTFOLIO STATUS ==
Equity: ${portfolio.get('equity', 0):,.2f}
Open Positions: {portfolio.get('open_positions_count', 0)}
Current Drawdown: {portfolio.get('current_drawdown', 0):.2f}%
Risk Level: {portfolio.get('risk_level', 'LOW')}
"""

        # News
        if news.get('available'):
            prompt += f"""
== NEWS ==
Recent News: {news.get('recent_news_count', 0)} articles
Sentiment: {news.get('bullish_news', 0)} bullish / {news.get('bearish_news', 0)} bearish
Summary: {news.get('summary', 'N/A')}
"""

        # Summary
        prompt += f"""
== OVERALL SUMMARY ==
Bias: {summary.get('overall_bias', 'NEUTRAL')}
Confidence: {summary.get('confidence', 50)}%
Trade Recommended: {'YES' if summary.get('trade_recommended') else 'NO'}

Bullish Factors: {', '.join(summary.get('bullish_factors', [])) or 'None'}
Bearish Factors: {', '.join(summary.get('bearish_factors', [])) or 'None'}
Warnings: {', '.join(summary.get('warnings', [])) or 'None'}
"""

        return prompt

    def get_stats(self) -> Dict:
        """Get provider statistics"""
        return {
            "cache_entries": len(self.cache),
            "sources_available": {
                "feature_aggregator": self.feature_aggregator is not None,
                "sentiment_analyzer": self.sentiment_analyzer is not None,
                "orderbook_analyzer": self.orderbook_analyzer is not None,
                "news_collector": self.news_collector is not None,
                "ml_system": self.ml_system is not None,
                "rl_agent": self.rl_agent is not None,
                "portfolio": self.portfolio is not None
            }
        }
