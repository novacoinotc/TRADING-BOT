"""
Advanced Technical Analysis Module
Professional-grade analysis with volume, divergences, multi-timeframe, and more
"""
import pandas as pd
import numpy as np
from ta.trend import MACD, EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator
from config import config
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class AdvancedTechnicalAnalyzer:
    """
    Advanced technical analyzer with professional-grade indicators
    """

    def __init__(self):
        self.rsi_oversold = config.RSI_OVERSOLD
        self.rsi_overbought = config.RSI_OVERBOUGHT

    def analyze_multi_timeframe(self, dfs: Dict[str, pd.DataFrame]) -> dict:
        """
        Perform multi-timeframe analysis

        Args:
            dfs: Dictionary of DataFrames {timeframe: df}
                 e.g., {'1h': df_1h, '4h': df_4h, '1d': df_1d}

        Returns:
            Complete analysis with signals, indicators, and recommendations
        """
        if not dfs or '1h' not in dfs:
            return None

        df_1h = dfs['1h']

        if df_1h is None or len(df_1h) < 50:
            return None

        # Calculate all indicators for primary timeframe (1h)
        indicators = self._calculate_all_indicators(df_1h)

        # Analyze volume
        volume_analysis = self._analyze_volume(df_1h)
        indicators.update(volume_analysis)

        # Detect divergences
        divergences = self._detect_divergences(df_1h)
        indicators['divergences'] = divergences

        # Find support/resistance levels
        levels = self._find_support_resistance(df_1h)
        indicators['support_resistance'] = levels

        # Multi-timeframe trend analysis
        mtf_trends = self._analyze_multi_timeframe_trends(dfs)
        indicators['mtf_trends'] = mtf_trends

        # Calculate ATR for stop-loss/take-profit
        atr = self._calculate_atr(df_1h)
        indicators['atr'] = atr

        # Generate advanced signals with scoring
        signals = self._generate_advanced_signals(indicators, df_1h)

        # Calculate stop-loss and take-profit levels
        if signals['action'] != 'HOLD':
            sl_tp = self._calculate_sl_tp(
                indicators['current_price'],
                signals['action'],
                atr,
                levels
            )
            signals['stop_loss'] = sl_tp['stop_loss']
            signals['take_profit'] = sl_tp['take_profit']
            signals['risk_reward'] = sl_tp['risk_reward']

        return {
            'indicators': indicators,
            'signals': signals
        }

    def _calculate_all_indicators(self, df: pd.DataFrame) -> dict:
        """Calculate all technical indicators"""
        indicators = {}
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        # RSI
        rsi_indicator = RSIIndicator(close=close, window=config.RSI_PERIOD)
        indicators['rsi'] = round(rsi_indicator.rsi().iloc[-1], 2)
        indicators['rsi_series'] = rsi_indicator.rsi()

        # MACD
        macd_indicator = MACD(
            close=close,
            window_slow=config.MACD_SLOW,
            window_fast=config.MACD_FAST,
            window_sign=config.MACD_SIGNAL
        )
        indicators['macd'] = round(macd_indicator.macd().iloc[-1], 4)
        indicators['macd_signal'] = round(macd_indicator.macd_signal().iloc[-1], 4)
        indicators['macd_diff'] = round(macd_indicator.macd_diff().iloc[-1], 4)
        indicators['macd_series'] = macd_indicator.macd()
        indicators['macd_histogram'] = macd_indicator.macd_diff()

        # EMAs
        ema_short = EMAIndicator(close=close, window=config.EMA_SHORT).ema_indicator()
        ema_medium = EMAIndicator(close=close, window=config.EMA_MEDIUM).ema_indicator()
        ema_long = EMAIndicator(close=close, window=config.EMA_LONG).ema_indicator()

        indicators['ema_short'] = round(ema_short.iloc[-1], 2)
        indicators['ema_medium'] = round(ema_medium.iloc[-1], 2)
        indicators['ema_long'] = round(ema_long.iloc[-1], 2)

        # Bollinger Bands
        bb_indicator = BollingerBands(close=close, window=config.BB_PERIOD, window_dev=config.BB_STD)
        indicators['bb_upper'] = round(bb_indicator.bollinger_hband().iloc[-1], 2)
        indicators['bb_middle'] = round(bb_indicator.bollinger_mavg().iloc[-1], 2)
        indicators['bb_lower'] = round(bb_indicator.bollinger_lband().iloc[-1], 2)

        # Safe BB width calculation (avoid division by zero)
        if indicators['bb_middle'] > 0 and (indicators['bb_upper'] - indicators['bb_lower']) > 0:
            indicators['bb_width'] = round(
                ((indicators['bb_upper'] - indicators['bb_lower']) / indicators['bb_middle']) * 100, 2
            )
        else:
            indicators['bb_width'] = 0.0

        # ADX (Trend Strength)
        adx_indicator = ADXIndicator(high=high, low=low, close=close, window=14)
        indicators['adx'] = round(adx_indicator.adx().iloc[-1], 2)

        # Current price
        indicators['current_price'] = round(close.iloc[-1], 2)
        indicators['high_24h'] = round(high.iloc[-24:].max(), 2) if len(high) >= 24 else round(high.max(), 2)
        indicators['low_24h'] = round(low.iloc[-24:].min(), 2) if len(low) >= 24 else round(low.min(), 2)

        return indicators

    def _analyze_volume(self, df: pd.DataFrame) -> dict:
        """Analyze volume patterns"""
        volume_data = {}
        volume = df['volume']
        close = df['close']

        # Volume average
        avg_volume_20 = volume.iloc[-20:].mean()
        current_volume = volume.iloc[-1]
        volume_data['avg_volume'] = round(avg_volume_20, 2)
        volume_data['current_volume'] = round(current_volume, 2)
        volume_data['volume_ratio'] = round((current_volume / avg_volume_20), 2)

        # Volume trend (increasing/decreasing)
        recent_volume = volume.iloc[-5:].mean()
        prev_volume = volume.iloc[-10:-5].mean()
        volume_data['volume_trend'] = 'increasing' if recent_volume > prev_volume else 'decreasing'

        # On-Balance Volume (OBV)
        obv_indicator = OnBalanceVolumeIndicator(close=close, volume=volume)
        obv = obv_indicator.on_balance_volume()
        volume_data['obv_trend'] = 'bullish' if obv.iloc[-1] > obv.iloc[-5] else 'bearish'

        return volume_data

    def _detect_divergences(self, df: pd.DataFrame) -> dict:
        """Detect bullish/bearish divergences"""
        divergences = {
            'rsi_divergence': None,
            'macd_divergence': None
        }

        close = df['close'].iloc[-20:]

        # RSI divergence
        rsi_indicator = RSIIndicator(close=df['close'], window=config.RSI_PERIOD)
        rsi = rsi_indicator.rsi().iloc[-20:]

        if len(close) >= 10:
            # Bullish divergence: price makes lower low, RSI makes higher low
            price_trend = close.iloc[-1] < close.iloc[-10]
            rsi_trend = rsi.iloc[-1] > rsi.iloc[-10]

            if price_trend and rsi_trend:
                divergences['rsi_divergence'] = 'bullish'
            elif not price_trend and not rsi_trend:
                divergences['rsi_divergence'] = 'bearish'

        return divergences

    def _find_support_resistance(self, df: pd.DataFrame) -> dict:
        """Find key support and resistance levels"""
        close = df['close'].iloc[-100:] if len(df) >= 100 else df['close']
        high = df['high'].iloc[-100:] if len(df) >= 100 else df['high']
        low = df['low'].iloc[-100:] if len(df) >= 100 else df['low']

        current_price = close.iloc[-1]

        # Simple support/resistance based on recent highs/lows
        resistance_levels = []
        support_levels = []

        # Find local peaks and troughs
        for i in range(5, len(close) - 5):
            # Resistance (local high)
            if high.iloc[i] == high.iloc[i-5:i+5].max():
                resistance_levels.append(high.iloc[i])
            # Support (local low)
            if low.iloc[i] == low.iloc[i-5:i+5].min():
                support_levels.append(low.iloc[i])

        # Get nearest levels
        resistance_levels = sorted([r for r in resistance_levels if r > current_price])
        support_levels = sorted([s for s in support_levels if s < current_price], reverse=True)

        nearest_resistance = resistance_levels[0] if resistance_levels else current_price * 1.05
        nearest_support = support_levels[0] if support_levels else current_price * 0.95

        return {
            'nearest_support': round(nearest_support, 2),
            'nearest_resistance': round(nearest_resistance, 2),
            'distance_to_support': round(((current_price - nearest_support) / current_price) * 100, 2),
            'distance_to_resistance': round(((nearest_resistance - current_price) / current_price) * 100, 2)
        }

    def _analyze_multi_timeframe_trends(self, dfs: Dict[str, pd.DataFrame]) -> dict:
        """Analyze trends across multiple timeframes"""
        trends = {}

        for tf, df in dfs.items():
            if df is None or len(df) < 50:
                trends[tf] = 'unknown'
                continue

            close = df['close']
            ema_20 = EMAIndicator(close=close, window=20).ema_indicator()
            ema_50 = EMAIndicator(close=close, window=50).ema_indicator() if len(close) >= 50 else ema_20

            current_price = close.iloc[-1]

            if current_price > ema_20.iloc[-1] > ema_50.iloc[-1]:
                trends[tf] = 'bullish'
            elif current_price < ema_20.iloc[-1] < ema_50.iloc[-1]:
                trends[tf] = 'bearish'
            else:
                trends[tf] = 'neutral'

        # Determine overall trend alignment
        bullish_count = sum(1 for t in trends.values() if t == 'bullish')
        bearish_count = sum(1 for t in trends.values() if t == 'bearish')

        if bullish_count >= 2:
            trends['alignment'] = 'bullish'
        elif bearish_count >= 2:
            trends['alignment'] = 'bearish'
        else:
            trends['alignment'] = 'neutral'

        return trends

    def _calculate_atr(self, df: pd.DataFrame) -> float:
        """Calculate Average True Range for volatility"""
        atr_indicator = AverageTrueRange(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=14
        )
        return round(atr_indicator.average_true_range().iloc[-1], 2)

    def _generate_advanced_signals(self, indicators: dict, df: pd.DataFrame) -> dict:
        """
        Generate advanced trading signals with 0-10 scoring system
        """
        score = 0.0
        max_score = 10.0
        reasons = []
        risk_level = 'LOW'
        confidence = 0

        current_price = indicators['current_price']

        # 1. RSI Analysis (0-2 points)
        if indicators['rsi'] < 25:
            score += 2.0
            reasons.append(f"🔥 RSI extremely oversold ({indicators['rsi']:.1f})")
        elif indicators['rsi'] < self.rsi_oversold:
            score += 1.5
            reasons.append(f"RSI oversold ({indicators['rsi']:.1f})")
        elif indicators['rsi'] > 75:
            score -= 2.0
            reasons.append(f"🔥 RSI extremely overbought ({indicators['rsi']:.1f})")
        elif indicators['rsi'] > self.rsi_overbought:
            score -= 1.5
            reasons.append(f"RSI overbought ({indicators['rsi']:.1f})")

        # 2. Volume Confirmation (0-2 points) - OPTIMIZADO
        if indicators['volume_ratio'] > 1.5:
            score += 2.0 if score > 0 else -2.0
            reasons.append(f"📊 High volume confirmation (+{(indicators['volume_ratio']-1)*100:.0f}%)")
        elif indicators['volume_ratio'] > 1.2:
            score += 1.0 if score > 0 else -1.0
            reasons.append(f"Volume above average (+{(indicators['volume_ratio']-1)*100:.0f}%)")
        elif indicators['volume_ratio'] > 0.9:
            # Volumen normal - sin penalización
            pass
        elif indicators['volume_ratio'] < 0.7:
            reasons.append("⚠️ Low volume - weak signal")
            score *= 0.85  # Penalización reducida (antes 0.7, ahora 0.85)

        # 3. Divergence Detection (0-1.5 points)
        if indicators['divergences']['rsi_divergence'] == 'bullish':
            score += 1.5
            reasons.append("💎 Bullish RSI divergence detected")
        elif indicators['divergences']['rsi_divergence'] == 'bearish':
            score -= 1.5
            reasons.append("💎 Bearish RSI divergence detected")

        # 4. MACD (0-1.5 points)
        if indicators['macd_diff'] > 0 and indicators['macd'] > indicators['macd_signal']:
            score += 1.5
            reasons.append("MACD bullish crossover")
        elif indicators['macd_diff'] < 0 and indicators['macd'] < indicators['macd_signal']:
            score -= 1.5
            reasons.append("MACD bearish crossover")

        # 5. Multi-timeframe Alignment (0-2 points)
        mtf_trends = indicators.get('mtf_trends', {})
        if mtf_trends.get('alignment') == 'bullish':
            score += 2.0
            reasons.append("✅ Multi-timeframe bullish alignment")
        elif mtf_trends.get('alignment') == 'bearish':
            score -= 2.0
            reasons.append("✅ Multi-timeframe bearish alignment")

        # 6. Support/Resistance (0-1 point)
        sr_levels = indicators.get('support_resistance', {})
        if sr_levels.get('distance_to_support', 100) < 2:
            score += 1.0
            reasons.append(f"📍 Near support level (${sr_levels['nearest_support']:.2f})")
        elif sr_levels.get('distance_to_resistance', 100) < 2:
            score -= 1.0
            reasons.append(f"📍 Near resistance level (${sr_levels['nearest_resistance']:.2f})")

        # 7. ADX Trend Strength (bonus/penalty) - OPTIMIZADO
        if indicators['adx'] > 30:
            score += 0.5  # Bonus por tendencia muy fuerte
            reasons.append(f"💪 Very strong trend (ADX: {indicators['adx']:.1f})")
        elif indicators['adx'] > 25:
            reasons.append(f"💪 Strong trend (ADX: {indicators['adx']:.1f})")
        elif indicators['adx'] < 15:
            reasons.append(f"⚠️ Very weak trend (ADX: {indicators['adx']:.1f})")
            score *= 0.90  # Penalización reducida (antes 0.8, ahora 0.90)

        # 8. Bollinger Bands
        if current_price < indicators['bb_lower']:
            score += 1.0
            reasons.append("Price below lower Bollinger Band")
        elif current_price > indicators['bb_upper']:
            score -= 1.0
            reasons.append("Price above upper Bollinger Band")

        # 9. EMA Alignment Bonus (0-1.5 points) - NUEVO
        ema_short = indicators.get('ema_short', 0)
        ema_medium = indicators.get('ema_medium', 0)
        ema_long = indicators.get('ema_long', 0)

        # Alineación alcista perfecta: Price > EMA9 > EMA21 > EMA50
        if current_price > ema_short > ema_medium > ema_long and score > 0:
            score += 1.5
            reasons.append("🚀 Perfect bullish EMA alignment")
        # Alineación bajista perfecta: Price < EMA9 < EMA21 < EMA50
        elif current_price < ema_short < ema_medium < ema_long and score < 0:
            score -= 1.5
            reasons.append("🔻 Perfect bearish EMA alignment")
        # Alineación alcista parcial
        elif current_price > ema_short > ema_medium and score > 0:
            score += 0.75
            reasons.append("Partial bullish EMA alignment")
        # Alineación bajista parcial
        elif current_price < ema_short < ema_medium and score < 0:
            score -= 0.75
            reasons.append("Partial bearish EMA alignment")

        # Normalize score to 0-10 range
        final_score = max(0, min(abs(score), max_score))

        # Determine action (usa threshold dinámico del config)
        threshold = config.CONSERVATIVE_THRESHOLD
        if score >= threshold:
            action = 'BUY'
            risk_level = 'LOW' if indicators['volume_ratio'] > 1.2 else 'MEDIUM'
            confidence = int((final_score / max_score) * 100)
        elif score <= -threshold:
            action = 'SELL'
            risk_level = 'LOW' if indicators['volume_ratio'] > 1.2 else 'MEDIUM'
            confidence = int((final_score / max_score) * 100)
        else:
            action = 'HOLD'
            # Keep the real score for visibility (don't force to 0)
            confidence = 0
            reasons.append(f'Signal not strong enough (need {threshold}+ points, has {abs(score):.1f})')  # Show actual score

        return {
            'action': action,
            'score': round(final_score, 1),
            'max_score': max_score,
            'strength': min(int(final_score / 2), 5),  # Convert to 0-5 stars for compatibility
            'reasons': reasons,
            'risk_level': risk_level,
            'confidence': confidence
        }

    def _calculate_sl_tp(self, entry_price: float, action: str, atr: float, levels: dict) -> dict:
        """
        Calculate dynamic stop-loss and take-profit levels
        """
        # Use ATR for dynamic stops
        atr_multiplier_sl = 2.0  # Stop loss at 2x ATR
        atr_multiplier_tp1 = 2.5  # TP1 at 2.5x ATR
        atr_multiplier_tp2 = 4.0  # TP2 at 4x ATR
        atr_multiplier_tp3 = 6.0  # TP3 at 6x ATR

        if action == 'BUY':
            stop_loss = entry_price - (atr * atr_multiplier_sl)
            # Adjust stop loss if below support
            if levels['nearest_support'] and levels['nearest_support'] < entry_price:
                stop_loss = max(stop_loss, levels['nearest_support'] * 0.98)

            tp1 = entry_price + (atr * atr_multiplier_tp1)
            tp2 = entry_price + (atr * atr_multiplier_tp2)
            tp3 = entry_price + (atr * atr_multiplier_tp3)

            # Adjust TPs if near resistance
            if levels['nearest_resistance'] and levels['nearest_resistance'] > entry_price:
                tp1 = min(tp1, levels['nearest_resistance'] * 0.99)
        else:  # SELL
            stop_loss = entry_price + (atr * atr_multiplier_sl)
            # Adjust stop loss if above resistance
            if levels['nearest_resistance'] and levels['nearest_resistance'] > entry_price:
                stop_loss = min(stop_loss, levels['nearest_resistance'] * 1.02)

            tp1 = entry_price - (atr * atr_multiplier_tp1)
            tp2 = entry_price - (atr * atr_multiplier_tp2)
            tp3 = entry_price - (atr * atr_multiplier_tp3)

            # Adjust TPs if near support
            if levels['nearest_support'] and levels['nearest_support'] < entry_price:
                tp1 = max(tp1, levels['nearest_support'] * 1.01)

        risk = abs(entry_price - stop_loss)
        reward = abs(tp2 - entry_price)  # Use TP2 for R:R calculation
        risk_reward = round(reward / risk, 2) if risk > 0 else 0

        return {
            'stop_loss': round(stop_loss, 2),
            'take_profit': {
                'tp1': round(tp1, 2),
                'tp2': round(tp2, 2),
                'tp3': round(tp3, 2)
            },
            'risk_reward': risk_reward,
            'risk_amount': round(risk, 2),
            'reward_amount': round(reward, 2)
        }
