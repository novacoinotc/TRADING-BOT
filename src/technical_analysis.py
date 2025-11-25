"""
Technical Analysis Module
Calculates various technical indicators for trading signals
"""
import pandas as pd
import numpy as np
from ta.trend import MACD, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from config import config


class TechnicalAnalyzer:
    """
    Performs technical analysis on price data
    """

    def __init__(self):
        self.rsi_oversold = config.RSI_OVERSOLD
        self.rsi_overbought = config.RSI_OVERBOUGHT

    def analyze(self, df: pd.DataFrame) -> dict:
        """
        Perform comprehensive technical analysis

        Args:
            df: DataFrame with OHLCV data (columns: open, high, low, close, volume)

        Returns:
            Dictionary with analysis results and signals
        """
        if df is None or len(df) < 50:
            return None

        close = df['close']

        # Calculate indicators
        indicators = {}

        # RSI
        rsi_indicator = RSIIndicator(close=close, window=config.RSI_PERIOD)
        rsi = rsi_indicator.rsi().iloc[-1]
        indicators['rsi'] = round(rsi, 2)

        # MACD
        macd_indicator = MACD(
            close=close,
            window_slow=config.MACD_SLOW,
            window_fast=config.MACD_FAST,
            window_sign=config.MACD_SIGNAL
        )
        macd = macd_indicator.macd().iloc[-1]
        macd_signal = macd_indicator.macd_signal().iloc[-1]
        macd_diff = macd_indicator.macd_diff().iloc[-1]

        indicators['macd'] = round(macd, 4)
        indicators['macd_signal'] = round(macd_signal, 4)
        indicators['macd_diff'] = round(macd_diff, 4)

        # EMAs
        ema_short = EMAIndicator(close=close, window=config.EMA_SHORT).ema_indicator().iloc[-1]
        ema_medium = EMAIndicator(close=close, window=config.EMA_MEDIUM).ema_indicator().iloc[-1]
        ema_long = EMAIndicator(close=close, window=config.EMA_LONG).ema_indicator().iloc[-1]

        indicators['ema_short'] = round(ema_short, 2)
        indicators['ema_medium'] = round(ema_medium, 2)
        indicators['ema_long'] = round(ema_long, 2)

        # Bollinger Bands
        bb_indicator = BollingerBands(close=close, window=config.BB_PERIOD, window_dev=config.BB_STD)
        bb_upper = bb_indicator.bollinger_hband().iloc[-1]
        bb_middle = bb_indicator.bollinger_mavg().iloc[-1]
        bb_lower = bb_indicator.bollinger_lband().iloc[-1]

        indicators['bb_upper'] = round(bb_upper, 2)
        indicators['bb_middle'] = round(bb_middle, 2)
        indicators['bb_lower'] = round(bb_lower, 2)

        # Current price
        current_price = close.iloc[-1]
        indicators['current_price'] = round(current_price, 2)

        # Generate signals
        signals = self._generate_signals(indicators, current_price)

        return {
            'indicators': indicators,
            'signals': signals
        }

    def _generate_signals(self, indicators: dict, current_price: float) -> dict:
        """
        Generate buy/sell signals based on indicators

        Args:
            indicators: Dictionary of calculated indicators
            current_price: Current price

        Returns:
            Dictionary with signal information
        """
        signals = {
            'action': 'HOLD',
            'strength': 0,
            'reasons': []
        }

        buy_score = 0
        sell_score = 0

        # RSI Signals
        if indicators['rsi'] < self.rsi_oversold:
            buy_score += 2
            signals['reasons'].append(f"RSI oversold ({indicators['rsi']:.2f})")
        elif indicators['rsi'] > self.rsi_overbought:
            sell_score += 2
            signals['reasons'].append(f"RSI overbought ({indicators['rsi']:.2f})")

        # MACD Signals
        if indicators['macd_diff'] > 0 and indicators['macd'] > indicators['macd_signal']:
            buy_score += 1
            signals['reasons'].append("MACD bullish crossover")
        elif indicators['macd_diff'] < 0 and indicators['macd'] < indicators['macd_signal']:
            sell_score += 1
            signals['reasons'].append("MACD bearish crossover")

        # EMA Trend
        if indicators['ema_short'] > indicators['ema_medium'] > indicators['ema_long']:
            buy_score += 1
            signals['reasons'].append("Strong uptrend (EMA alignment)")
        elif indicators['ema_short'] < indicators['ema_medium'] < indicators['ema_long']:
            sell_score += 1
            signals['reasons'].append("Strong downtrend (EMA alignment)")

        # Price vs EMA
        if current_price > indicators['ema_long']:
            buy_score += 0.5
        else:
            sell_score += 0.5

        # Bollinger Bands
        if current_price < indicators['bb_lower']:
            buy_score += 1
            signals['reasons'].append("Price below lower Bollinger Band")
        elif current_price > indicators['bb_upper']:
            sell_score += 1
            signals['reasons'].append("Price above upper Bollinger Band")

        # Determine final signal
        net_score = buy_score - sell_score

        if net_score >= 2:
            signals['action'] = 'BUY'
            signals['strength'] = min(int(buy_score), 5)
        elif net_score <= -2:
            signals['action'] = 'SELL'
            signals['strength'] = min(int(sell_score), 5)
        else:
            signals['action'] = 'HOLD'
            signals['strength'] = 0
            signals['reasons'] = ['No clear signal']

        return signals
