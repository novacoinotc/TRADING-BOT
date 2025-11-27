"""
Flash Signal Analyzer - 15 minute timeframe for quick opportunities
Medium threshold: 5.0+ points, 50%+ confidence, MEDIUM risk
More selective than previous version (3.5+ points, HIGH risk)
"""
import pandas as pd
import numpy as np
from ta.trend import MACD, EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from config import config
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class FlashSignalAnalyzer:
    """
    Analyzer for quick 15-minute signals (flash signals)
    Less strict requirements but marked as RISKY
    """

    def __init__(self):
        self.rsi_oversold = config.RSI_OVERSOLD
        self.rsi_overbought = config.RSI_OVERBOUGHT
        self.flash_threshold = config.FLASH_THRESHOLD  # Dynamic threshold from config (5.0)
        self.min_confidence = config.FLASH_MIN_CONFIDENCE  # Minimum confidence (50%)

    def analyze_flash(self, df: pd.DataFrame) -> dict:
        """
        Quick analysis on 15-minute timeframe
        Medium threshold: 5.0+ points, 50%+ confidence, MEDIUM risk

        Args:
            df: DataFrame with OHLCV data for 15m timeframe

        Returns:
            Flash signal analysis (only if score >= 5.0 AND confidence >= 50%)
        """
        if df is None or len(df) < 30:
            return None

        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        indicators = {}

        # RSI
        rsi_indicator = RSIIndicator(close=close, window=config.RSI_PERIOD)
        indicators['rsi'] = round(rsi_indicator.rsi().iloc[-1], 2)

        # MACD
        macd_indicator = MACD(close=close, window_slow=config.MACD_SLOW,
                             window_fast=config.MACD_FAST, window_sign=config.MACD_SIGNAL)
        indicators['macd'] = round(macd_indicator.macd().iloc[-1], 4)
        indicators['macd_signal'] = round(macd_indicator.macd_signal().iloc[-1], 4)
        indicators['macd_diff'] = round(macd_indicator.macd_diff().iloc[-1], 4)

        # EMAs (shorter periods for 10m)
        ema_9 = EMAIndicator(close=close, window=9).ema_indicator()
        ema_21 = EMAIndicator(close=close, window=21).ema_indicator()

        indicators['ema_short'] = round(ema_9.iloc[-1], 2)
        indicators['ema_medium'] = round(ema_21.iloc[-1], 2)

        # Bollinger Bands
        bb_indicator = BollingerBands(close=close, window=20, window_dev=2)
        indicators['bb_upper'] = round(bb_indicator.bollinger_hband().iloc[-1], 2)
        indicators['bb_middle'] = round(bb_indicator.bollinger_mavg().iloc[-1], 2)
        indicators['bb_lower'] = round(bb_indicator.bollinger_lband().iloc[-1], 2)

        # Volume analysis
        avg_volume = volume.iloc[-20:].mean()
        current_volume = volume.iloc[-1]
        indicators['volume_ratio'] = round(current_volume / avg_volume, 2)

        # Price
        indicators['current_price'] = round(close.iloc[-1], 2)

        # ATR for stop-loss
        atr_indicator = AverageTrueRange(high=high, low=low, close=close, window=14)
        indicators['atr'] = round(atr_indicator.average_true_range().iloc[-1], 2)

        # Generate flash signals
        signals = self._generate_flash_signals(indicators)

        # Calculate SL/TP for flash signals
        if signals['action'] != 'HOLD':
            sl_tp = self._calculate_flash_sl_tp(
                indicators['current_price'],
                signals['action'],
                indicators['atr']
            )
            signals['stop_loss'] = sl_tp['stop_loss']
            signals['take_profit'] = sl_tp['take_profit']
            signals['risk_reward'] = sl_tp['risk_reward']

        return {
            'indicators': indicators,
            'signals': signals,
            'timeframe': config.FLASH_TIMEFRAME,  # Use configured flash timeframe (15m)
            'type': 'FLASH'
        }

    def _generate_flash_signals(self, indicators: dict) -> dict:
        """
        Generate flash signals with lower threshold (configurable, default 3.5+ points)
        """
        score = 0.0
        reasons = []

        current_price = indicators['current_price']

        # 1. RSI (0-2.5 points)
        if indicators['rsi'] < 25:
            score += 2.5
            reasons.append(f"🔥 RSI muy sobreventa ({indicators['rsi']:.1f})")
        elif indicators['rsi'] < self.rsi_oversold:
            score += 1.5
            reasons.append(f"RSI sobreventa ({indicators['rsi']:.1f})")
        elif indicators['rsi'] > 75:
            score -= 2.5
            reasons.append(f"🔥 RSI muy sobrecompra ({indicators['rsi']:.1f})")
        elif indicators['rsi'] > self.rsi_overbought:
            score -= 1.5
            reasons.append(f"RSI sobrecompra ({indicators['rsi']:.1f})")

        # 2. Volume (0-2 points)
        if indicators['volume_ratio'] > 1.8:
            score += 2.0 if score > 0 else -2.0
            reasons.append(f"📊 Volumen explosivo (+{(indicators['volume_ratio']-1)*100:.0f}%)")
        elif indicators['volume_ratio'] > 1.3:
            score += 1.0 if score > 0 else -1.0
            reasons.append(f"Volumen alto (+{(indicators['volume_ratio']-1)*100:.0f}%)")

        # 3. MACD (0-2 points)
        if indicators['macd_diff'] > 0 and indicators['macd'] > indicators['macd_signal']:
            score += 2.0
            reasons.append("MACD cruce alcista")
        elif indicators['macd_diff'] < 0 and indicators['macd'] < indicators['macd_signal']:
            score -= 2.0
            reasons.append("MACD cruce bajista")

        # 4. EMA Trend (0-1.5 points)
        if indicators['ema_short'] > indicators['ema_medium']:
            score += 1.5 if score > 0 else 0
            reasons.append("Tendencia alcista corto plazo")
        elif indicators['ema_short'] < indicators['ema_medium']:
            score -= 1.5 if score < 0 else 0
            reasons.append("Tendencia bajista corto plazo")

        # 5. Bollinger Bands (0-1.5 points)
        if current_price < indicators['bb_lower']:
            score += 1.5
            reasons.append("Precio bajo banda inferior BB")
        elif current_price > indicators['bb_upper']:
            score -= 1.5
            reasons.append("Precio sobre banda superior BB")

        # 6. Price momentum (safe BB position calculation)
        bb_range = indicators['bb_upper'] - indicators['bb_lower']
        if bb_range > 0:
            bb_position = (current_price - indicators['bb_lower']) / bb_range
            if bb_position < 0.2:
                score += 1.0
                reasons.append("Precio en zona de sobreventa (BB)")
            elif bb_position > 0.8:
                score -= 1.0
                reasons.append("Precio en zona de sobrecompra (BB)")

        # 7. Momentum Bonus (0-1 point) - NUEVO
        # Si RSI y MACD concuerdan en dirección, es más fuerte
        rsi_bullish = indicators['rsi'] < 40
        rsi_bearish = indicators['rsi'] > 60
        macd_bullish = indicators['macd_diff'] > 0
        macd_bearish = indicators['macd_diff'] < 0

        if rsi_bullish and macd_bullish and score > 0:
            score += 1.0
            reasons.append("🔥 Strong bullish momentum (RSI + MACD)")
        elif rsi_bearish and macd_bearish and score < 0:
            score -= 1.0
            reasons.append("🔥 Strong bearish momentum (RSI + MACD)")

        # Normalize score
        final_score = max(0, min(abs(score), 10))

        # Medium threshold for flash signals (5.0+ points, 50%+ confidence, MEDIUM risk)
        confidence = int((final_score / 10) * 100)

        if score >= self.flash_threshold and confidence >= self.min_confidence:
            action = 'BUY'
            risk_level = 'MEDIUM'  # Flash signals are medium risk (más selectivas)
        elif score <= -self.flash_threshold and confidence >= self.min_confidence:
            action = 'SELL'
            risk_level = 'MEDIUM'  # Flash signals are medium risk (más selectivas)
        else:
            action = 'HOLD'
            # Keep the real score for visibility (don't force to 0)
            confidence_status = f", confianza {confidence}%" if confidence > 0 else ""
            reasons.append(f'Señal flash insuficiente (necesita {self.flash_threshold}+ puntos y {self.min_confidence}%+ confianza, tiene {abs(score):.1f} pts{confidence_status})')

        return {
            'action': action,
            'score': round(final_score, 1),
            'max_score': 10,
            'strength': min(int(final_score / 2), 5),
            'reasons': reasons,
            'risk_level': risk_level if action != 'HOLD' else 'LOW',
            'confidence': confidence,
            'signal_type': 'FLASH',
            'timeframe': config.FLASH_TIMEFRAME  # Use configured flash timeframe (15m)
        }

    def _calculate_flash_sl_tp(self, entry_price: float, action: str, atr: float) -> dict:
        """
        Calculate tighter stop-loss and take-profit for flash signals (15m)
        SCALPING STRATEGY: TPs pequeños y frecuentes (0.3%, 0.8%, 1.5%)
        """
        # Tighter stops for flash trading
        atr_multiplier_sl = 1.5  # 1.5x ATR for stop

        # MÍNIMO porcentual para SL (protección contra SL = entry_price)
        min_sl_pct = 0.0015  # 0.15% mínimo de diferencia

        # SCALPING PURO: Single TP for quick wins
        # Closes 100% of position on first TP hit
        tp_pct = 0.005  # Single TP at 0.5% for pure scalping

        if action == 'BUY':
            # Calcular SL usando ATR
            stop_loss_atr = entry_price - (atr * atr_multiplier_sl)
            # Aplicar mínimo porcentual
            stop_loss_min = entry_price * (1 - min_sl_pct)
            # Usar el que esté más lejos (más protección)
            stop_loss = min(stop_loss_atr, stop_loss_min)

            # Single TP for scalping (above entry for BUY)
            tp = entry_price * (1 + tp_pct)
        else:  # SELL
            # Calcular SL usando ATR
            stop_loss_atr = entry_price + (atr * atr_multiplier_sl)
            # Aplicar mínimo porcentual
            stop_loss_min = entry_price * (1 + min_sl_pct)
            # Usar el que esté más lejos (más protección)
            stop_loss = max(stop_loss_atr, stop_loss_min)

            # Single TP for scalping (below entry for SELL)
            tp = entry_price * (1 - tp_pct)

        # Ajustar decimales de redondeo según el precio
        # Para precios bajos (< $10), usar más decimales para evitar que TP = entry_price
        if entry_price < 0.1:
            decimals = 6
        elif entry_price < 1.0:
            decimals = 5
        elif entry_price < 10.0:
            decimals = 4  # Fixed: was 2, now 4 for prices $1-$10
        else:
            decimals = 2

        tick_size = 10 ** (-decimals)

        # Redondear valores
        stop_loss_rounded = round(stop_loss, decimals)
        tp_rounded = round(tp, decimals)

        # PROTECCIÓN CRÍTICA: Verificar que SL ≠ entry_price después del redondeo
        # Si son iguales, forzar al menos 1 tick de diferencia
        if stop_loss_rounded == entry_price:
            if action == 'BUY':
                stop_loss_rounded = entry_price - tick_size
            else:  # SELL
                stop_loss_rounded = entry_price + tick_size
            logger.warning(
                f"⚠️ FLASH SL igualaba entry price ({entry_price}) después de redondeo. "
                f"Ajustado a {stop_loss_rounded} ({action})"
            )

        # PROTECCIÓN CRÍTICA: Verificar que TP ≠ entry_price después del redondeo
        # Si son iguales, forzar al menos 2 ticks de diferencia
        if tp_rounded == entry_price:
            if action == 'BUY':
                tp_rounded = entry_price + (tick_size * 2)
            else:  # SELL
                tp_rounded = entry_price - (tick_size * 2)
            logger.warning(
                f"⚠️ FLASH TP igualaba entry price ({entry_price}) después de redondeo. "
                f"Ajustado a {tp_rounded} ({action})"
            )

        # Calcular riesgo/recompensa con valores finales
        risk = abs(entry_price - stop_loss_rounded)
        reward = abs(tp_rounded - entry_price)
        risk_reward = round(reward / risk, 2) if risk > 0 else 0

        return {
            'stop_loss': stop_loss_rounded,
            'take_profit': {
                'tp': tp_rounded  # SCALPING: Single TP closes 100% of position
            },
            'risk_reward': risk_reward,
            'risk_amount': round(risk, decimals),
            'reward_amount': round(reward, decimals)
        }
