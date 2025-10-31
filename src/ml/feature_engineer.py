"""
Feature Engineer - Crea features para Machine Learning desde indicadores técnicos
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Transforma indicadores técnicos en features para ML
    Crea features avanzadas combinando múltiples indicadores
    """

    def __init__(self):
        self.feature_names = []

    def create_features(self, indicators: Dict, signals: Dict, mtf_indicators: Dict = None) -> Dict:
        """
        Crea conjunto completo de features desde indicadores

        Args:
            indicators: Dict con indicadores (RSI, MACD, etc.)
            signals: Dict con señales generadas
            mtf_indicators: Multi-timeframe indicators (opcional)

        Returns:
            Dict con features para ML
        """
        features = {}

        # === FEATURES BÁSICAS ===

        # RSI
        features['rsi'] = indicators.get('rsi', 50.0)
        features['rsi_oversold'] = 1 if features['rsi'] < 30 else 0
        features['rsi_overbought'] = 1 if features['rsi'] > 70 else 0
        features['rsi_extreme'] = 1 if (features['rsi'] < 25 or features['rsi'] > 75) else 0

        # MACD
        features['macd'] = indicators.get('macd', 0.0)
        features['macd_signal'] = indicators.get('macd_signal', 0.0)
        features['macd_diff'] = indicators.get('macd_diff', 0.0)
        features['macd_bullish'] = 1 if features['macd'] > features['macd_signal'] else 0

        # EMAs
        features['ema_short'] = indicators.get('ema_short', 0.0)
        features['ema_medium'] = indicators.get('ema_medium', 0.0)
        features['ema_long'] = indicators.get('ema_long', 0.0)
        features['ema_trend'] = 1 if (features['ema_short'] > features['ema_medium'] > features['ema_long']) else 0

        # Bollinger Bands
        features['bb_upper'] = indicators.get('bb_upper', 0.0)
        features['bb_middle'] = indicators.get('bb_middle', 0.0)
        features['bb_lower'] = indicators.get('bb_lower', 0.0)
        features['bb_width'] = indicators.get('bb_width', 0.0)

        current_price = indicators.get('current_price', 0.0)
        if features['bb_upper'] > features['bb_lower']:
            bb_position = (current_price - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
            features['bb_position'] = bb_position
            features['bb_near_upper'] = 1 if bb_position > 0.8 else 0
            features['bb_near_lower'] = 1 if bb_position < 0.2 else 0
        else:
            features['bb_position'] = 0.5
            features['bb_near_upper'] = 0
            features['bb_near_lower'] = 0

        # Volume
        features['volume_ratio'] = indicators.get('volume_ratio', 1.0)
        features['high_volume'] = 1 if features['volume_ratio'] > 1.5 else 0
        features['low_volume'] = 1 if features['volume_ratio'] < 0.7 else 0

        # ADX (Trend Strength)
        features['adx'] = indicators.get('adx', 0.0)
        features['strong_trend'] = 1 if features['adx'] > 25 else 0
        features['weak_trend'] = 1 if features['adx'] < 20 else 0

        # ATR (Volatility)
        features['atr'] = indicators.get('atr', 0.0)

        # Divergencias
        divergences = indicators.get('divergences', {})
        features['rsi_bullish_div'] = 1 if divergences.get('rsi_divergence') == 'bullish' else 0
        features['rsi_bearish_div'] = 1 if divergences.get('rsi_divergence') == 'bearish' else 0

        # === FEATURES AVANZADAS (COMBINACIONES) ===

        # Momentum combinado
        features['momentum_score'] = (
            (100 - features['rsi']) / 100 +  # RSI invertido para sobreventa
            (1 if features['macd_bullish'] else -1) +
            features['volume_ratio']
        ) / 3.0

        # Trend strength combinado
        features['trend_strength'] = (
            features['adx'] / 100.0 +
            (1 if features['ema_trend'] else 0) +
            (features['bb_width'] / 10.0)
        ) / 3.0

        # Signal quality (desde el análisis)
        features['signal_score'] = signals.get('score', 0.0)
        features['signal_confidence'] = signals.get('confidence', 0.0)
        features['signal_strength'] = signals.get('strength', 0.0)

        # === FEATURES MULTI-TIMEFRAME ===
        if mtf_indicators:
            mtf_trends = mtf_indicators.get('mtf_trends', {})

            features['mtf_1h_bullish'] = 1 if mtf_trends.get('1h') == 'buy' else 0
            features['mtf_4h_bullish'] = 1 if mtf_trends.get('4h') == 'buy' else 0
            features['mtf_1d_bullish'] = 1 if mtf_trends.get('1d') == 'buy' else 0

            features['mtf_alignment'] = (
                features['mtf_1h_bullish'] +
                features['mtf_4h_bullish'] +
                features['mtf_1d_bullish']
            ) / 3.0

        # === FEATURES DE CONTEXTO ===

        # Support/Resistance
        sr = indicators.get('support_resistance', {})
        features['distance_to_support'] = sr.get('distance_to_support', 100.0)
        features['distance_to_resistance'] = sr.get('distance_to_resistance', 100.0)
        features['near_support'] = 1 if features['distance_to_support'] < 2.0 else 0
        features['near_resistance'] = 1 if features['distance_to_resistance'] < 2.0 else 0

        # === FEATURES DERIVADAS ===

        # Ratios
        if features['ema_medium'] > 0:
            features['price_to_ema_ratio'] = current_price / features['ema_medium']
        else:
            features['price_to_ema_ratio'] = 1.0

        # Volatility ratio
        if features['atr'] > 0 and current_price > 0:
            features['volatility_pct'] = (features['atr'] / current_price) * 100
        else:
            features['volatility_pct'] = 0.0

        # Guardar nombres de features
        self.feature_names = list(features.keys())

        return features

    def features_to_array(self, features: Dict) -> np.ndarray:
        """
        Convierte Dict de features a numpy array para ML

        Args:
            features: Dict con features

        Returns:
            numpy array con valores
        """
        if not self.feature_names:
            self.feature_names = list(features.keys())

        return np.array([features.get(name, 0.0) for name in self.feature_names])

    def create_features_dataframe(self, features_list: List[Dict]) -> pd.DataFrame:
        """
        Convierte lista de features a DataFrame para entrenamiento

        Args:
            features_list: Lista de Dicts con features

        Returns:
            DataFrame con todas las features
        """
        return pd.DataFrame(features_list)

    def get_feature_importance_names(self) -> List[str]:
        """Retorna nombres de todas las features"""
        return self.feature_names.copy()
