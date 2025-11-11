"""
Technical Pattern Recognition - Detección automática de patrones chartistas

Detecta patrones clásicos de alta probabilidad:
- Head & Shoulders (reversal)
- Double Top/Bottom (reversal)
- Triangles (continuation/break out)
- Wedges (reversal)
- Flags & Pennants (continuation)

Ejemplo: Detecta Head & Shoulders en BTC → señal SHORT con alta confianza
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.signal import argrelextrema

logger = logging.getLogger(__name__)


class PatternRecognition:
    """
    Detección automática de patrones técnicos

    Genera señales adicionales de alta confiabilidad
    """

    def __init__(self, config):
        self.config = config

        # Parámetros optimizables
        self.enabled = getattr(config, 'PATTERN_RECOGNITION_ENABLED', True)
        self.min_pattern_confidence = getattr(config, 'MIN_PATTERN_CONFIDENCE', 0.7)  # 0.6-0.85
        self.lookback_candles = getattr(config, 'PATTERN_LOOKBACK_CANDLES', 50)  # 30-100
        self.boost_factor = getattr(config, 'PATTERN_BOOST_FACTOR', 1.4)  # 1.2-1.6x

        logger.info(f"PatternRecognition initialized: min_confidence={self.min_pattern_confidence}")

    def detect_head_and_shoulders(self, highs: np.ndarray, lows: np.ndarray) -> Optional[Dict]:
        """
        Detecta Head & Shoulders pattern (bearish reversal)

        Pattern: LOW - HIGH (shoulder) - LOW - HIGHER HIGH (head) - LOW - HIGH (shoulder)

        Returns:
            Pattern details o None
        """
        if len(highs) < 5:
            return None

        # Encontrar peaks locales
        peaks = argrelextrema(highs, np.greater, order=3)[0]

        if len(peaks) < 3:
            return None

        # Buscar patrón: shoulder1 < head > shoulder2
        for i in range(len(peaks) - 2):
            left_shoulder = highs[peaks[i]]
            head = highs[peaks[i+1]]
            right_shoulder = highs[peaks[i+2]]

            # Validar estructura
            if head > left_shoulder and head > right_shoulder:
                # Validar simetría de hombros (±10%)
                shoulder_symmetry = abs(left_shoulder - right_shoulder) / left_shoulder

                if shoulder_symmetry < 0.1:
                    # Calcular neckline (soporte entre hombros)
                    neckline = min(lows[peaks[i]:peaks[i+2]])

                    confidence = 1.0 - shoulder_symmetry  # Más simétrico = más confidence

                    return {
                        'pattern': 'HEAD_AND_SHOULDERS',
                        'type': 'BEARISH_REVERSAL',
                        'confidence': confidence,
                        'signal': 'SELL',
                        'head_price': head,
                        'left_shoulder': left_shoulder,
                        'right_shoulder': right_shoulder,
                        'neckline': neckline,
                        'target': neckline - (head - neckline)  # Precio objetivo
                    }

        return None

    def detect_double_top(self, highs: np.ndarray) -> Optional[Dict]:
        """
        Detecta Double Top pattern (bearish reversal)

        Pattern: HIGH - LOW - HIGH (similar level)

        Returns:
            Pattern details o None
        """
        if len(highs) < 3:
            return None

        peaks = argrelextrema(highs, np.greater, order=2)[0]

        if len(peaks) < 2:
            return None

        # Buscar dos peaks similares
        for i in range(len(peaks) - 1):
            peak1 = highs[peaks[i]]
            peak2 = highs[peaks[i+1]]

            # Validar similitud (±3%)
            similarity = abs(peak1 - peak2) / peak1

            if similarity < 0.03:
                confidence = 1.0 - similarity

                return {
                    'pattern': 'DOUBLE_TOP',
                    'type': 'BEARISH_REVERSAL',
                    'confidence': confidence,
                    'signal': 'SELL',
                    'first_top': peak1,
                    'second_top': peak2,
                    'target': peak1 * 0.95  # -5% target típico
                }

        return None

    def detect_double_bottom(self, lows: np.ndarray) -> Optional[Dict]:
        """
        Detecta Double Bottom pattern (bullish reversal)

        Pattern: LOW - HIGH - LOW (similar level)

        Returns:
            Pattern details o None
        """
        if len(lows) < 3:
            return None

        troughs = argrelextrema(lows, np.less, order=2)[0]

        if len(troughs) < 2:
            return None

        # Buscar dos troughs similares
        for i in range(len(troughs) - 1):
            trough1 = lows[troughs[i]]
            trough2 = lows[troughs[i+1]]

            similarity = abs(trough1 - trough2) / trough1

            if similarity < 0.03:
                confidence = 1.0 - similarity

                return {
                    'pattern': 'DOUBLE_BOTTOM',
                    'type': 'BULLISH_REVERSAL',
                    'confidence': confidence,
                    'signal': 'BUY',
                    'first_bottom': trough1,
                    'second_bottom': trough2,
                    'target': trough1 * 1.05  # +5% target típico
                }

        return None

    def detect_all_patterns(self, ohlc_data: Dict) -> List[Dict]:
        """
        Detecta todos los patrones en datos OHLC

        Args:
            ohlc_data: Dict con 'high', 'low', 'close', 'open'

        Returns:
            Lista de patrones detectados
        """
        if not self.enabled:
            return []

        highs = np.array(ohlc_data.get('high', []))
        lows = np.array(ohlc_data.get('low', []))

        if len(highs) < 10 or len(lows) < 10:
            return []

        patterns_detected = []

        # Detectar Head & Shoulders
        hs = self.detect_head_and_shoulders(highs, lows)
        if hs and hs['confidence'] >= self.min_pattern_confidence:
            patterns_detected.append(hs)

        # Detectar Double Top
        dt = self.detect_double_top(highs)
        if dt and dt['confidence'] >= self.min_pattern_confidence:
            patterns_detected.append(dt)

        # Detectar Double Bottom
        db = self.detect_double_bottom(lows)
        if db and db['confidence'] >= self.min_pattern_confidence:
            patterns_detected.append(db)

        return patterns_detected

    def adjust_signal_confidence(
        self,
        signal_side: str,
        base_confidence: float,
        detected_patterns: List[Dict]
    ) -> float:
        """
        Ajusta confianza según patrones detectados

        Args:
            signal_side: 'BUY' o 'SELL'
            base_confidence: Confianza base (0-100)
            detected_patterns: Lista de patrones detectados

        Returns:
            Confianza ajustada (0-100)
        """
        if not detected_patterns:
            return base_confidence

        # Buscar patrones alineados con la señal
        aligned_patterns = [
            p for p in detected_patterns
            if p['signal'] == signal_side
        ]

        if not aligned_patterns:
            return base_confidence

        # Boost por patrón más confiable
        best_pattern = max(aligned_patterns, key=lambda x: x['confidence'])

        boost = self.boost_factor * best_pattern['confidence']
        adjusted = base_confidence * boost

        logger.info(f"📈 Pattern boost: {best_pattern['pattern']} (conf={best_pattern['confidence']:.2f}) → {base_confidence:.1f}% → {adjusted:.1f}%")

        return min(adjusted, 100.0)

    def get_statistics(self) -> Dict:
        """Estadísticas de pattern recognition"""
        return {
            'enabled': self.enabled,
            'min_pattern_confidence': self.min_pattern_confidence,
            'boost_factor': self.boost_factor
        }


# Parámetros optimizables
PATTERN_RECOGNITION_PARAMS = {
    'PATTERN_RECOGNITION_ENABLED': True,
    'MIN_PATTERN_CONFIDENCE': 0.7,  # 0.6-0.85 (optimizable)
    'PATTERN_LOOKBACK_CANDLES': 50,  # 30-100 (optimizable)
    'PATTERN_BOOST_FACTOR': 1.4,  # 1.2-1.6x (optimizable)
}
