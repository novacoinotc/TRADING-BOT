"""
Order Flow Imbalance - Análisis de desbalance bid/ask en tiempo real

Detecta presión compradora/vendedora antes que el precio:
- Bid/Ask ratio > 2:1 = presión compradora fuerte
- Bid/Ask ratio < 1:2 = presión vendedora fuerte
- Delta volume = volumen comprador - volumen vendedor

Ejemplo: Bid/Ask 3:1 con delta +500 BTC = señal bullish fuerte
"""

import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class OrderFlowImbalance:
    """Análisis de order flow para detectar momentum temprano"""

    def __init__(self, config):
        self.config = config
        self.enabled = config.get('ORDER_FLOW_ENABLED', True)
        self.strong_imbalance_ratio = config.get('STRONG_IMBALANCE_RATIO', 2.5)  # 2.0-3.5
        self.moderate_imbalance_ratio = config.get('MODERATE_IMBALANCE_RATIO', 1.5)  # 1.3-2.0
        self.boost_strong = config.get('ORDER_FLOW_BOOST_STRONG', 1.3)  # 1.2-1.5x
        self.boost_moderate = config.get('ORDER_FLOW_BOOST_MODERATE', 1.15)  # 1.1-1.3x

    def analyze_orderbook(self, orderbook: Dict) -> Tuple[str, float, float]:
        """
        Analiza order book para detectar imbalance

        Args:
            orderbook: Dict con 'bids' y 'asks'

        Returns:
            (bias, ratio, confidence)
        """
        if not self.enabled or not orderbook:
            return 'neutral', 1.0, 0.0

        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])

        if not bids or not asks:
            return 'neutral', 1.0, 0.0

        # Sumar volumen top 10 bids/asks
        bid_volume = sum([bid[1] for bid in bids[:10]])
        ask_volume = sum([ask[1] for ask in asks[:10]])

        if ask_volume == 0:
            return 'bullish', 10.0, 1.0

        ratio = bid_volume / ask_volume

        # Determinar bias
        if ratio >= self.strong_imbalance_ratio:
            return 'bullish', ratio, 1.0
        elif ratio >= self.moderate_imbalance_ratio:
            return 'bullish', ratio, 0.6
        elif ratio <= (1 / self.strong_imbalance_ratio):
            return 'bearish', ratio, 1.0
        elif ratio <= (1 / self.moderate_imbalance_ratio):
            return 'bearish', ratio, 0.6
        else:
            return 'neutral', ratio, 0.0

    def adjust_signal_confidence(
        self,
        signal_side: str,
        base_confidence: float,
        orderbook: Dict
    ) -> float:
        """Ajusta confianza según order flow"""
        bias, ratio, strength = self.analyze_orderbook(orderbook)

        if bias == 'neutral':
            return base_confidence

        # Boost si alineado
        if (signal_side == 'BUY' and bias == 'bullish') or \
           (signal_side == 'SELL' and bias == 'bearish'):

            if strength >= 0.9:
                boost = self.boost_strong
            else:
                boost = self.boost_moderate

            adjusted = base_confidence * boost
            logger.info(f"💹 Order flow boost: ratio={ratio:.2f} → {base_confidence:.1f}% → {adjusted:.1f}%")
            return min(adjusted, 100.0)

        return base_confidence

    def get_statistics(self) -> Dict:
        """Estadísticas de order flow"""
        return {
            'enabled': self.enabled,
            'strong_imbalance_ratio': self.strong_imbalance_ratio,
            'boost_strong': self.boost_strong
        }


ORDER_FLOW_PARAMS = {
    'ORDER_FLOW_ENABLED': True,
    'STRONG_IMBALANCE_RATIO': 2.5,  # 2.0-3.5
    'MODERATE_IMBALANCE_RATIO': 1.5,  # 1.3-2.0
    'ORDER_FLOW_BOOST_STRONG': 1.3,  # 1.2-1.5x
    'ORDER_FLOW_BOOST_MODERATE': 1.15,  # 1.1-1.3x
}
