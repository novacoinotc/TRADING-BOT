"""
Funding Rate Analyzer - Análisis de funding rate de perpetuals

El funding rate indica sentiment extremo en futures:
- Funding rate muy positivo (>0.1%) → Muchos longs, oportunidad de short
- Funding rate muy negativo (<-0.1%) → Muchos shorts, oportunidad de long
- Funding extremo → Overleveraged traders, alta probabilidad de squeeze

Ejemplo: BTC funding rate +0.15% (muy alto) → Detecta top local, señal SHORT
"""

import logging
import ccxt
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)


class FundingRateAnalyzer:
    """
    Análisis de funding rate para detectar sentiment extremo

    Mejora detección de tops/bottoms locales
    """

    def __init__(self, config, exchange: Optional[ccxt.Exchange] = None):
        self.config = config
        self.exchange = exchange

        # Parámetros optimizables
        self.enabled = getattr(config, 'FUNDING_RATE_ANALYSIS_ENABLED', True)
        self.extreme_positive_threshold = getattr(config, 'FUNDING_EXTREME_POSITIVE', 0.10)  # 0.08-0.15%
        self.extreme_negative_threshold = getattr(config, 'FUNDING_EXTREME_NEGATIVE', -0.10)  # -0.15 a -0.08%
        self.high_positive_threshold = getattr(config, 'FUNDING_HIGH_POSITIVE', 0.05)  # 0.03-0.08%
        self.high_negative_threshold = getattr(config, 'FUNDING_HIGH_NEGATIVE', -0.05)  # -0.08 a -0.03%
        self.boost_factor_extreme = getattr(config, 'FUNDING_BOOST_EXTREME', 1.5)  # 1.3-1.8x
        self.boost_factor_high = getattr(config, 'FUNDING_BOOST_HIGH', 1.2)  # 1.1-1.4x

        # Cache de funding rates {pair: deque([rates])}
        self.funding_history: Dict[str, deque] = {}
        self.current_funding: Dict[str, float] = {}
        self.last_update: Dict[str, datetime] = {}
        self.update_interval = timedelta(hours=1)  # Actualizar cada hora

        logger.info(f"FundingRateAnalyzer initialized: extreme_threshold={self.extreme_positive_threshold}%")

    def fetch_funding_rate(self, pair: str) -> Optional[float]:
        """
        Obtiene funding rate actual de un par

        Args:
            pair: Par de trading (e.g., 'BTC/USDT')

        Returns:
            Funding rate (%) o None
        """
        if not self.enabled or not self.exchange:
            return None

        # Verificar si necesita actualizar
        if pair in self.last_update:
            if datetime.now() - self.last_update[pair] < self.update_interval:
                return self.current_funding.get(pair)

        try:
            # Obtener funding rate del exchange
            # CCXT método: fetchFundingRate(symbol)
            funding_info = self.exchange.fetch_funding_rate(pair)

            if funding_info and 'fundingRate' in funding_info:
                # Funding rate típicamente en formato decimal (0.0001 = 0.01%)
                funding_rate = funding_info['fundingRate'] * 100  # Convertir a %

                # Actualizar cache
                self.current_funding[pair] = funding_rate
                self.last_update[pair] = datetime.now()

                # Agregar a historial
                if pair not in self.funding_history:
                    self.funding_history[pair] = deque(maxlen=24)  # Últimas 24 horas
                self.funding_history[pair].append(funding_rate)

                logger.debug(f"📊 Funding rate {pair}: {funding_rate:.3f}%")

                return funding_rate

        except Exception as e:
            logger.warning(f"⚠️ Error fetching funding rate para {pair}: {e}")

        return None

    def get_funding_sentiment(self, pair: str) -> Tuple[str, float, str]:
        """
        Determina sentiment basado en funding rate

        Args:
            pair: Par de trading

        Returns:
            (sentiment, strength, signal)
            sentiment: 'bullish', 'bearish', 'neutral'
            strength: 0-1
            signal: 'EXTREME_SHORT', 'HIGH_SHORT', 'EXTREME_LONG', 'HIGH_LONG', 'NEUTRAL'
        """
        funding_rate = self.fetch_funding_rate(pair)

        if funding_rate is None:
            return 'neutral', 0.0, 'NEUTRAL'

        # Clasificar funding rate
        if funding_rate >= self.extreme_positive_threshold:
            # Funding muy positivo = muchos longs = oportunidad SHORT
            return 'bearish', 1.0, 'EXTREME_SHORT'

        elif funding_rate >= self.high_positive_threshold:
            return 'bearish', 0.6, 'HIGH_SHORT'

        elif funding_rate <= self.extreme_negative_threshold:
            # Funding muy negativo = muchos shorts = oportunidad LONG
            return 'bullish', 1.0, 'EXTREME_LONG'

        elif funding_rate <= self.high_negative_threshold:
            return 'bullish', 0.6, 'HIGH_LONG'

        else:
            return 'neutral', 0.0, 'NEUTRAL'

    def adjust_signal_confidence(self, pair: str, signal_side: str, base_confidence: float) -> float:
        """
        Ajusta confianza de señal considerando funding rate

        Args:
            pair: Par de trading
            signal_side: 'BUY' o 'SELL'
            base_confidence: Confianza base (0-100)

        Returns:
            Confianza ajustada (0-100)
        """
        if not self.enabled:
            return base_confidence

        sentiment, strength, signal = self.get_funding_sentiment(pair)

        if sentiment == 'neutral':
            return base_confidence

        # Lógica contrarian: funding extremo indica reversión
        # Funding positivo extremo → SHORT (bearish)
        # Funding negativo extremo → LONG (bullish)

        if signal == 'EXTREME_SHORT' and signal_side == 'SELL':
            # Señal SHORT alineada con funding extremo positivo
            boost = self.boost_factor_extreme
            adjusted = base_confidence * boost
            logger.info(f"💸 Funding EXTREME boost para SHORT {pair}: {base_confidence:.1f}% → {adjusted:.1f}%")
            return min(adjusted, 100.0)

        elif signal == 'EXTREME_LONG' and signal_side == 'BUY':
            # Señal LONG alineada con funding extremo negativo
            boost = self.boost_factor_extreme
            adjusted = base_confidence * boost
            logger.info(f"💸 Funding EXTREME boost para LONG {pair}: {base_confidence:.1f}% → {adjusted:.1f}%")
            return min(adjusted, 100.0)

        elif signal == 'HIGH_SHORT' and signal_side == 'SELL':
            boost = self.boost_factor_high
            adjusted = base_confidence * boost
            logger.info(f"💰 Funding HIGH boost para SHORT {pair}: {base_confidence:.1f}% → {adjusted:.1f}%")
            return min(adjusted, 100.0)

        elif signal == 'HIGH_LONG' and signal_side == 'BUY':
            boost = self.boost_factor_high
            adjusted = base_confidence * boost
            logger.info(f"💰 Funding HIGH boost para LONG {pair}: {base_confidence:.1f}% → {adjusted:.1f}%")
            return min(adjusted, 100.0)

        # Penalty si señal contraria a funding
        elif (signal in ['EXTREME_SHORT', 'HIGH_SHORT'] and signal_side == 'BUY') or \
             (signal in ['EXTREME_LONG', 'HIGH_LONG'] and signal_side == 'SELL'):
            penalty = 0.8  # 20% penalty
            adjusted = base_confidence * penalty
            logger.warning(f"⚠️ Funding penalty para {pair}: {base_confidence:.1f}% → {adjusted:.1f}% (contrarian)")
            return adjusted

        return base_confidence

    def get_funding_trend(self, pair: str) -> Optional[str]:
        """
        Detecta tendencia de funding rate (subiendo/bajando)

        Args:
            pair: Par de trading

        Returns:
            'rising', 'falling', 'stable', o None
        """
        if pair not in self.funding_history or len(self.funding_history[pair]) < 3:
            return None

        history = list(self.funding_history[pair])

        # Comparar últimos 3 valores
        recent_3 = history[-3:]

        if recent_3[-1] > recent_3[-2] > recent_3[-3]:
            return 'rising'  # Funding subiendo = longs aumentando
        elif recent_3[-1] < recent_3[-2] < recent_3[-3]:
            return 'falling'  # Funding bajando = longs disminuyendo
        else:
            return 'stable'

    def get_statistics(self) -> Dict:
        """
        Estadísticas de funding rate

        Returns:
            Dict con métricas
        """
        return {
            'enabled': self.enabled,
            'pairs_tracked': len(self.current_funding),
            'current_funding_rates': self.current_funding,
            'extreme_positive_threshold': self.extreme_positive_threshold,
            'extreme_negative_threshold': self.extreme_negative_threshold,
            'last_updates': {
                pair: dt.isoformat()
                for pair, dt in self.last_update.items()
            }
        }


# Parámetros optimizables para config.py
FUNDING_RATE_PARAMS = {
    'FUNDING_RATE_ANALYSIS_ENABLED': True,
    'FUNDING_EXTREME_POSITIVE': 0.10,  # 0.08-0.15% (optimizable)
    'FUNDING_EXTREME_NEGATIVE': -0.10,  # -0.15 a -0.08% (optimizable)
    'FUNDING_HIGH_POSITIVE': 0.05,  # 0.03-0.08% (optimizable)
    'FUNDING_HIGH_NEGATIVE': -0.05,  # -0.08 a -0.03% (optimizable)
    'FUNDING_BOOST_EXTREME': 1.5,  # 1.3-1.8x (optimizable)
    'FUNDING_BOOST_HIGH': 1.2,  # 1.1-1.4x (optimizable)
}
