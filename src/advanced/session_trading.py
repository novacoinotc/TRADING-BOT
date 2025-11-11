"""
Session-Based Trading - Trading según sesiones de mercado

Ajusta agresividad según la sesión:
- Asian session (baja volatilidad)
- European session (media volatilidad)
- US session (alta volatilidad)
- Overlaps (máxima volatilidad)

Ejemplo: US market open (9:30 AM EST) = boost 1.3x en agresividad
"""

import logging
from datetime import datetime, time
from typing import Dict, Tuple
import pytz

logger = logging.getLogger(__name__)


class SessionBasedTrading:
    """Trading adaptado a sesiones de mercado"""

    def __init__(self, config):
        self.config = config
        self.enabled = getattr(config, 'SESSION_TRADING_ENABLED', True)
        self.us_open_boost = getattr(config, 'US_OPEN_BOOST', 1.3)  # 1.2-1.5x
        self.overlap_boost = getattr(config, 'SESSION_OVERLAP_BOOST', 1.2)  # 1.1-1.4x
        self.asian_penalty = getattr(config, 'ASIAN_SESSION_PENALTY', 0.9)  # 0.85-0.95x

    def get_current_session(self) -> Tuple[str, float]:
        """
        Determina sesión actual y multiplier de agresividad

        Returns:
            (session_name, multiplier)
        """
        if not self.enabled:
            return 'NONE', 1.0

        utc_now = datetime.now(pytz.UTC)
        hour_utc = utc_now.hour

        # Asian session: 00:00-08:00 UTC
        if 0 <= hour_utc < 8:
            return 'ASIAN', self.asian_penalty

        # European session: 08:00-16:00 UTC
        elif 8 <= hour_utc < 13:
            return 'EUROPEAN', 1.0

        # EU-US overlap: 13:00-16:00 UTC (alta volatilidad)
        elif 13 <= hour_utc < 16:
            return 'EU_US_OVERLAP', self.overlap_boost

        # US session: 16:00-22:00 UTC (NYSE open)
        elif 16 <= hour_utc < 22:
            return 'US', self.us_open_boost

        # After hours
        else:
            return 'AFTER_HOURS', 0.95

    def adjust_position_size(self, base_size_pct: float) -> float:
        """Ajusta tamaño de posición según sesión"""
        session, multiplier = self.get_current_session()
        adjusted = base_size_pct * multiplier
        return max(1.0, min(adjusted, 12.0))  # Clamp 1-12%

    def get_statistics(self) -> Dict:
        """Estadísticas de session trading"""
        session, multiplier = self.get_current_session()
        return {
            'enabled': self.enabled,
            'current_session': session,
            'current_multiplier': multiplier
        }


SESSION_TRADING_PARAMS = {
    'SESSION_TRADING_ENABLED': True,
    'US_OPEN_BOOST': 1.3,  # 1.2-1.5x
    'SESSION_OVERLAP_BOOST': 1.2,  # 1.1-1.4x
    'ASIAN_SESSION_PENALTY': 0.9,  # 0.85-0.95x
}
