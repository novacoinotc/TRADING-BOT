"""
Liquidation Heatmap - Análisis de zonas de liquidación

Detecta niveles de precio con alta concentración de liquidaciones pendientes:
- Stop hunts: El precio tiende a ir a liquidar posiciones
- Zonas magnéticas: Áreas con muchas liquidaciones actúan como imán
- Oportunidades de reversión: Después de liquidar, el precio suele revertir

Ejemplo: Si hay $500M en liquidaciones en $42,000, hay alta probabilidad
de que el precio vaya a ese nivel antes de revertir.
"""

import logging
import requests
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


class LiquidationHeatmap:
    """
    Análisis de liquidaciones para detectar zonas críticas

    Mejora win rate en trades cerca de niveles de liquidación
    """

    def __init__(self, config):
        self.config = config

        # Parámetros optimizables
        self.enabled = getattr(config, 'LIQUIDATION_ANALYSIS_ENABLED', True)
        self.min_liquidation_volume = getattr(config, 'MIN_LIQUIDATION_VOLUME_USD', 1_000_000)  # $1M (0.5M-5M)
        self.proximity_threshold_pct = getattr(config, 'LIQUIDATION_PROXIMITY_THRESHOLD_PCT', 2.0)  # 2% (1-5%)
        self.boost_factor = getattr(config, 'LIQUIDATION_BOOST_FACTOR', 1.3)  # 1.1-1.5x
        self.lookback_hours = getattr(config, 'LIQUIDATION_LOOKBACK_HOURS', 24)  # 12-48h

        # Cache de liquidaciones {pair: {price_level: volume}}
        self.liquidation_data: Dict[str, Dict[float, float]] = defaultdict(dict)
        self.last_update: Dict[str, datetime] = {}
        self.update_interval = timedelta(minutes=15)  # Actualizar cada 15 min

        # API endpoints (múltiples fuentes)
        self.coinglass_api = "https://open-api.coinglass.com/public/v2/indicator/liquidation_heatmap"
        self.use_mock_data = getattr(config, 'USE_MOCK_LIQUIDATION_DATA', False)  # Para testing

        logger.info(f"LiquidationHeatmap initialized: min_volume=${self.min_liquidation_volume:,.0f}, proximity={self.proximity_threshold_pct}%")

    def fetch_liquidation_data(self, pair: str) -> bool:
        """
        Obtiene datos de liquidación de APIs externas

        Args:
            pair: Par de trading

        Returns:
            True si exitoso
        """
        if not self.enabled:
            return False

        # Verificar si necesita actualizar
        if pair in self.last_update:
            if datetime.now() - self.last_update[pair] < self.update_interval:
                return True  # Data reciente, skip

        try:
            # Mock data para testing (si no hay API key)
            if self.use_mock_data:
                return self._generate_mock_liquidation_data(pair)

            # Intentar Coinglass API
            symbol = pair.replace('/', '').replace('USDT', '')  # BTC/USDT -> BTC
            response = requests.get(
                self.coinglass_api,
                params={'symbol': symbol, 'timeframe': '24h'},
                timeout=5
            )

            if response.status_code == 200:
                data = response.json()
                self._parse_liquidation_data(pair, data)
                self.last_update[pair] = datetime.now()
                logger.debug(f"✅ Liquidation data actualizada para {pair}")
                return True

        except Exception as e:
            logger.warning(f"⚠️ Error fetching liquidation data para {pair}: {e}")

        return False

    def _generate_mock_liquidation_data(self, pair: str) -> bool:
        """
        Genera datos mock para testing (sin API)

        Simula distribución realista de liquidaciones
        """
        # Simular precio actual (esto debería venir del mercado real)
        # Por ahora, usar valores típicos
        base_prices = {
            'BTC/USDT': 42000,
            'ETH/USDT': 2200,
            'SOL/USDT': 100,
            'BNB/USDT': 300,
        }

        base_price = base_prices.get(pair, 100)

        # Generar niveles de liquidación mock
        # Concentración en niveles psicológicos y ±5%
        levels = []

        # Zona inferior (-2% a -5%)
        for i in range(3):
            price = base_price * (0.95 + i * 0.01)
            volume = np.random.uniform(1_000_000, 10_000_000)
            levels.append((price, volume))

        # Zona superior (+2% a +5%)
        for i in range(3):
            price = base_price * (1.02 + i * 0.01)
            volume = np.random.uniform(1_000_000, 10_000_000)
            levels.append((price, volume))

        # Almacenar
        self.liquidation_data[pair] = {price: vol for price, vol in levels}
        self.last_update[pair] = datetime.now()

        return True

    def _parse_liquidation_data(self, pair: str, data: Dict) -> None:
        """
        Parsea respuesta de API y extrae niveles de liquidación

        Args:
            pair: Par de trading
            data: Respuesta JSON de la API
        """
        # Estructura de Coinglass API (ejemplo):
        # {"data": [{"price": 42000, "volume": 5000000}, ...]}

        liquidations = {}

        try:
            # Validar que data existe y tiene el campo 'data'
            if data and isinstance(data, dict) and 'data' in data:
                data_list = data['data']

                # VALIDACIÓN CRÍTICA: Verificar que data_list NO es un string
                if isinstance(data_list, str):
                    logger.warning(f"⚠️ Liquidation API devolvió string en lugar de lista para {pair}: {data_list[:100]}")
                    self.liquidation_data[pair] = {}
                    return

                # Validar que data['data'] es iterable y no None
                if data_list is not None and isinstance(data_list, (list, tuple)):
                    for level in data_list:
                        if not isinstance(level, dict):
                            continue

                        price = float(level.get('price', 0))
                        volume = float(level.get('volume', 0))

                        if volume >= self.min_liquidation_volume:
                            liquidations[price] = volume
                else:
                    logger.warning(f"⚠️ Liquidation data['data'] is not iterable for {pair}: {type(data_list)}")
            else:
                logger.warning(f"⚠️ Invalid liquidation data format for {pair}: {type(data)}")

            self.liquidation_data[pair] = liquidations

        except Exception as e:
            logger.error(f"❌ Error parsing liquidation data: {e}")

    def get_nearest_liquidation_zone(self, pair: str, current_price: float) -> Optional[Tuple[float, float, str]]:
        """
        Obtiene zona de liquidación más cercana

        Args:
            pair: Par de trading
            current_price: Precio actual

        Returns:
            (price_level, volume, direction) o None
            direction: 'above' o 'below'
        """
        if pair not in self.liquidation_data or not self.liquidation_data[pair]:
            # Intentar fetch si no hay data
            self.fetch_liquidation_data(pair)

        if pair not in self.liquidation_data or not self.liquidation_data[pair]:
            return None

        liquidations = self.liquidation_data[pair]

        # Filtrar por proximidad
        proximity_range = current_price * (self.proximity_threshold_pct / 100)
        min_price = current_price - proximity_range
        max_price = current_price + proximity_range

        nearby_liquidations = {
            price: vol
            for price, vol in liquidations.items()
            if min_price <= price <= max_price
        }

        if not nearby_liquidations:
            return None

        # Encontrar nivel con mayor volumen
        max_level = max(nearby_liquidations.items(), key=lambda x: x[1])
        price_level, volume = max_level

        direction = 'above' if price_level > current_price else 'below'

        return (price_level, volume, direction)

    def is_near_liquidation_zone(self, pair: str, current_price: float) -> Tuple[bool, Optional[Dict]]:
        """
        Verifica si el precio está cerca de zona de liquidación

        Args:
            pair: Par de trading
            current_price: Precio actual

        Returns:
            (is_near, details_dict)
        """
        nearest = self.get_nearest_liquidation_zone(pair, current_price)

        if nearest is None:
            return False, None

        price_level, volume, direction = nearest

        distance_pct = abs((price_level - current_price) / current_price) * 100

        details = {
            'liquidation_price': price_level,
            'liquidation_volume': volume,
            'direction': direction,
            'distance_pct': distance_pct,
            'is_significant': volume >= self.min_liquidation_volume * 2  # 2x threshold = muy significativo
        }

        return True, details

    def get_liquidation_bias(self, pair: str, current_price: float) -> Tuple[str, float]:
        """
        Determina sesgo de liquidación (alcista/bajista)

        Args:
            pair: Par de trading
            current_price: Precio actual

        Returns:
            (bias, confidence)
            bias: 'bullish', 'bearish', 'neutral'
            confidence: 0-1
        """
        is_near, details = self.is_near_liquidation_zone(pair, current_price)

        if not is_near or details is None:
            return 'neutral', 0.0

        direction = details['direction']
        volume = details['liquidation_volume']
        distance_pct = details['distance_pct']

        # Lógica de stop hunt:
        # Si hay liquidaciones arriba → precio tenderá a subir a liquidarlas
        # Si hay liquidaciones abajo → precio tenderá a bajar a liquidarlas

        if direction == 'above':
            bias = 'bullish'  # Subir para liquidar longs/shorts arriba
        else:
            bias = 'bearish'  # Bajar para liquidar shorts/longs abajo

        # Confidence basada en volumen y proximidad
        volume_factor = min(volume / (self.min_liquidation_volume * 5), 1.0)  # Max at 5x threshold
        proximity_factor = 1.0 - (distance_pct / self.proximity_threshold_pct)  # Más cerca = más confidence

        confidence = (volume_factor * 0.6 + proximity_factor * 0.4)

        return bias, confidence

    def adjust_signal_confidence(self, pair: str, signal_side: str, current_price: float, base_confidence: float) -> float:
        """
        Ajusta confianza de señal considerando liquidaciones

        Args:
            pair: Par de trading
            signal_side: 'BUY' o 'SELL'
            current_price: Precio actual
            base_confidence: Confianza base (0-100)

        Returns:
            Confianza ajustada (0-100)
        """
        if not self.enabled:
            return base_confidence

        bias, liq_confidence = self.get_liquidation_bias(pair, current_price)

        if bias == 'neutral':
            return base_confidence

        # Boost si señal alineada con liquidation bias
        if (signal_side == 'BUY' and bias == 'bullish') or (signal_side == 'SELL' and bias == 'bearish'):
            boost = self.boost_factor * liq_confidence
            adjusted = base_confidence * boost

            logger.info(f"💀 Liquidation boost para {pair}: {base_confidence:.1f}% → {adjusted:.1f}% (bias={bias}, conf={liq_confidence:.2f})")

            return min(adjusted, 100.0)  # Cap at 100

        # Penalty si señal contraria
        elif (signal_side == 'BUY' and bias == 'bearish') or (signal_side == 'SELL' and bias == 'bullish'):
            penalty = 1.0 - (0.2 * liq_confidence)  # Max 20% penalty
            adjusted = base_confidence * penalty

            logger.warning(f"⚠️ Liquidation penalty para {pair}: {base_confidence:.1f}% → {adjusted:.1f}% (señal contraria a bias)")

            return adjusted

        return base_confidence

    def get_statistics(self) -> Dict:
        """
        Estadísticas de liquidaciones

        Returns:
            Dict con métricas
        """
        total_liquidations = sum(len(levels) for levels in self.liquidation_data.values())
        total_volume = sum(
            sum(volumes.values())
            for volumes in self.liquidation_data.values()
        )

        return {
            'enabled': self.enabled,
            'pairs_tracked': len(self.liquidation_data),
            'total_liquidation_levels': total_liquidations,
            'total_volume_usd': total_volume,
            'min_volume_threshold': self.min_liquidation_volume,
            'proximity_threshold_pct': self.proximity_threshold_pct,
            'last_updates': {
                pair: dt.isoformat()
                for pair, dt in self.last_update.items()
            }
        }


# Parámetros optimizables para config.py
LIQUIDATION_PARAMS = {
    'LIQUIDATION_ANALYSIS_ENABLED': True,
    'MIN_LIQUIDATION_VOLUME_USD': 1_000_000,  # $1M (0.5M-5M optimizable)
    'LIQUIDATION_PROXIMITY_THRESHOLD_PCT': 2.0,  # 2% (1-5% optimizable)
    'LIQUIDATION_BOOST_FACTOR': 1.3,  # 1.1-1.5x (optimizable)
    'LIQUIDATION_LOOKBACK_HOURS': 24,  # 12-48h (optimizable)
    'USE_MOCK_LIQUIDATION_DATA': False,  # True para testing sin API
}
