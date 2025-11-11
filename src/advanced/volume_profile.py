"""
Volume Profile & POC (Point of Control) - Análisis de volumen por nivel de precio

Identifica zonas de alto volumen que actúan como soporte/resistencia real:
- POC (Point of Control): Nivel con máximo volumen = zona magnética
- Value Area: Zona donde ocurre 70% del volumen (alta probabilidad)
- High/Low Volume Nodes: Niveles de aceptación/rechazo

Ejemplo: POC en $41,500 con 30% del volumen = muy fuerte zona de soporte
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class VolumeNode:
    """Nodo de volumen en un nivel de precio"""
    price: float
    volume: float
    percentage: float  # % del volumen total


class VolumeProfile:
    """
    Análisis de Volume Profile para detectar zonas de valor

    Mejora win rate en trades cerca de POC y Value Area
    """

    def __init__(self, config):
        self.config = config

        # Parámetros optimizables
        self.enabled = getattr(config, 'VOLUME_PROFILE_ENABLED', True)
        self.lookback_periods = getattr(config, 'VOLUME_PROFILE_LOOKBACK', 100)  # 50-200
        self.price_bins = getattr(config, 'VOLUME_PROFILE_BINS', 50)  # 30-100
        self.value_area_pct = getattr(config, 'VOLUME_PROFILE_VALUE_AREA', 70.0)  # 65-75%
        self.poc_proximity_pct = getattr(config, 'POC_PROXIMITY_PCT', 1.0)  # 0.5-2%
        self.boost_factor_poc = getattr(config, 'POC_BOOST_FACTOR', 1.3)  # 1.2-1.5x
        self.boost_factor_value_area = getattr(config, 'VALUE_AREA_BOOST_FACTOR', 1.15)  # 1.1-1.3x

        # Cache de volume profiles {pair: VolumeProfileData}
        self.volume_profiles: Dict[str, Dict] = {}

        logger.info(f"VolumeProfile initialized: lookback={self.lookback_periods}, bins={self.price_bins}")

    def calculate_volume_profile(
        self,
        pair: str,
        prices: List[float],
        volumes: List[float]
    ) -> Dict:
        """
        Calcula volume profile de datos OHLCV

        Args:
            pair: Par de trading
            prices: Lista de precios (típicamente close prices)
            volumes: Lista de volúmenes

        Returns:
            Dict con POC, Value Area, y distribución
        """
        if len(prices) < 10 or len(prices) != len(volumes):
            return {}

        # Definir bins de precio
        min_price = min(prices)
        max_price = max(prices)
        price_range = max_price - min_price

        if price_range == 0:
            return {}

        bin_size = price_range / self.price_bins
        bins = np.arange(min_price, max_price + bin_size, bin_size)

        # Agregar volumen por bin
        volume_by_bin = defaultdict(float)

        for price, volume in zip(prices, volumes):
            # Encontrar bin correspondiente
            bin_idx = min(int((price - min_price) / bin_size), self.price_bins - 1)
            bin_price = min_price + (bin_idx * bin_size) + (bin_size / 2)  # Centro del bin
            volume_by_bin[bin_price] += volume

        # Convertir a lista de VolumeNodes
        total_volume = sum(volume_by_bin.values())
        if total_volume == 0:
            return {}

        volume_nodes = [
            VolumeNode(
                price=price,
                volume=vol,
                percentage=(vol / total_volume) * 100
            )
            for price, vol in volume_by_bin.items()
        ]

        # Ordenar por volumen descendente
        volume_nodes.sort(key=lambda x: x.volume, reverse=True)

        # POC = nivel con máximo volumen
        poc = volume_nodes[0]

        # Value Area = niveles que suman 70% del volumen
        value_area_nodes = []
        accumulated_volume = 0.0
        target_volume = total_volume * (self.value_area_pct / 100)

        for node in volume_nodes:
            value_area_nodes.append(node)
            accumulated_volume += node.volume

            if accumulated_volume >= target_volume:
                break

        # Value Area High/Low
        value_area_prices = [node.price for node in value_area_nodes]
        vah = max(value_area_prices)  # Value Area High
        val = min(value_area_prices)  # Value Area Low

        profile_data = {
            'poc': poc.price,
            'poc_volume': poc.volume,
            'poc_percentage': poc.percentage,
            'value_area_high': vah,
            'value_area_low': val,
            'value_area_mid': (vah + val) / 2,
            'total_volume': total_volume,
            'volume_nodes': volume_nodes[:10],  # Top 10 nodes
            'price_range': (min_price, max_price)
        }

        self.volume_profiles[pair] = profile_data

        logger.debug(f"📊 Volume Profile calculado para {pair}: POC=${poc.price:.2f} ({poc.percentage:.1f}%)")

        return profile_data

    def is_near_poc(self, pair: str, current_price: float) -> Tuple[bool, Optional[float]]:
        """
        Verifica si precio está cerca del POC

        Args:
            pair: Par de trading
            current_price: Precio actual

        Returns:
            (is_near, distance_pct)
        """
        if pair not in self.volume_profiles:
            return False, None

        profile = self.volume_profiles[pair]
        poc = profile['poc']

        distance_pct = abs((current_price - poc) / poc) * 100

        is_near = distance_pct <= self.poc_proximity_pct

        return is_near, distance_pct

    def is_in_value_area(self, pair: str, current_price: float) -> bool:
        """
        Verifica si precio está dentro del Value Area

        Args:
            pair: Par de trading
            current_price: Precio actual

        Returns:
            True si está en value area
        """
        if pair not in self.volume_profiles:
            return False

        profile = self.volume_profiles[pair]
        vah = profile['value_area_high']
        val = profile['value_area_low']

        return val <= current_price <= vah

    def adjust_signal_confidence(
        self,
        pair: str,
        signal_side: str,
        current_price: float,
        base_confidence: float
    ) -> float:
        """
        Ajusta confianza considerando volume profile

        Args:
            pair: Par de trading
            signal_side: 'BUY' o 'SELL'
            current_price: Precio actual
            base_confidence: Confianza base (0-100)

        Returns:
            Confianza ajustada (0-100)
        """
        if not self.enabled or pair not in self.volume_profiles:
            return base_confidence

        profile = self.volume_profiles[pair]
        poc = profile['poc']

        # Boost si cerca del POC
        is_near_poc, poc_distance = self.is_near_poc(pair, current_price)

        if is_near_poc:
            # POC actúa como zona magnética + soporte/resistencia
            # Boost señales de rebote en POC
            if (signal_side == 'BUY' and current_price <= poc) or \
               (signal_side == 'SELL' and current_price >= poc):
                boost = self.boost_factor_poc
                adjusted = base_confidence * boost
                logger.info(f"📍 POC boost para {pair}: {base_confidence:.1f}% → {adjusted:.1f}% (precio cerca de POC)")
                return min(adjusted, 100.0)

        # Boost si en value area
        in_value_area = self.is_in_value_area(pair, current_price)

        if in_value_area:
            boost = self.boost_factor_value_area
            adjusted = base_confidence * boost
            logger.info(f"📦 Value Area boost para {pair}: {base_confidence:.1f}% → {adjusted:.1f}%")
            return min(adjusted, 100.0)

        return base_confidence

    def get_support_resistance_levels(self, pair: str) -> List[Tuple[float, str]]:
        """
        Obtiene niveles de soporte/resistencia del volume profile

        Args:
            pair: Par de trading

        Returns:
            Lista de (price, type) donde type es 'POC', 'VAH', 'VAL'
        """
        if pair not in self.volume_profiles:
            return []

        profile = self.volume_profiles[pair]

        levels = [
            (profile['poc'], 'POC'),
            (profile['value_area_high'], 'VAH'),
            (profile['value_area_low'], 'VAL')
        ]

        return levels

    def get_statistics(self) -> Dict:
        """
        Estadísticas de volume profile

        Returns:
            Dict con métricas
        """
        return {
            'enabled': self.enabled,
            'pairs_tracked': len(self.volume_profiles),
            'lookback_periods': self.lookback_periods,
            'price_bins': self.price_bins,
            'value_area_pct': self.value_area_pct,
            'poc_proximity_pct': self.poc_proximity_pct
        }


# Parámetros optimizables para config.py
VOLUME_PROFILE_PARAMS = {
    'VOLUME_PROFILE_ENABLED': True,
    'VOLUME_PROFILE_LOOKBACK': 100,  # 50-200 (optimizable)
    'VOLUME_PROFILE_BINS': 50,  # 30-100 (optimizable)
    'VOLUME_PROFILE_VALUE_AREA': 70.0,  # 65-75% (optimizable)
    'POC_PROXIMITY_PCT': 1.0,  # 0.5-2% (optimizable)
    'POC_BOOST_FACTOR': 1.3,  # 1.2-1.5x (optimizable)
    'VALUE_AREA_BOOST_FACTOR': 1.15,  # 1.1-1.3x (optimizable)
}
