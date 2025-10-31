"""
Order Book Analyzer - Analiza profundidad de mercado en tiempo real
Detecta paredes de compra/venta, imbalances, y soportes/resistencias
"""
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)


class OrderBookAnalyzer:
    """
    Analiza el order book para detectar:
    - Profundidad de mercado (bids vs asks)
    - Paredes (large orders)
    - Imbalances (presión de compra/venta)
    - Soportes/resistencias basados en liquidez
    """

    def __init__(self, depth_limit: int = 100):
        """
        Args:
            depth_limit: Cantidad de niveles del order book a analizar
        """
        self.depth_limit = depth_limit
        self.cache = {}  # Cache para evitar spam de API
        self.cache_duration = timedelta(seconds=10)  # Cache 10 segundos

        logger.info(f"📚 Order Book Analyzer inicializado (depth={depth_limit})")

    def analyze(self, exchange, pair: str, current_price: float) -> Dict:
        """
        Analiza order book completo para un par

        Args:
            exchange: Instancia de ccxt exchange
            pair: Par de trading (ej: 'BTC/USDT')
            current_price: Precio actual del activo

        Returns:
            Dict con análisis completo del order book
        """
        # Check cache
        cache_key = f"{pair}_{int(datetime.now().timestamp() / 10)}"  # Cache de 10s
        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            # Fetch order book
            order_book = exchange.fetch_order_book(pair, limit=self.depth_limit)

            bids = order_book['bids']  # [[price, amount], ...]
            asks = order_book['asks']

            if not bids or not asks:
                logger.warning(f"Order book vacío para {pair}")
                return self._empty_analysis()

            # Análisis completo
            analysis = {
                'timestamp': datetime.now().isoformat(),
                'pair': pair,
                'current_price': current_price,

                # Profundidad básica
                'bid_depth': self._calculate_depth(bids, current_price),
                'ask_depth': self._calculate_depth(asks, current_price),

                # Imbalance (presión)
                'imbalance': self._calculate_imbalance(bids, asks, current_price),

                # Paredes (large orders)
                'bid_walls': self._detect_walls(bids, 'bid', current_price),
                'ask_walls': self._detect_walls(asks, 'ask', current_price),

                # Soportes/Resistencias
                'support_levels': self._find_support_levels(bids, current_price),
                'resistance_levels': self._find_resistance_levels(asks, current_price),

                # Spread
                'spread': self._calculate_spread(bids, asks),

                # Liquidez total
                'total_bid_volume': sum(bid[1] for bid in bids),
                'total_ask_volume': sum(ask[1] for ask in asks),
            }

            # Clasificación de presión del mercado
            analysis['market_pressure'] = self._classify_pressure(analysis['imbalance'])

            # Confidence adjustment basado en order book
            analysis['confidence_adjustment'] = self._calculate_confidence_adjustment(analysis)

            # Cache result
            self.cache[cache_key] = analysis

            # Limpiar cache viejo
            self._clean_cache()

            logger.debug(
                f"Order Book {pair}: "
                f"Imbalance={analysis['imbalance']:.2f}, "
                f"Pressure={analysis['market_pressure']}, "
                f"Bid Walls={len(analysis['bid_walls'])}, "
                f"Ask Walls={len(analysis['ask_walls'])}"
            )

            return analysis

        except Exception as e:
            logger.error(f"Error analizando order book para {pair}: {e}")
            return self._empty_analysis()

    def _calculate_depth(self, orders: List[List[float]], current_price: float,
                         range_pct: float = 0.01) -> Dict:
        """
        Calcula profundidad de mercado dentro de un rango del precio actual

        Args:
            orders: Lista de [price, amount]
            current_price: Precio actual
            range_pct: Rango como % del precio (default 1%)

        Returns:
            Dict con métricas de profundidad
        """
        if not orders:
            return {'volume': 0, 'value': 0, 'orders_count': 0}

        # Filtrar órdenes dentro del rango
        range_min = current_price * (1 - range_pct)
        range_max = current_price * (1 + range_pct)

        orders_in_range = [
            order for order in orders
            if range_min <= order[0] <= range_max
        ]

        if not orders_in_range:
            return {'volume': 0, 'value': 0, 'orders_count': 0}

        total_volume = sum(order[1] for order in orders_in_range)
        total_value = sum(order[0] * order[1] for order in orders_in_range)

        return {
            'volume': total_volume,
            'value': total_value,
            'orders_count': len(orders_in_range),
            'avg_price': total_value / total_volume if total_volume > 0 else 0
        }

    def _calculate_imbalance(self, bids: List[List[float]], asks: List[List[float]],
                             current_price: float) -> float:
        """
        Calcula imbalance del order book (presión de compra vs venta)

        Returns:
            Float entre -1 (presión de venta) y +1 (presión de compra)
        """
        # Analizar órdenes dentro del 1% del precio actual
        bid_depth = self._calculate_depth(bids, current_price, 0.01)
        ask_depth = self._calculate_depth(asks, current_price, 0.01)

        bid_volume = bid_depth['volume']
        ask_volume = ask_depth['volume']

        total_volume = bid_volume + ask_volume

        if total_volume == 0:
            return 0.0

        # Imbalance: +1 = mucha presión de compra, -1 = mucha presión de venta
        imbalance = (bid_volume - ask_volume) / total_volume

        return imbalance

    def _detect_walls(self, orders: List[List[float]], side: str,
                      current_price: float, threshold_multiplier: float = 3.0) -> List[Dict]:
        """
        Detecta "paredes" (large orders) en el order book

        Args:
            orders: Lista de [price, amount]
            side: 'bid' o 'ask'
            current_price: Precio actual
            threshold_multiplier: Múltiplo del promedio para considerar "pared"

        Returns:
            Lista de paredes detectadas
        """
        if not orders or len(orders) < 10:
            return []

        # Calcular volumen promedio
        volumes = [order[1] for order in orders[:50]]  # Top 50 niveles
        avg_volume = np.mean(volumes)
        std_volume = np.std(volumes)

        # Threshold: promedio + 3 desviaciones estándar
        threshold = avg_volume + (threshold_multiplier * std_volume)

        # Detectar paredes
        walls = []
        for price, amount in orders[:50]:  # Analizar top 50 niveles
            if amount >= threshold:
                distance_pct = abs((price - current_price) / current_price) * 100

                walls.append({
                    'price': price,
                    'amount': amount,
                    'value': price * amount,
                    'distance_pct': distance_pct,
                    'side': side,
                    'strength': amount / avg_volume  # Cuántas veces el promedio
                })

        # Ordenar por cercanía al precio actual
        walls.sort(key=lambda x: x['distance_pct'])

        return walls[:5]  # Top 5 paredes más cercanas

    def _find_support_levels(self, bids: List[List[float]],
                             current_price: float) -> List[Dict]:
        """
        Identifica niveles de soporte basados en concentración de bids

        Args:
            bids: Lista de bids [price, amount]
            current_price: Precio actual

        Returns:
            Lista de niveles de soporte
        """
        if not bids:
            return []

        # Agrupar bids por rangos de precio
        support_levels = []

        # Analizar bids hasta 5% abajo del precio actual
        min_price = current_price * 0.95
        relevant_bids = [bid for bid in bids if bid[0] >= min_price]

        if not relevant_bids:
            return []

        # Crear buckets de 0.2% para agrupar liquidez
        bucket_size = current_price * 0.002  # 0.2%
        buckets = {}

        for price, amount in relevant_bids:
            bucket = int(price / bucket_size) * bucket_size
            if bucket not in buckets:
                buckets[bucket] = 0
            buckets[bucket] += amount

        # Encontrar buckets con más liquidez
        sorted_buckets = sorted(buckets.items(), key=lambda x: x[1], reverse=True)

        for price, volume in sorted_buckets[:3]:  # Top 3 soportes
            distance_pct = abs((price - current_price) / current_price) * 100

            support_levels.append({
                'price': price,
                'volume': volume,
                'distance_pct': distance_pct,
                'strength': 'STRONG' if volume > np.mean(list(buckets.values())) * 2 else 'MEDIUM'
            })

        return support_levels

    def _find_resistance_levels(self, asks: List[List[float]],
                                current_price: float) -> List[Dict]:
        """
        Identifica niveles de resistencia basados en concentración de asks

        Args:
            asks: Lista de asks [price, amount]
            current_price: Precio actual

        Returns:
            Lista de niveles de resistencia
        """
        if not asks:
            return []

        # Analizar asks hasta 5% arriba del precio actual
        max_price = current_price * 1.05
        relevant_asks = [ask for ask in asks if ask[0] <= max_price]

        if not relevant_asks:
            return []

        # Crear buckets de 0.2%
        bucket_size = current_price * 0.002
        buckets = {}

        for price, amount in relevant_asks:
            bucket = int(price / bucket_size) * bucket_size
            if bucket not in buckets:
                buckets[bucket] = 0
            buckets[bucket] += amount

        # Encontrar buckets con más liquidez
        sorted_buckets = sorted(buckets.items(), key=lambda x: x[1], reverse=True)

        resistance_levels = []
        for price, volume in sorted_buckets[:3]:  # Top 3 resistencias
            distance_pct = abs((price - current_price) / current_price) * 100

            resistance_levels.append({
                'price': price,
                'volume': volume,
                'distance_pct': distance_pct,
                'strength': 'STRONG' if volume > np.mean(list(buckets.values())) * 2 else 'MEDIUM'
            })

        return resistance_levels

    def _calculate_spread(self, bids: List[List[float]],
                          asks: List[List[float]]) -> Dict:
        """
        Calcula spread entre best bid y best ask

        Returns:
            Dict con spread absoluto y porcentual
        """
        if not bids or not asks:
            return {'absolute': 0, 'percentage': 0}

        best_bid = bids[0][0]
        best_ask = asks[0][0]

        spread_abs = best_ask - best_bid
        spread_pct = (spread_abs / best_bid) * 100 if best_bid > 0 else 0

        return {
            'absolute': spread_abs,
            'percentage': spread_pct,
            'best_bid': best_bid,
            'best_ask': best_ask
        }

    def _classify_pressure(self, imbalance: float) -> str:
        """
        Clasifica presión del mercado basado en imbalance

        Args:
            imbalance: Float entre -1 y +1

        Returns:
            'STRONG_BUY', 'BUY', 'NEUTRAL', 'SELL', 'STRONG_SELL'
        """
        if imbalance >= 0.3:
            return 'STRONG_BUY'
        elif imbalance >= 0.1:
            return 'BUY'
        elif imbalance <= -0.3:
            return 'STRONG_SELL'
        elif imbalance <= -0.1:
            return 'SELL'
        else:
            return 'NEUTRAL'

    def _calculate_confidence_adjustment(self, analysis: Dict) -> float:
        """
        Calcula ajuste de confidence basado en order book

        Returns:
            Float entre -15 y +15 (% de ajuste)
        """
        adjustment = 0.0

        # Factor 1: Imbalance (máx ±10%)
        imbalance = analysis['imbalance']
        adjustment += imbalance * 10  # -10 a +10

        # Factor 2: Paredes (±5%)
        bid_walls = len(analysis['bid_walls'])
        ask_walls = len(analysis['ask_walls'])

        if bid_walls > ask_walls:
            adjustment += min(5, (bid_walls - ask_walls) * 2)
        elif ask_walls > bid_walls:
            adjustment -= min(5, (ask_walls - bid_walls) * 2)

        # Limitar ajuste entre -15 y +15
        adjustment = max(-15, min(15, adjustment))

        return round(adjustment, 1)

    def _empty_analysis(self) -> Dict:
        """Retorna análisis vacío cuando hay error"""
        return {
            'timestamp': datetime.now().isoformat(),
            'pair': '',
            'current_price': 0,
            'bid_depth': {'volume': 0, 'value': 0, 'orders_count': 0},
            'ask_depth': {'volume': 0, 'value': 0, 'orders_count': 0},
            'imbalance': 0.0,
            'bid_walls': [],
            'ask_walls': [],
            'support_levels': [],
            'resistance_levels': [],
            'spread': {'absolute': 0, 'percentage': 0},
            'total_bid_volume': 0,
            'total_ask_volume': 0,
            'market_pressure': 'NEUTRAL',
            'confidence_adjustment': 0.0
        }

    def _clean_cache(self):
        """Limpia cache viejo (más de 1 minuto)"""
        current_time = int(datetime.now().timestamp() / 10)
        keys_to_delete = [
            key for key in self.cache.keys()
            if int(key.split('_')[-1]) < current_time - 6  # 60 segundos
        ]
        for key in keys_to_delete:
            del self.cache[key]

    def get_orderbook_features(self, analysis: Dict) -> Dict:
        """
        Convierte análisis de order book en features para ML

        Returns:
            Dict con 6 features normalizadas para el modelo ML
        """
        return {
            'ob_imbalance': analysis['imbalance'],  # -1 a +1
            'ob_bid_walls_count': min(len(analysis['bid_walls']) / 5, 1.0),  # 0 a 1
            'ob_ask_walls_count': min(len(analysis['ask_walls']) / 5, 1.0),  # 0 a 1
            'ob_spread_pct': min(analysis['spread']['percentage'] / 0.5, 1.0),  # Normalizado
            'ob_support_strength': 1.0 if analysis['support_levels'] and analysis['support_levels'][0].get('strength') == 'STRONG' else 0.5,
            'ob_resistance_strength': 1.0 if analysis['resistance_levels'] and analysis['resistance_levels'][0].get('strength') == 'STRONG' else 0.5,
        }
