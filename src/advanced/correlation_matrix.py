"""
Correlation Matrix - Análisis de correlación entre pares

Previene overexposure al trading:
- Detecta pares altamente correlacionados (>0.7)
- Evita abrir múltiples posiciones en pares correlacionados
- Diversifica el portfolio automáticamente
- Reduce riesgo sistémico de drawdown

Ejemplo: Si BTC y ETH están 85% correlacionados, no abrir ambos al mismo tiempo
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import deque
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class CorrelationMatrix:
    """
    Análisis de correlación entre pares de trading

    Reduce riesgo de overexposure y mejora diversificación
    """

    def __init__(self, config):
        self.config = config

        # Parámetros optimizables
        self.enabled = getattr(config, 'CORRELATION_ANALYSIS_ENABLED', True)
        self.high_correlation_threshold = getattr(config, 'HIGH_CORRELATION_THRESHOLD', 0.7)  # 0.6-0.85
        self.lookback_periods = getattr(config, 'CORRELATION_LOOKBACK_PERIODS', 100)  # 50-200 períodos
        self.min_data_points = getattr(config, 'CORRELATION_MIN_DATA_POINTS', 30)  # 20-50
        self.max_correlated_positions = getattr(config, 'MAX_CORRELATED_POSITIONS', 2)  # 1-3

        # Almacenamiento de precios históricos {pair: deque([prices])}
        self.price_history: Dict[str, deque] = {}

        # Matriz de correlación calculada
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.last_calculation: Optional[datetime] = None
        self.calculation_interval = timedelta(hours=1)  # Recalcular cada hora

        logger.info(f"CorrelationMatrix initialized: threshold={self.high_correlation_threshold}, lookback={self.lookback_periods}")

    def update_price(self, pair: str, price: float) -> None:
        """
        Actualiza precio histórico de un par

        Args:
            pair: Par de trading (e.g., 'BTC/USDT')
            price: Precio actual
        """
        if not self.enabled:
            return

        if pair not in self.price_history:
            self.price_history[pair] = deque(maxlen=self.lookback_periods)

        self.price_history[pair].append(price)

    def calculate_correlation_matrix(self) -> pd.DataFrame:
        """
        Calcula matriz de correlación entre todos los pares

        Returns:
            DataFrame con correlaciones (-1 a +1)
        """
        # Verificar si necesita recalcular
        if (self.correlation_matrix is not None and
            self.last_calculation and
            datetime.now() - self.last_calculation < self.calculation_interval):
            return self.correlation_matrix

        # Filtrar pares con suficiente data
        valid_pairs = {
            pair: prices
            for pair, prices in self.price_history.items()
            if len(prices) >= self.min_data_points
        }

        if len(valid_pairs) < 2:
            logger.warning("⚠️ Insuficientes pares para calcular correlación")
            return pd.DataFrame()

        # Crear DataFrame de precios
        price_data = {}
        for pair, prices in valid_pairs.items():
            price_data[pair] = list(prices)

        df = pd.DataFrame(price_data)

        # Calcular retornos logarítmicos (más apropiado que precios absolutos)
        returns = np.log(df / df.shift(1)).dropna()

        # Calcular matriz de correlación
        self.correlation_matrix = returns.corr()
        self.last_calculation = datetime.now()

        logger.info(f"📊 Correlation matrix recalculada para {len(valid_pairs)} pares")

        return self.correlation_matrix

    def get_correlation(self, pair1: str, pair2: str) -> Optional[float]:
        """
        Obtiene correlación entre dos pares

        Args:
            pair1: Primer par
            pair2: Segundo par

        Returns:
            Correlación (-1 a +1) o None si no hay data
        """
        if self.correlation_matrix is None or self.correlation_matrix.empty:
            self.calculate_correlation_matrix()

        if self.correlation_matrix is None or self.correlation_matrix.empty:
            return None

        if pair1 not in self.correlation_matrix.index or pair2 not in self.correlation_matrix.columns:
            return None

        return self.correlation_matrix.loc[pair1, pair2]

    def get_highly_correlated_pairs(self, pair: str, threshold: Optional[float] = None) -> List[Tuple[str, float]]:
        """
        Obtiene pares altamente correlacionados con un par dado

        Args:
            pair: Par de referencia
            threshold: Threshold de correlación (usa default si None)

        Returns:
            Lista de tuplas (pair, correlation) ordenadas por correlación
        """
        if threshold is None:
            threshold = self.high_correlation_threshold

        if self.correlation_matrix is None or self.correlation_matrix.empty:
            self.calculate_correlation_matrix()

        if self.correlation_matrix is None or pair not in self.correlation_matrix.index:
            return []

        # Obtener correlaciones del par
        correlations = self.correlation_matrix[pair].copy()

        # Excluir el par mismo (correlación = 1.0)
        correlations = correlations[correlations.index != pair]

        # Filtrar por threshold (valor absoluto para incluir correlación negativa)
        high_corr = correlations[abs(correlations) >= threshold]

        # Ordenar por correlación absoluta descendente
        high_corr = high_corr.reindex(high_corr.abs().sort_values(ascending=False).index)

        return list(zip(high_corr.index, high_corr.values))

    def can_open_position(self, pair: str, open_positions: List[str]) -> Tuple[bool, str]:
        """
        Verifica si se puede abrir posición considerando correlación

        Args:
            pair: Par que se quiere tradear
            open_positions: Lista de pares con posiciones abiertas

        Returns:
            (puede_abrir, razón)
        """
        if not self.enabled:
            return True, "Correlation analysis disabled"

        if not open_positions:
            return True, "No open positions"

        # Calcular matriz si es necesario
        if self.correlation_matrix is None or self.correlation_matrix.empty:
            self.calculate_correlation_matrix()

        if self.correlation_matrix is None or self.correlation_matrix.empty:
            logger.warning("⚠️ Correlation matrix vacía, permitiendo trade")
            return True, "Insufficient correlation data"

        # Contar posiciones correlacionadas
        correlated_positions = []
        for open_pair in open_positions:
            corr = self.get_correlation(pair, open_pair)

            if corr is not None and abs(corr) >= self.high_correlation_threshold:
                correlated_positions.append((open_pair, corr))

        # Verificar límite
        if len(correlated_positions) >= self.max_correlated_positions:
            corr_str = ", ".join([f"{p}({c:.2f})" for p, c in correlated_positions])
            return False, f"Demasiadas posiciones correlacionadas: {corr_str}"

        if correlated_positions:
            corr_str = ", ".join([f"{p}({c:.2f})" for p, c in correlated_positions])
            return True, f"Correlación aceptable con: {corr_str}"

        return True, "No hay correlación significativa"

    def get_diversification_score(self, open_positions: List[str]) -> float:
        """
        Calcula score de diversificación del portfolio (0-1)

        0 = muy correlacionado (mal)
        1 = muy diversificado (bien)

        Args:
            open_positions: Lista de pares con posiciones abiertas

        Returns:
            Score de diversificación (0-1)
        """
        if len(open_positions) <= 1:
            return 1.0  # Un solo par = máxima diversificación relativa

        if self.correlation_matrix is None or self.correlation_matrix.empty:
            return 0.5  # Score neutral si no hay data

        # Calcular correlación promedio entre todas las posiciones abiertas
        correlations = []
        for i, pair1 in enumerate(open_positions):
            for pair2 in open_positions[i+1:]:
                corr = self.get_correlation(pair1, pair2)
                if corr is not None:
                    correlations.append(abs(corr))

        if not correlations:
            return 0.5

        avg_correlation = np.mean(correlations)

        # Convertir a score (inverso de correlación)
        # Correlación 0 = score 1.0 (perfecto)
        # Correlación 1 = score 0.0 (terrible)
        diversification_score = 1.0 - avg_correlation

        return diversification_score

    def get_optimal_next_pair(self, candidate_pairs: List[str], open_positions: List[str]) -> Optional[Tuple[str, float]]:
        """
        Sugiere el mejor próximo par para tradear (máxima diversificación)

        Args:
            candidate_pairs: Lista de pares candidatos
            open_positions: Pares con posiciones abiertas

        Returns:
            (mejor_par, diversification_score) o None
        """
        if not open_positions:
            return None  # Sin posiciones abiertas, cualquier par es válido

        if self.correlation_matrix is None or self.correlation_matrix.empty:
            return None

        best_pair = None
        best_score = -1.0

        for candidate in candidate_pairs:
            if candidate in open_positions:
                continue  # Ya tiene posición abierta

            # Calcular correlación promedio con posiciones abiertas
            correlations = []
            for open_pair in open_positions:
                corr = self.get_correlation(candidate, open_pair)
                if corr is not None:
                    correlations.append(abs(corr))

            if correlations:
                avg_corr = np.mean(correlations)
                # Score = inverso de correlación
                score = 1.0 - avg_corr

                if score > best_score:
                    best_score = score
                    best_pair = candidate

        if best_pair:
            return (best_pair, best_score)

        return None

    def get_statistics(self) -> Dict:
        """
        Estadísticas de correlación

        Returns:
            Dict con métricas
        """
        if self.correlation_matrix is None or self.correlation_matrix.empty:
            return {
                'enabled': self.enabled,
                'pairs_tracked': len(self.price_history),
                'correlation_matrix_size': 0,
                'last_calculation': None
            }

        # Calcular estadísticas de la matriz
        # Extraer solo el triángulo superior (sin diagonal)
        mask = np.triu(np.ones_like(self.correlation_matrix, dtype=bool), k=1)
        upper_triangle = self.correlation_matrix.where(mask)
        correlations = upper_triangle.values[mask]

        return {
            'enabled': self.enabled,
            'pairs_tracked': len(self.price_history),
            'correlation_matrix_size': len(self.correlation_matrix),
            'last_calculation': self.last_calculation.isoformat() if self.last_calculation else None,
            'avg_correlation': float(np.mean(correlations)),
            'max_correlation': float(np.max(correlations)),
            'min_correlation': float(np.min(correlations)),
            'high_correlation_threshold': self.high_correlation_threshold,
            'max_correlated_positions': self.max_correlated_positions
        }


# Parámetros optimizables para config.py
CORRELATION_PARAMS = {
    'CORRELATION_ANALYSIS_ENABLED': True,
    'HIGH_CORRELATION_THRESHOLD': 0.7,  # 0.6-0.85 (optimizable)
    'CORRELATION_LOOKBACK_PERIODS': 100,  # 50-200 (optimizable)
    'CORRELATION_MIN_DATA_POINTS': 30,  # 20-50 (optimizable)
    'MAX_CORRELATED_POSITIONS': 2,  # 1-3 (optimizable)
}
