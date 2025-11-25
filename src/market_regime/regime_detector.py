"""
Market Regime Detector - Clasifica mercado en BULL / BEAR / SIDEWAYS
Analiza tendencias, volatilidad, y momentum para determinar régimen actual
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class RegimeDetector:
    """
    Detecta el régimen de mercado actual:
    - BULL: Tendencia alcista fuerte
    - BEAR: Tendencia bajista fuerte
    - SIDEWAYS: Mercado lateral/consolidación

    Usa múltiples indicadores:
    - Tendencia de precio (30/60/90 días)
    - Moving averages (MA20, MA50, MA200)
    - RSI promedio
    - ATR (volatilidad)
    - Volume trends
    """

    def __init__(self):
        """Initialize regime detector"""
        self.cache = {}  # Cache para evitar recálculos
        self.cache_duration = timedelta(minutes=15)  # Cache de 15 minutos

        logger.info("🎯 Market Regime Detector inicializado")

    def detect(self, exchange, pair: str, current_price: float) -> Dict:
        """
        Detecta régimen de mercado para un par

        Args:
            exchange: Instancia de ccxt exchange
            pair: Par de trading
            current_price: Precio actual

        Returns:
            Dict con régimen y métricas
        """
        # Validar current_price
        if current_price <= 0:
            logger.warning(f"Precio inválido para {pair}: {current_price}")
            return self._default_regime()

        # Check cache
        cache_key = f"{pair}_{datetime.now().strftime('%Y%m%d%H%M')[:11]}"  # Cache 15 min
        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            # Fetch historical data (necesitamos 200 días para MA200)
            df_1d = self._fetch_daily_data(exchange, pair, limit=200)

            if df_1d is None or len(df_1d) < 90:
                logger.warning(f"Datos insuficientes para detectar régimen en {pair}")
                return self._default_regime()

            # Calcular indicadores
            indicators = self._calculate_indicators(df_1d, current_price)

            # Detectar régimen basado en múltiples factores
            regime = self._classify_regime(indicators)

            # Calcular confianza del régimen
            confidence = self._calculate_regime_confidence(indicators)

            # Recomendaciones de trading basadas en régimen
            recommendations = self._get_trading_recommendations(regime)

            result = {
                'timestamp': datetime.now().isoformat(),
                'pair': pair,
                'regime': regime,  # 'BULL', 'BEAR', 'SIDEWAYS'
                'confidence': confidence,  # 0-100
                'indicators': indicators,
                'recommendations': recommendations,
                'regime_strength': self._calculate_regime_strength(indicators, regime)
            }

            # Cache result
            self.cache[cache_key] = result

            # Limpiar cache viejo
            self._clean_cache()

            logger.info(
                f"Market Regime {pair}: {regime} "
                f"(confidence={confidence:.0f}%, strength={result['regime_strength']})"
            )

            return result

        except Exception as e:
            logger.error(f"Error detectando régimen para {pair}: {e}")
            return self._default_regime()

    def _fetch_daily_data(self, exchange, pair: str, limit: int = 200) -> Optional[pd.DataFrame]:
        """Fetch daily OHLCV data"""
        try:
            ohlcv = exchange.fetch_ohlcv(pair, timeframe='1d', limit=limit)

            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')

            return df

        except Exception as e:
            logger.error(f"Error fetching daily data para {pair}: {e}")
            return None

    def _calculate_indicators(self, df: pd.DataFrame, current_price: float) -> Dict:
        """
        Calcula todos los indicadores necesarios para detectar régimen

        Args:
            df: DataFrame con datos OHLCV diarios
            current_price: Precio actual

        Returns:
            Dict con indicadores calculados
        """
        indicators = {}

        # === TENDENCIAS DE PRECIO ===
        if len(df) >= 30:
            price_30d_ago = df['close'].iloc[-30]
            indicators['trend_30d'] = ((current_price - price_30d_ago) / price_30d_ago) * 100

        if len(df) >= 60:
            price_60d_ago = df['close'].iloc[-60]
            indicators['trend_60d'] = ((current_price - price_60d_ago) / price_60d_ago) * 100

        if len(df) >= 90:
            price_90d_ago = df['close'].iloc[-90]
            indicators['trend_90d'] = ((current_price - price_90d_ago) / price_90d_ago) * 100

        # === MOVING AVERAGES ===
        if len(df) >= 20:
            ma20 = df['close'].rolling(window=20).mean().iloc[-1]
            indicators['ma20'] = ma20
            indicators['price_vs_ma20'] = ((current_price - ma20) / ma20) * 100

        if len(df) >= 50:
            ma50 = df['close'].rolling(window=50).mean().iloc[-1]
            indicators['ma50'] = ma50
            indicators['price_vs_ma50'] = ((current_price - ma50) / ma50) * 100

        if len(df) >= 200:
            ma200 = df['close'].rolling(window=200).mean().iloc[-1]
            indicators['ma200'] = ma200
            indicators['price_vs_ma200'] = ((current_price - ma200) / ma200) * 100

        # === RSI PROMEDIO (30 días) ===
        if len(df) >= 30:
            rsi = self._calculate_rsi(df['close'], period=14)
            indicators['rsi_avg_30d'] = rsi.iloc[-30:].mean()
            indicators['rsi_current'] = rsi.iloc[-1]

        # === VOLATILIDAD (ATR) ===
        if len(df) >= 14:
            atr = self._calculate_atr(df, period=14)
            indicators['atr'] = atr.iloc[-1]
            indicators['atr_pct'] = (atr.iloc[-1] / current_price) * 100

            # ATR promedio últimos 30 días
            if len(df) >= 30:
                atr_avg = atr.iloc[-30:].mean()
                indicators['atr_avg_30d'] = atr_avg
                indicators['volatility_regime'] = 'HIGH' if atr.iloc[-1] > atr_avg * 1.5 else 'NORMAL'

        # === VOLUMEN ===
        if len(df) >= 30:
            volume_avg = df['volume'].iloc[-30:].mean()
            volume_recent = df['volume'].iloc[-7:].mean()  # Última semana
            indicators['volume_trend'] = ((volume_recent - volume_avg) / volume_avg) * 100

        # === HIGHER HIGHS / LOWER LOWS ===
        if len(df) >= 60:
            recent_highs = df['high'].iloc[-30:]
            previous_highs = df['high'].iloc[-60:-30]

            indicators['higher_highs'] = recent_highs.max() > previous_highs.max()
            indicators['lower_lows'] = df['low'].iloc[-30:].min() < df['low'].iloc[-60:-30].min()

        return indicators

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calcula RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calcula Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())

        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)

        atr = true_range.rolling(window=period).mean()

        return atr

    def _classify_regime(self, indicators: Dict) -> str:
        """
        Clasifica régimen basado en múltiples indicadores

        Returns:
            'BULL', 'BEAR', o 'SIDEWAYS'
        """
        score = 0  # Score positivo = BULL, negativo = BEAR, neutro = SIDEWAYS

        # Factor 1: Tendencias de precio (peso: 30%)
        if 'trend_30d' in indicators:
            if indicators['trend_30d'] > 10:
                score += 3
            elif indicators['trend_30d'] < -10:
                score -= 3

        if 'trend_60d' in indicators:
            if indicators['trend_60d'] > 15:
                score += 2
            elif indicators['trend_60d'] < -15:
                score -= 2

        if 'trend_90d' in indicators:
            if indicators['trend_90d'] > 20:
                score += 2
            elif indicators['trend_90d'] < -20:
                score -= 2

        # Factor 2: Precio vs Moving Averages (peso: 30%)
        if 'price_vs_ma20' in indicators:
            if indicators['price_vs_ma20'] > 3:
                score += 1
            elif indicators['price_vs_ma20'] < -3:
                score -= 1

        if 'price_vs_ma50' in indicators:
            if indicators['price_vs_ma50'] > 5:
                score += 2
            elif indicators['price_vs_ma50'] < -5:
                score -= 2

        if 'price_vs_ma200' in indicators:
            if indicators['price_vs_ma200'] > 0:
                score += 2
            elif indicators['price_vs_ma200'] < 0:
                score -= 2

        # Factor 3: RSI promedio (peso: 20%)
        if 'rsi_avg_30d' in indicators:
            rsi_avg = indicators['rsi_avg_30d']
            if rsi_avg > 60:
                score += 2
            elif rsi_avg < 40:
                score -= 2

        # Factor 4: Higher Highs / Lower Lows (peso: 20%)
        if 'higher_highs' in indicators and 'lower_lows' in indicators:
            if indicators['higher_highs'] and not indicators['lower_lows']:
                score += 2  # Uptrend
            elif indicators['lower_lows'] and not indicators['higher_highs']:
                score -= 2  # Downtrend

        # Clasificación final
        if score >= 5:
            return 'BULL'
        elif score <= -5:
            return 'BEAR'
        else:
            return 'SIDEWAYS'

    def _calculate_regime_confidence(self, indicators: Dict) -> float:
        """
        Calcula confidence del régimen detectado (0-100)

        Mayor confidence = indicadores más alineados
        """
        alignments = []

        # Check alignment de tendencias
        trends = []
        if 'trend_30d' in indicators:
            trends.append(1 if indicators['trend_30d'] > 0 else -1)
        if 'trend_60d' in indicators:
            trends.append(1 if indicators['trend_60d'] > 0 else -1)
        if 'trend_90d' in indicators:
            trends.append(1 if indicators['trend_90d'] > 0 else -1)

        if trends:
            # Si todas las tendencias apuntan en misma dirección → alta confidence
            alignment = abs(sum(trends)) / len(trends)
            alignments.append(alignment)

        # Check alignment de MAs
        ma_positions = []
        if 'price_vs_ma20' in indicators:
            ma_positions.append(1 if indicators['price_vs_ma20'] > 0 else -1)
        if 'price_vs_ma50' in indicators:
            ma_positions.append(1 if indicators['price_vs_ma50'] > 0 else -1)
        if 'price_vs_ma200' in indicators:
            ma_positions.append(1 if indicators['price_vs_ma200'] > 0 else -1)

        if ma_positions:
            alignment = abs(sum(ma_positions)) / len(ma_positions)
            alignments.append(alignment)

        # Confidence final
        if alignments:
            avg_alignment = sum(alignments) / len(alignments)
            confidence = avg_alignment * 100
        else:
            confidence = 50.0  # Default si no hay suficientes datos

        return round(confidence, 1)

    def _calculate_regime_strength(self, indicators: Dict, regime: str) -> str:
        """
        Calcula fuerza del régimen (WEAK, MODERATE, STRONG)

        Args:
            indicators: Indicadores calculados
            regime: Régimen detectado

        Returns:
            'WEAK', 'MODERATE', 'STRONG'
        """
        if regime == 'SIDEWAYS':
            return 'MODERATE'  # Sideways no tiene "strength"

        strength_score = 0

        # Factor 1: Magnitud de tendencias
        if 'trend_30d' in indicators:
            trend = abs(indicators['trend_30d'])
            if trend > 20:
                strength_score += 2
            elif trend > 10:
                strength_score += 1

        # Factor 2: Distancia de MAs
        if 'price_vs_ma50' in indicators:
            distance = abs(indicators['price_vs_ma50'])
            if distance > 10:
                strength_score += 2
            elif distance > 5:
                strength_score += 1

        # Factor 3: RSI extremo
        if 'rsi_avg_30d' in indicators:
            rsi = indicators['rsi_avg_30d']
            if rsi > 65 or rsi < 35:
                strength_score += 1

        # Clasificación
        if strength_score >= 4:
            return 'STRONG'
        elif strength_score >= 2:
            return 'MODERATE'
        else:
            return 'WEAK'

    def _get_trading_recommendations(self, regime: str) -> Dict:
        """
        Genera recomendaciones de trading basadas en régimen

        Args:
            regime: 'BULL', 'BEAR', 'SIDEWAYS'

        Returns:
            Dict con recomendaciones
        """
        recommendations = {
            'BULL': {
                'strategy': 'AGGRESSIVE_LONG',
                'position_size': 'LARGE',
                'signal_bias': 'BUY',
                'stop_loss': 'WIDE',  # SL más amplio en bull markets
                'take_profit': 'EXTENDED',  # TPs más ambiciosos
                'description': 'Mercado alcista - Favorecer señales de compra, posiciones más grandes'
            },
            'BEAR': {
                'strategy': 'DEFENSIVE',
                'position_size': 'SMALL',
                'signal_bias': 'SELL',
                'stop_loss': 'TIGHT',  # SL más ajustado en bear markets
                'take_profit': 'CONSERVATIVE',
                'description': 'Mercado bajista - Reducir exposición, solo señales muy fuertes'
            },
            'SIDEWAYS': {
                'strategy': 'RANGE_TRADING',
                'position_size': 'MEDIUM',
                'signal_bias': 'NEUTRAL',
                'stop_loss': 'MODERATE',
                'take_profit': 'MODERATE',
                'description': 'Mercado lateral - Trading de rango, tomar ganancias rápido'
            }
        }

        return recommendations.get(regime, recommendations['SIDEWAYS'])

    def _default_regime(self) -> Dict:
        """Retorna régimen por defecto cuando no hay datos suficientes"""
        return {
            'timestamp': datetime.now().isoformat(),
            'pair': '',
            'regime': 'SIDEWAYS',  # Más conservador
            'confidence': 50.0,
            'indicators': {},
            'recommendations': self._get_trading_recommendations('SIDEWAYS'),
            'regime_strength': 'MODERATE'
        }

    def _clean_cache(self):
        """Limpia cache viejo (más de 30 minutos)"""
        now = datetime.now()
        keys_to_delete = []

        for key in self.cache.keys():
            # Extract timestamp from cache key
            try:
                cache_time_str = key.split('_')[-1]
                cache_time = datetime.strptime(cache_time_str, '%Y%m%d%H%M')
                if now - cache_time > timedelta(minutes=30):
                    keys_to_delete.append(key)
            except:
                keys_to_delete.append(key)  # Delete malformed keys

        for key in keys_to_delete:
            del self.cache[key]

    def get_regime_features(self, regime_data: Dict) -> Dict:
        """
        Convierte análisis de régimen en features para ML

        Returns:
            Dict con 4 features normalizadas
        """
        regime = regime_data['regime']
        confidence = regime_data['confidence'] / 100  # Normalizar a 0-1

        return {
            'regime_is_bull': 1.0 if regime == 'BULL' else 0.0,
            'regime_is_bear': 1.0 if regime == 'BEAR' else 0.0,
            'regime_is_sideways': 1.0 if regime == 'SIDEWAYS' else 0.0,
            'regime_confidence': confidence,
        }
