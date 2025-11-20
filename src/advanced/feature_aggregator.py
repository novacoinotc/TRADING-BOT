"""
Feature Aggregator - Integrador central de todos los análisis avanzados

Este módulo orquesta TODOS los análisis avanzados y los combina en un sistema unificado:
1. Correlation Matrix
2. Liquidation Heatmap
3. Funding Rate
4. Volume Profile & POC
5. Pattern Recognition
6. Session-Based Trading
7. Order Flow Imbalance

Expone features enriquecidos para:
- ML (XGBoost) → Features adicionales
- RL Agent → Estados extendidos
- Signal confidence adjustment → Boosts/penalties inteligentes
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

# Imports de módulos avanzados
from src.advanced.correlation_matrix import CorrelationMatrix
from src.advanced.liquidation_heatmap import LiquidationHeatmap
from src.advanced.funding_rate_analyzer import FundingRateAnalyzer
from src.advanced.volume_profile import VolumeProfile
from src.advanced.pattern_recognition import PatternRecognition
from src.advanced.session_trading import SessionBasedTrading
from src.advanced.order_flow import OrderFlowImbalance

logger = logging.getLogger(__name__)


class FeatureAggregator:
    """
    Agregador central de features avanzados

    Orquesta todos los análisis y los expone de forma unificada
    """

    def __init__(self, config, exchange=None):
        self.config = config
        self.exchange = exchange

        # Inicializar todos los módulos avanzados
        self.correlation_matrix = CorrelationMatrix(config)
        self.liquidation_heatmap = LiquidationHeatmap(config)
        self.funding_rate_analyzer = FundingRateAnalyzer(config, exchange)
        self.volume_profile = VolumeProfile(config)
        self.pattern_recognition = PatternRecognition(config)
        self.session_trading = SessionBasedTrading(config)
        self.order_flow = OrderFlowImbalance(config)

        logger.info("🚀 FeatureAggregator initialized: ALL advanced modules loaded")

    def enrich_signal(
        self,
        pair: str,
        signal: Dict,
        current_price: float,
        ohlc_data: Optional[Dict] = None,
        orderbook: Optional[Dict] = None,
        open_positions: Optional[List[str]] = None
    ) -> Dict:
        """
        Enriquece una señal de trading con TODOS los análisis avanzados

        Args:
            pair: Par de trading
            signal: Señal original con 'side', 'score', 'confidence', etc.
            current_price: Precio actual
            ohlc_data: Datos OHLCV opcionales
            orderbook: Order book opcional
            open_positions: Posiciones abiertas

        Returns:
            Señal enriquecida con features adicionales y confidence ajustada
        """
        enriched_signal = signal.copy()
        base_confidence = signal.get('confidence', 50.0)
        signal_side = signal.get('side', 'BUY')

        # === 1. CORRELATION ANALYSIS ===
        if open_positions:
            can_open, corr_reason = self.correlation_matrix.can_open_position(pair, open_positions)
            enriched_signal['correlation_check'] = {
                'can_open': can_open,
                'reason': corr_reason,
                'diversification_score': self.correlation_matrix.get_diversification_score(open_positions)
            }

            if not can_open:
                enriched_signal['blocked_by'] = 'CORRELATION'
                logger.warning(f"⛔ Señal bloqueada por correlación: {pair} - {corr_reason}")
                return enriched_signal

        # === 2. LIQUIDATION ANALYSIS ===
        liq_confidence = self.liquidation_heatmap.adjust_signal_confidence(
            pair, signal_side, current_price, base_confidence
        )
        if liq_confidence != base_confidence:
            enriched_signal['liquidation_boost'] = liq_confidence / base_confidence
            base_confidence = liq_confidence

        is_near_liq, liq_details = self.liquidation_heatmap.is_near_liquidation_zone(pair, current_price)
        enriched_signal['liquidation_analysis'] = {
            'is_near_zone': is_near_liq,
            'details': liq_details
        }

        # === 3. FUNDING RATE ANALYSIS ===
        funding_confidence = self.funding_rate_analyzer.adjust_signal_confidence(
            pair, signal_side, base_confidence
        )
        if funding_confidence != base_confidence:
            enriched_signal['funding_boost'] = funding_confidence / base_confidence
            base_confidence = funding_confidence

        sentiment, strength, funding_signal = self.funding_rate_analyzer.get_funding_sentiment(pair)
        enriched_signal['funding_analysis'] = {
            'sentiment': sentiment,
            'strength': strength,
            'signal': funding_signal
        }

        # === 4. VOLUME PROFILE ANALYSIS ===
        if ohlc_data is not None:
            # Calcular volume profile
            # ohlc_data puede ser pandas DataFrame, necesitamos los arrays
            try:
                closes = ohlc_data['close'].values if hasattr(ohlc_data, 'values') else ohlc_data.get('close', [])
                volumes = ohlc_data['volume'].values if hasattr(ohlc_data, 'values') else ohlc_data.get('volume', [])

                self.volume_profile.calculate_volume_profile(pair, closes, volumes)

                vp_confidence = self.volume_profile.adjust_signal_confidence(
                    pair, signal_side, current_price, base_confidence
                )
                if vp_confidence != base_confidence:
                    enriched_signal['volume_profile_boost'] = vp_confidence / base_confidence
                    base_confidence = vp_confidence

                is_near_poc, poc_distance = self.volume_profile.is_near_poc(pair, current_price)
                enriched_signal['volume_profile'] = {
                    'is_near_poc': is_near_poc,
                    'poc_distance_pct': poc_distance,
                    'in_value_area': self.volume_profile.is_in_value_area(pair, current_price)
                }
            except Exception as e:
                logger.debug(f"Error en volume profile analysis: {e}")

        # === 5. PATTERN RECOGNITION ===
        if ohlc_data is not None:
            detected_patterns = self.pattern_recognition.detect_all_patterns(ohlc_data)

            if detected_patterns:
                pattern_confidence = self.pattern_recognition.adjust_signal_confidence(
                    signal_side, base_confidence, detected_patterns
                )
                if pattern_confidence != base_confidence:
                    enriched_signal['pattern_boost'] = pattern_confidence / base_confidence
                    base_confidence = pattern_confidence

                enriched_signal['patterns_detected'] = [
                    {
                        'pattern': p['pattern'],
                        'type': p['type'],
                        'confidence': p['confidence'],
                        'signal': p['signal']
                    }
                    for p in detected_patterns
                ]

        # === 6. SESSION-BASED ADJUSTMENT ===
        session, session_multiplier = self.session_trading.get_current_session()
        enriched_signal['session'] = {
            'name': session,
            'multiplier': session_multiplier
        }

        # Ajustar position size por sesión
        if 'position_size_pct' in enriched_signal:
            original_size = enriched_signal['position_size_pct']
            adjusted_size = self.session_trading.adjust_position_size(original_size)
            if adjusted_size != original_size:
                enriched_signal['session_size_adjustment'] = adjusted_size / original_size
                enriched_signal['position_size_pct'] = adjusted_size

        # === 7. ORDER FLOW ANALYSIS ===
        if orderbook:
            flow_confidence = self.order_flow.adjust_signal_confidence(
                signal_side, base_confidence, orderbook
            )
            if flow_confidence != base_confidence:
                enriched_signal['order_flow_boost'] = flow_confidence / base_confidence
                base_confidence = flow_confidence

            bias, ratio, strength = self.order_flow.analyze_orderbook(orderbook)
            enriched_signal['order_flow'] = {
                'bias': bias,
                'bid_ask_ratio': ratio,
                'strength': strength
            }

        # === CONFIDENCE FINAL ===
        enriched_signal['original_confidence'] = signal.get('confidence', 50.0)
        enriched_signal['final_confidence'] = base_confidence

        # Calcular boost total
        total_boost = base_confidence / enriched_signal['original_confidence']
        enriched_signal['total_boost'] = total_boost

        if total_boost > 1.1:
            logger.info(f"🚀 Señal BOOSTED para {pair}: {enriched_signal['original_confidence']:.1f}% → {base_confidence:.1f}% (+{(total_boost-1)*100:.1f}%)")
        elif total_boost < 0.9:
            logger.warning(f"⚠️ Señal PENALIZADA para {pair}: {enriched_signal['original_confidence']:.1f}% → {base_confidence:.1f}% ({(total_boost-1)*100:.1f}%)")

        return enriched_signal

    def get_ml_features(
        self,
        pair: str,
        current_price: float,
        base_features: Dict,
        ohlc_data: Optional[Dict] = None
    ) -> Dict:
        """
        Genera features adicionales para ML (XGBoost)

        Args:
            pair: Par de trading
            current_price: Precio actual
            base_features: Features técnicos base
            ohlc_data: Datos OHLCV

        Returns:
            Dict con features adicionales
        """
        ml_features = base_features.copy()

        # Funding rate features
        funding_rate = self.funding_rate_analyzer.fetch_funding_rate(pair)
        if funding_rate is not None:
            ml_features['funding_rate'] = funding_rate
            ml_features['funding_is_extreme'] = abs(funding_rate) > 0.1
            ml_features['funding_is_positive'] = funding_rate > 0
        else:
            ml_features['funding_rate'] = 0.0

        # Funding sentiment (para logging)
        try:
            sentiment, strength, _ = self.funding_rate_analyzer.get_funding_sentiment(pair)
            # Validar tipo
            if not isinstance(sentiment, str):
                logger.warning(f"⚠️ FundingRateAnalyzer.get_funding_sentiment devolvió tipo incorrecto para {pair}: {type(sentiment)}")
                sentiment = 'neutral'
            ml_features['funding_sentiment'] = sentiment
        except Exception as e:
            logger.error(f"❌ Error en FundingRateAnalyzer para {pair}: {e}")
            ml_features['funding_sentiment'] = 'neutral'

        # Liquidation features
        is_near_liq, liq_details = self.liquidation_heatmap.is_near_liquidation_zone(pair, current_price)
        ml_features['near_liquidation'] = 1.0 if is_near_liq else 0.0
        if liq_details:
            ml_features['liquidation_direction'] = 1.0 if liq_details['direction'] == 'above' else 0.0

        # Liquidation bias (para logging)
        try:
            liq_bias, liq_conf = self.liquidation_heatmap.get_liquidation_bias(pair, current_price)
            # Validar tipos
            if not isinstance(liq_bias, str):
                logger.warning(f"⚠️ LiquidationHeatmap.get_liquidation_bias devolvió tipo incorrecto para {pair}: {type(liq_bias)}")
                liq_bias = 'neutral'
            if not isinstance(liq_conf, (int, float)):
                logger.warning(f"⚠️ LiquidationHeatmap.get_liquidation_bias confidence tipo incorrecto para {pair}: {type(liq_conf)}")
                liq_conf = 0
            ml_features['liquidation_bias'] = liq_bias
            ml_features['liquidation_confidence'] = liq_conf
        except Exception as e:
            logger.error(f"❌ Error en LiquidationHeatmap para {pair}: {e}")
            ml_features['liquidation_bias'] = 'neutral'
            ml_features['liquidation_confidence'] = 0

        # Volume profile features
        if ohlc_data is not None and pair in self.volume_profile.volume_profiles:
            is_near_poc, poc_distance = self.volume_profile.is_near_poc(pair, current_price)
            ml_features['near_poc'] = 1.0 if is_near_poc else 0.0
            ml_features['in_value_area'] = 1.0 if self.volume_profile.is_in_value_area(pair, current_price) else 0.0

        # Pattern recognition features
        if ohlc_data is not None:
            try:
                patterns = self.pattern_recognition.detect_all_patterns(ohlc_data)
                ml_features['has_pattern'] = 1.0 if patterns else 0.0
                ml_features['pattern_confidence'] = max([p['confidence'] for p in patterns], default=0.0)
                ml_features['pattern_detected'] = bool(patterns)
                ml_features['pattern_type'] = patterns[0]['pattern'] if patterns else 'NONE'
            except Exception as e:
                logger.error(f"❌ Error en pattern recognition para {pair}: {e}")
                ml_features['pattern_detected'] = False
                ml_features['pattern_type'] = 'NONE'
        else:
            ml_features['pattern_detected'] = False
            ml_features['pattern_type'] = 'NONE'

        # Session features
        session, multiplier = self.session_trading.get_current_session()
        ml_features['session_multiplier'] = multiplier
        ml_features['is_us_session'] = 1.0 if session == 'US' else 0.0

        logger.debug(f"📊 ML features generados para {pair}: {len(ml_features)} features totales")

        # VALIDACIÓN FINAL: Asegurar que siempre devolvemos un dict
        if not isinstance(ml_features, dict):
            logger.error(f"❌ CRÍTICO: ml_features no es dict para {pair}: {type(ml_features)}")
            logger.error(f"   Valor: {ml_features}")
            return {}  # Devolver dict vacío como fallback

        return ml_features

    def get_rl_state_extensions(
        self,
        pair: str,
        current_price: float,
        open_positions: List[str]
    ) -> Dict:
        """
        Genera extensiones de estado para RL Agent

        Args:
            pair: Par de trading
            current_price: Precio actual
            open_positions: Posiciones abiertas

        Returns:
            Dict con estados extendidos
        """
        state_extensions = {}

        # Correlation & Diversification
        if open_positions:
            state_extensions['diversification_score'] = self.correlation_matrix.get_diversification_score(open_positions)
            # Calcular correlaciones con posiciones abiertas
            correlated_count = 0
            for open_pair in open_positions:
                corr = self.correlation_matrix.get_correlation(pair, open_pair)
                if corr is not None and abs(corr) > 0.7:
                    correlated_count += 1
            state_extensions['correlated_positions'] = correlated_count
            state_extensions['correlation_risk'] = 1.0 - state_extensions['diversification_score']
        else:
            state_extensions['diversification_score'] = 1.0
            state_extensions['correlated_positions'] = 0
            state_extensions['correlation_risk'] = 0.0

        # Funding sentiment (string para consistencia con ML features)
        try:
            sentiment, strength, _ = self.funding_rate_analyzer.get_funding_sentiment(pair)
            # Validar tipos
            if not isinstance(sentiment, str):
                logger.warning(f"⚠️ FundingRateAnalyzer.get_funding_sentiment devolvió tipo incorrecto para {pair} en RL: {type(sentiment)}")
                sentiment = 'neutral'
            state_extensions['funding_sentiment'] = sentiment  # String: 'bullish', 'bearish', 'neutral'
            state_extensions['funding_strength'] = strength
            state_extensions['funding_rate'] = self.funding_rate_analyzer.fetch_funding_rate(pair) or 0.0
        except Exception as e:
            logger.error(f"❌ Error en FundingRateAnalyzer para {pair} en RL: {e}")
            state_extensions['funding_sentiment'] = 'neutral'
            state_extensions['funding_strength'] = 0
            state_extensions['funding_rate'] = 0.0

        # Liquidation bias (string para consistencia con ML features)
        try:
            bias, confidence = self.liquidation_heatmap.get_liquidation_bias(pair, current_price)
            # Validar tipos
            if not isinstance(bias, str):
                logger.warning(f"⚠️ LiquidationHeatmap.get_liquidation_bias devolvió tipo incorrecto para {pair} en RL: {type(bias)}")
                bias = 'neutral'
            state_extensions['liquidation_bias'] = bias  # String: 'bullish', 'bearish', 'neutral'
            state_extensions['liquidation_confidence'] = confidence
        except Exception as e:
            logger.error(f"❌ Error en LiquidationHeatmap para {pair} en RL: {e}")
            state_extensions['liquidation_bias'] = 'neutral'
            state_extensions['liquidation_confidence'] = 0

        # Session info
        session, multiplier = self.session_trading.get_current_session()
        state_extensions['current_session'] = session
        state_extensions['session_multiplier'] = multiplier

        # Order Flow (defaults neutros, se actualiza en enrich_signal con orderbook real)
        state_extensions['order_flow_bias'] = 'neutral'
        state_extensions['order_flow_ratio'] = 1.0

        # 🔧 FIX CRÍTICO: Añadir pattern detection a RL state extensions
        # Estos campos son CRÍTICOS para que el RL Agent calcule correctamente el composite score
        state_extensions['pattern_detected'] = False
        state_extensions['pattern_type'] = 'NONE'
        state_extensions['pattern_confidence'] = 0.0

        logger.debug(f"🤖 RL state extensions generados para {pair}")

        # VALIDACIÓN FINAL: Asegurar que siempre devolvemos un dict
        if not isinstance(state_extensions, dict):
            logger.error(f"❌ CRÍTICO: state_extensions no es dict para {pair}: {type(state_extensions)}")
            logger.error(f"   Valor: {state_extensions}")
            return {}  # Devolver dict vacío como fallback

        return state_extensions

    def update_price_history(self, pair: str, price: float, volume: float) -> None:
        """
        Actualiza historial de precios en todos los módulos

        Args:
            pair: Par de trading
            price: Precio actual
            volume: Volumen actual
        """
        self.correlation_matrix.update_price(pair, price)

    def get_full_statistics(self) -> Dict:
        """
        Obtiene estadísticas completas de todos los módulos

        Returns:
            Dict con stats de cada módulo
        """
        return {
            'correlation_matrix': self.correlation_matrix.get_statistics(),
            'liquidation_heatmap': self.liquidation_heatmap.get_statistics(),
            'funding_rate_analyzer': self.funding_rate_analyzer.get_statistics(),
            'volume_profile': self.volume_profile.get_statistics(),
            'pattern_recognition': self.pattern_recognition.get_statistics(),
            'session_trading': self.session_trading.get_statistics(),
            'order_flow': self.order_flow.get_statistics()
        }

    def is_everything_enabled(self) -> Dict[str, bool]:
        """
        Verifica qué módulos están habilitados

        Returns:
            Dict con status de cada módulo
        """
        return {
            'correlation_matrix': self.correlation_matrix.enabled,
            'liquidation_heatmap': self.liquidation_heatmap.enabled,
            'funding_rate_analyzer': self.funding_rate_analyzer.enabled,
            'volume_profile': self.volume_profile.enabled,
            'pattern_recognition': self.pattern_recognition.enabled,
            'session_trading': self.session_trading.enabled,
            'order_flow': self.order_flow.enabled
        }
