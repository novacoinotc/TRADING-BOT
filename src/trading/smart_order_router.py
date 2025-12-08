"""
Smart Order Router - Selección inteligente de tipo de orden (Spot vs Futures)

Este módulo permite a la IA decidir automáticamente si usar:
- SPOT trading (sin apalancamiento, más seguro)
- FUTURES trading (con leverage, más agresivo)

Decisión basada en:
- Market regime (BULL = Futures, BEAR = Spot)
- Volatilidad (Alta vol = Futures, Baja vol = Spot)
- Confianza de señal (Alta confianza = Futures)
- Win rate histórico (Alto WR = más agresivo)
- Drawdown actual (Alto DD = conservador Spot)
"""

import logging
from typing import Dict, Literal, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    """Resultado de la decisión de routing"""
    order_type: Literal['spot', 'futures']
    leverage: int
    reasoning: str
    confidence: float  # 0-1


class SmartOrderRouter:
    """
    Router inteligente que decide entre Spot y Futures trading

    La IA tiene control total sobre esta decisión sin intervención humana
    """

    def __init__(self, config):
        self.config = config

        # Parámetros optimizables por la IA (configurables vía parameter_optimizer)
        self.min_confidence_for_futures = config.get('MIN_CONFIDENCE_FOR_FUTURES', 70.0)  # 60-85%
        self.min_winrate_for_futures = config.get('MIN_WINRATE_FOR_FUTURES', 55.0)  # 45-65%
        self.max_drawdown_for_futures = config.get('MAX_DRAWDOWN_FOR_FUTURES', 10.0)  # 5-15%
        self.volatility_threshold = config.get('VOLATILITY_THRESHOLD_FUTURES', 0.02)  # 0.015-0.03 (2%)

        # Leverage escalonado según condiciones (MAX 10x por seguridad)
        self.conservative_leverage = config.get('CONSERVATIVE_LEVERAGE', 3)  # 2-5x
        self.balanced_leverage = config.get('BALANCED_LEVERAGE', 7)  # 5-8x
        self.aggressive_leverage = config.get('AGGRESSIVE_LEVERAGE', 10)  # 8-10x MAX

        logger.info(f"SmartOrderRouter initialized with futures threshold: confidence={self.min_confidence_for_futures}%, winrate={self.min_winrate_for_futures}%")

    def decide_order_type(
        self,
        pair: str,
        confidence: float,  # Confianza de la señal (0-100)
        score: float,  # Score técnico (0-10)
        market_regime: str,  # 'bull', 'bear', 'sideways'
        regime_strength: str,  # 'low', 'medium', 'high'
        volatility: float,  # ATR normalizado
        win_rate: float,  # Win rate histórico (0-100)
        current_drawdown: float,  # Drawdown actual (0-100)
        ml_probability: Optional[float] = None  # Probabilidad ML WIN (0-1)
    ) -> RoutingDecision:
        """
        Decide si usar SPOT o FUTURES basado en múltiples factores

        Lógica de decisión (sin intervención humana):
        1. Si drawdown > max_drawdown_for_futures → SPOT (protección)
        2. Si win_rate < min_winrate_for_futures → SPOT (no confiamos)
        3. Si confidence < min_confidence_for_futures → SPOT (señal débil)
        4. Si market_regime == BEAR → SPOT (evitar apalancamiento en caída)
        5. Si todas las condiciones favorables → FUTURES (agresivo)

        Returns:
            RoutingDecision con order_type, leverage y reasoning
        """
        reasons = []
        futures_score = 0  # 0-10 (>=6 = Futures, <6 = Spot)

        # Factor 1: Drawdown (crítico para protección)
        if current_drawdown > self.max_drawdown_for_futures:
            reasons.append(f"❌ Drawdown alto ({current_drawdown:.1f}% > {self.max_drawdown_for_futures}%)")
            futures_score -= 3
        elif current_drawdown < 5.0:
            reasons.append(f"✅ Drawdown bajo ({current_drawdown:.1f}%)")
            futures_score += 2
        else:
            reasons.append(f"⚠️ Drawdown moderado ({current_drawdown:.1f}%)")
            futures_score += 1

        # Factor 2: Win Rate histórico
        if win_rate < self.min_winrate_for_futures:
            reasons.append(f"❌ Win rate bajo ({win_rate:.1f}% < {self.min_winrate_for_futures}%)")
            futures_score -= 2
        elif win_rate > 65.0:
            reasons.append(f"✅ Win rate excelente ({win_rate:.1f}%)")
            futures_score += 3
        else:
            reasons.append(f"✅ Win rate aceptable ({win_rate:.1f}%)")
            futures_score += 1

        # Factor 3: Confianza de señal
        if confidence < self.min_confidence_for_futures:
            reasons.append(f"❌ Confianza baja ({confidence:.1f}% < {self.min_confidence_for_futures}%)")
            futures_score -= 2
        elif confidence > 80.0:
            reasons.append(f"✅ Confianza muy alta ({confidence:.1f}%)")
            futures_score += 2
        else:
            reasons.append(f"✅ Confianza aceptable ({confidence:.1f}%)")
            futures_score += 1

        # Factor 4: Market Regime
        if market_regime.lower() == 'bull':
            if regime_strength == 'high':
                reasons.append(f"✅ BULL market fuerte (ideal para Futures)")
                futures_score += 3
            else:
                reasons.append(f"✅ BULL market ({regime_strength})")
                futures_score += 1
        elif market_regime.lower() == 'bear':
            reasons.append(f"❌ BEAR market (evitar Futures)")
            futures_score -= 2
        else:  # sideways
            reasons.append(f"⚠️ Market SIDEWAYS (neutral)")
            # No suma ni resta

        # Factor 5: Volatilidad
        if volatility > self.volatility_threshold:
            reasons.append(f"✅ Alta volatilidad ({volatility:.3f}, bueno para scalping con leverage)")
            futures_score += 2
        elif volatility < 0.01:
            reasons.append(f"⚠️ Baja volatilidad ({volatility:.3f})")
            futures_score -= 1
        else:
            reasons.append(f"✅ Volatilidad moderada ({volatility:.3f})")
            futures_score += 1

        # Factor 6: ML Probability (si disponible)
        if ml_probability is not None:
            if ml_probability > 0.7:
                reasons.append(f"✅ ML muy confiado ({ml_probability*100:.1f}% WIN)")
                futures_score += 2
            elif ml_probability < 0.55:
                reasons.append(f"❌ ML poco confiado ({ml_probability*100:.1f}% WIN)")
                futures_score -= 1

        # Factor 7: Score técnico
        if score >= 7.0:
            reasons.append(f"✅ Score técnico excelente ({score:.1f}/10)")
            futures_score += 1
        elif score < 5.5:
            reasons.append(f"⚠️ Score técnico bajo ({score:.1f}/10)")
            futures_score -= 1

        # DECISIÓN FINAL
        if futures_score >= 6:
            order_type = 'futures'

            # Determinar leverage según agresividad
            if futures_score >= 10:
                leverage = self.aggressive_leverage  # 15x
                aggressiveness = "AGRESIVO"
            elif futures_score >= 8:
                leverage = self.balanced_leverage  # 8x
                aggressiveness = "BALANCEADO"
            else:
                leverage = self.conservative_leverage  # 3x
                aggressiveness = "CONSERVADOR"

            final_reason = f"🚀 FUTURES {leverage}x ({aggressiveness}) - Score={futures_score}/10"
        else:
            order_type = 'spot'
            leverage = 1
            final_reason = f"🛡️ SPOT (sin leverage) - Score={futures_score}/10 (insuficiente para Futures)"

        # Calcular confianza de la decisión
        decision_confidence = min(abs(futures_score - 5) / 5.0, 1.0)  # 0-1

        full_reasoning = f"{final_reason}\n" + "\n".join([f"  • {r}" for r in reasons])

        logger.info(f"🎯 Smart Routing para {pair}: {order_type.upper()} {leverage}x (confidence={decision_confidence:.2f})")
        logger.info(f"Reasoning:\n{full_reasoning}")

        return RoutingDecision(
            order_type=order_type,
            leverage=leverage,
            reasoning=full_reasoning,
            confidence=decision_confidence
        )

    def get_max_leverage_for_experience(self, total_trades: int) -> int:
        """
        Límite de leverage basado en experiencia (protección)

        Escalonamiento:
        - 0-50 trades: máximo 3x
        - 50-100 trades: máximo 5x
        - 100-200 trades: máximo 7x
        - 200+ trades: máximo 10x (límite absoluto del sistema)
        """
        if total_trades < 50:
            return 3
        elif total_trades < 100:
            return 5
        elif total_trades < 200:
            return 7
        else:
            return 10  # MAX ABSOLUTO - sincronizado con trading_schemas.py

    def adjust_leverage_by_experience(self, desired_leverage: int, total_trades: int) -> int:
        """
        Ajusta el leverage deseado según la experiencia del bot

        Esto evita que un bot novato use 10x inmediatamente
        """
        max_allowed = self.get_max_leverage_for_experience(total_trades)

        if desired_leverage > max_allowed:
            logger.warning(f"⚠️ Leverage ajustado: {desired_leverage}x → {max_allowed}x (experiencia: {total_trades} trades)")
            return max_allowed

        return desired_leverage

    def should_use_futures_for_pair(self, pair: str) -> bool:
        """
        Determina si un par soporta Futures trading

        Algunos pares exóticos solo tienen Spot
        """
        # Pares principales con Futures disponibles en Binance
        futures_supported_pairs = [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT',
            'SOL/USDT', 'DOT/USDT', 'DOGE/USDT', 'AVAX/USDT', 'MATIC/USDT',
            'LINK/USDT', 'UNI/USDT', 'ATOM/USDT', 'LTC/USDT', 'ETC/USDT',
            'XLM/USDT', 'ALGO/USDT', 'VET/USDT', 'FIL/USDT', 'TRX/USDT',
            'NEAR/USDT', 'AAVE/USDT', 'GRT/USDT', 'SAND/USDT', 'MANA/USDT'
        ]

        return pair in futures_supported_pairs


# Parámetros optimizables que se pueden agregar a config.py
SMART_ROUTING_PARAMS = {
    # Habilitación (la IA puede apagarlo si no funciona bien)
    'SMART_ROUTING_ENABLED': True,  # True/False

    # Thresholds para permitir Futures (optimizables 60-85%, 45-65%, 5-15%)
    'MIN_CONFIDENCE_FOR_FUTURES': 70.0,  # % confianza mínima
    'MIN_WINRATE_FOR_FUTURES': 55.0,  # % win rate mínimo
    'MAX_DRAWDOWN_FOR_FUTURES': 10.0,  # % drawdown máximo

    # Volatilidad threshold (optimizable 0.015-0.03)
    'VOLATILITY_THRESHOLD_FUTURES': 0.02,  # 2% ATR normalizado

    # Leverage levels (optimizables 2-5, 5-10, 10-20)
    'CONSERVATIVE_LEVERAGE': 3,  # Para señales normales
    'BALANCED_LEVERAGE': 8,  # Para señales buenas
    'AGGRESSIVE_LEVERAGE': 15,  # Para señales excelentes
}
