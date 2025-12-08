"""
Dynamic Take Profit Manager - Ajusta TPs según oportunidad

GROWTH API: Aprovecha criticality scores para maximizar profits
- Critical news (score 90+): TP hasta 1.5-2%
- Pre-pump (score 80-89): TP hasta 0.8-1.2%
- Normal signals: TP scalping 0.3-0.5%
"""
import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class DynamicTPManager:
    """
    Gestiona Take Profits dinámicos basados en contexto

    Adapta TPs según:
    - Criticality score de noticias
    - Market regime (BULL = TPs más altos)
    - Volatilidad (alta volatilidad = TPs más rápidos)
    - Social buzz (alto buzz = TPs más agresivos)
    """

    def __init__(
        self,
        tp1_base: float = 0.3,
        tp2_base: float = 0.8,
        tp3_base: float = 1.5,
        dynamic_multiplier: float = 1.5,
        high_criticality_threshold: int = 85
    ):
        """
        Args:
            tp1_base: TP1 base en % (scalping)
            tp2_base: TP2 base en % (medio)
            tp3_base: TP3 base en % (agresivo)
            dynamic_multiplier: Multiplicador para oportunidades críticas
            high_criticality_threshold: Score mínimo para TPs altos
        """
        self.tp1_base = tp1_base
        self.tp2_base = tp2_base
        self.tp3_base = tp3_base
        self.dynamic_multiplier = dynamic_multiplier
        self.high_criticality_threshold = high_criticality_threshold

        logger.info("📊 Dynamic TP Manager inicializado")
        logger.info(f"   Base TPs: {tp1_base}%, {tp2_base}%, {tp3_base}%")
        logger.info(f"   Multiplier: {dynamic_multiplier}x")
        logger.info(f"   High criticality threshold: {high_criticality_threshold}")

    def calculate_dynamic_tps(
        self,
        entry_price: float,
        signal_metadata: Optional[Dict] = None,
        market_regime: str = 'SIDEWAYS',
        volatility: str = 'medium',
        side: str = 'BUY'
    ) -> Tuple[float, float, float]:
        """
        Calcula TPs dinámicos basados en contexto

        Args:
            entry_price: Precio de entrada
            signal_metadata: Metadata del signal (criticality_score, source, etc)
            market_regime: BULL, BEAR, SIDEWAYS
            volatility: high, medium, low
            side: 'BUY' (LONG) o 'SELL' (SHORT) - CRÍTICO para dirección del TP

        Returns:
            Tuple (tp1, tp2, tp3) en precio absoluto
        """
        # Start with base TPs
        tp1_pct = self.tp1_base
        tp2_pct = self.tp2_base
        tp3_pct = self.tp3_base

        adjustment_reason = []

        # LAYER 1: News Criticality (GROWTH API)
        if signal_metadata:
            criticality_score = signal_metadata.get('criticality_score', 0)
            source = signal_metadata.get('source', '')

            if criticality_score >= self.high_criticality_threshold:
                # Oportunidad CRÍTICA → TPs más altos
                multiplier = self.dynamic_multiplier
                tp1_pct *= multiplier
                tp2_pct *= multiplier
                tp3_pct *= multiplier

                adjustment_reason.append(
                    f"Critical news (score={criticality_score}) → x{multiplier}"
                )

            elif criticality_score >= 75:
                # Oportunidad BUENA → TPs medios
                multiplier = 1.3
                tp1_pct *= multiplier
                tp2_pct *= multiplier
                tp3_pct *= multiplier

                adjustment_reason.append(
                    f"Good opportunity (score={criticality_score}) → x{multiplier}"
                )

            # Pre-pump específico → TPs rápidos pero más altos
            if 'pre_pump' in source:
                tp1_pct = max(tp1_pct, 0.5)  # Mínimo 0.5% en pre-pump
                tp2_pct = max(tp2_pct, 1.0)  # Mínimo 1.0%
                adjustment_reason.append("Pre-pump pattern → higher TPs")

            # Social buzz → TPs más agresivos (FOMO esperado)
            if signal_metadata.get('social_buzz', 0) == 1:
                tp2_pct *= 1.2
                tp3_pct *= 1.3
                adjustment_reason.append("High social buzz → +20-30% TPs")

        # LAYER 2: Market Regime
        if market_regime == 'BULL':
            # BULL → TPs más altos (momentum continúa)
            tp1_pct *= 1.15
            tp2_pct *= 1.25
            tp3_pct *= 1.35
            adjustment_reason.append("BULL market → +15-35% TPs")

        elif market_regime == 'BEAR':
            # BEAR → TPs más rápidos (reversión rápida)
            tp1_pct *= 0.9
            tp2_pct *= 0.85
            tp3_pct *= 0.8
            adjustment_reason.append("BEAR market → -10-20% TPs (fast exit)")

        # LAYER 3: Volatilidad
        if volatility == 'high':
            # Alta volatilidad → TPs más rápidos (capturar antes de reversión)
            tp1_pct *= 0.85
            tp2_pct *= 0.8
            adjustment_reason.append("High volatility → -15-20% TPs (quick capture)")

        elif volatility == 'low':
            # Baja volatilidad → TPs más pacientes
            tp1_pct *= 1.1
            tp2_pct *= 1.15
            adjustment_reason.append("Low volatility → +10-15% TPs (patient)")

        # Aplicar límites conservadores (nunca exceder máximos razonables)
        tp1_pct = min(tp1_pct, 1.0)   # Max 1%
        tp2_pct = min(tp2_pct, 2.0)   # Max 2%
        tp3_pct = min(tp3_pct, 3.5)   # Max 3.5%

        # Calcular precios absolutos según dirección (LONG vs SHORT)
        # ================================================================
        # LONG (BUY): TP está ARRIBA del entry → sumar porcentaje
        # SHORT (SELL): TP está ABAJO del entry → restar porcentaje
        # ================================================================
        if side == 'BUY':  # LONG
            tp1_price = entry_price * (1 + tp1_pct / 100)
            tp2_price = entry_price * (1 + tp2_pct / 100)
            tp3_price = entry_price * (1 + tp3_pct / 100)
        else:  # SHORT (SELL)
            tp1_price = entry_price * (1 - tp1_pct / 100)
            tp2_price = entry_price * (1 - tp2_pct / 100)
            tp3_price = entry_price * (1 - tp3_pct / 100)

        # Log ajustes
        if adjustment_reason:
            logger.info(
                f"📊 Dynamic TPs calculated: {tp1_pct:.2f}%, {tp2_pct:.2f}%, {tp3_pct:.2f}%"
            )
            logger.info(f"   Adjustments: {'; '.join(adjustment_reason)}")
        else:
            logger.debug(
                f"📊 Base TPs: {tp1_pct:.2f}%, {tp2_pct:.2f}%, {tp3_pct:.2f}%"
            )

        return tp1_price, tp2_price, tp3_price

    def should_hold_longer(
        self,
        current_price: float,
        entry_price: float,
        tp1_price: float,
        signal_metadata: Optional[Dict] = None,
        side: str = 'BUY'
    ) -> bool:
        """
        Determina si debe mantener el trade más tiempo

        Casos:
        - Criticality score muy alto (>90) + precio subiendo → hold para TP2/TP3
        - Pre-pump con momentum → hold
        - Social buzz aumentando → hold

        Args:
            current_price: Precio actual
            entry_price: Precio de entrada
            tp1_price: Precio TP1
            signal_metadata: Metadata del signal
            side: 'BUY' (LONG) o 'SELL' (SHORT)

        Returns:
            True si debe mantener más tiempo
        """
        if not signal_metadata:
            return False

        # Si ya llegó a TP1 o más (calcular profit según dirección)
        if side == 'BUY':  # LONG
            current_profit_pct = ((current_price - entry_price) / entry_price) * 100
        else:  # SHORT
            current_profit_pct = ((entry_price - current_price) / entry_price) * 100

        if current_profit_pct < (tp1_price / entry_price - 1) * 100:
            return False  # Aún no llegó a TP1

        # Verificar si debería hold para TPs más altos
        criticality_score = signal_metadata.get('criticality_score', 0)
        source = signal_metadata.get('source', '')

        # Critical news con score 90+ → hold para TP2 mínimo
        if criticality_score >= 90 and current_profit_pct < 0.8:
            logger.info(
                f"💎 Holding trade (critical news score={criticality_score}, "
                f"current: {current_profit_pct:.2f}%, target: 0.8%+)"
            )
            return True

        # Pre-pump con buen momentum → hold
        if 'pre_pump' in source and current_profit_pct < 1.0:
            logger.info(
                f"💎 Holding trade (pre-pump pattern, "
                f"current: {current_profit_pct:.2f}%, target: 1.0%+)"
            )
            return True

        return False

    def update_parameters(
        self,
        tp1_base: Optional[float] = None,
        tp2_base: Optional[float] = None,
        tp3_base: Optional[float] = None,
        dynamic_multiplier: Optional[float] = None,
        high_criticality_threshold: Optional[int] = None
    ):
        """
        Actualiza parámetros (llamado por Parameter Optimizer)

        Args:
            tp1_base: Nuevo TP1 base
            tp2_base: Nuevo TP2 base
            tp3_base: Nuevo TP3 base
            dynamic_multiplier: Nuevo multiplicador
            high_criticality_threshold: Nuevo threshold
        """
        if tp1_base is not None:
            self.tp1_base = tp1_base
        if tp2_base is not None:
            self.tp2_base = tp2_base
        if tp3_base is not None:
            self.tp3_base = tp3_base
        if dynamic_multiplier is not None:
            self.dynamic_multiplier = dynamic_multiplier
        if high_criticality_threshold is not None:
            self.high_criticality_threshold = high_criticality_threshold

        logger.info(
            f"🔄 Dynamic TP parameters updated: "
            f"TPs={self.tp1_base:.2f}%, {self.tp2_base:.2f}%, {self.tp3_base:.2f}% | "
            f"Multiplier={self.dynamic_multiplier}x | "
            f"Threshold={self.high_criticality_threshold}"
        )
