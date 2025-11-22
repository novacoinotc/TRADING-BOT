"""
Dynamic Threshold Manager - MODO EXPLORACIÓN
Sistema de threshold adaptativo que permite al RL Agent aprender del mercado real.

FILOSOFÍA:
- Explorar libremente en ENTRADA (threshold bajo, sin restricciones)
- Proteger agresivamente DURANTE el trade (Trade Manager)
- RL Agent aprende rápido de éxitos Y errores

El Trade Manager es nuestra red de seguridad 24/7:
- Stop Loss dinámico
- Breakeven protection
- Trailing stop
- Detección de reversiones

Por lo tanto, NO necesitamos sobre-restringir la ENTRADA.
"""
import logging
import json
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class DynamicThresholdManager:
    """
    Gestiona thresholds dinámicos para señales de trading.

    MODO EXPLORACIÓN:
    - Threshold BASE: 4.5 (accesible)
    - Threshold MIN: 3.5 (permite exploración)
    - Threshold MAX: 5.5 (sin sobre-restricción)

    Filosofía:
    - NO HAY BLOQUEOS - solo ajustes suaves
    - Trade Manager protege DURANTE el trade
    - RL Agent necesita experiencias para aprender

    Ajustes (informativos, no restrictivos):
    - Sesión US/EU: más agresivo (-0.7)
    - Sesión Asia: ligeramente conservador (+0.2)
    - F&G positivo: más agresivo (-0.4)
    - F&G bajo: ajuste mínimo (+0.1)
    - Racha de wins: aprovechar momentum (-0.3)
    - Racha de losses: ajuste suave (+0.2)
    """

    # Thresholds límites - MODO EXPLORACIÓN
    MIN_THRESHOLD = 3.5  # Permite exploración (pero no locura)
    MAX_THRESHOLD = 5.5  # Sin sobre-restricción
    DEFAULT_THRESHOLD = 4.5  # Base accesible

    # Ajustes por factor - MÁS AGRESIVOS para exploración
    ADJUSTMENTS = {
        'session_us_eu': -0.7,      # Más agresivo en mejores sesiones (antes -0.5)
        'session_asia': +0.2,       # Menos penalización (antes +0.3)
        'fear_greed_good': -0.4,    # Más agresivo con F&G positivo (antes -0.3)
        'fear_greed_moderate': -0.2, # Más agresivo (antes -0.1)
        'fear_greed_low': +0.1,      # Casi sin penalización (antes +0.5 bloqueante)
        'volatility_low': -0.3,      # Más agresivo en volatilidad baja (antes -0.2)
        'volatility_high': +0.2,     # Menos restrictivo (antes +0.4)
        'confidence_high': -0.4,     # Mucho más agresivo con ML alta (antes -0.3)
        'confidence_low': +0.1,      # Menos restrictivo (antes +0.2)
        'winrate_good': -0.3,        # Aprovechar momentum (antes -0.2)
        'winrate_bad': +0.2,         # Menos restrictivo (antes +0.3)
        'consecutive_losses': +0.2,  # Suave, no bloqueante (antes +0.5 bloqueante)
        'consecutive_wins': -0.3,    # Aprovechar racha (antes -0.2)
    }

    def __init__(self, experience_file: str = 'data/threshold_experiences.json'):
        """
        Args:
            experience_file: Archivo para guardar experiencias de aprendizaje
        """
        self.experience_file = Path(experience_file)
        self.experience_file.parent.mkdir(parents=True, exist_ok=True)

        # Estado interno
        self.current_threshold = self.DEFAULT_THRESHOLD
        self.last_adjustment_reasons = []
        self.experiences = []
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.recent_trades = []  # Últimos 20 trades para calcular win rate

        # Cargar experiencias previas
        self._load_experiences()

        # Log de filosofía del modo exploración
        logger.info("=" * 60)
        logger.info("🚀 MODO EXPLORACIÓN ACTIVADO")
        logger.info("=" * 60)
        logger.info("Filosofía: Explorar libremente, aprender de errores")
        logger.info(f"Threshold Base: {self.DEFAULT_THRESHOLD} (accesible)")
        logger.info(f"Threshold Min: {self.MIN_THRESHOLD} (exploración)")
        logger.info(f"Threshold Max: {self.MAX_THRESHOLD} (sin sobre-restricción)")
        logger.info("")
        logger.info("🛡️ PROTECCIÓN: Trade Manager monitorea 24/7")
        logger.info("   - Stop Loss dinámico")
        logger.info("   - Breakeven protection")
        logger.info("   - Trailing stop")
        logger.info("   - Detección de reversiones")
        logger.info("")
        logger.info("🎯 OBJETIVO: RL Agent aprende del mercado REAL")
        logger.info("=" * 60)

    def calculate_threshold(
        self,
        fear_greed_index: int = 50,
        current_session: str = 'UNKNOWN',
        volatility: str = 'medium',
        ml_confidence: float = 50.0,
        recent_win_rate: Optional[float] = None
    ) -> Tuple[float, str]:
        """
        Calcula el threshold óptimo basado en condiciones actuales.

        MODO EXPLORACIÓN: Sin bloqueos, solo ajustes suaves.
        El Trade Manager protege durante el trade.

        Args:
            fear_greed_index: Fear & Greed (0-100)
            current_session: Sesión actual (US, EUROPE, ASIA, etc.)
            volatility: Nivel de volatilidad (low, medium, high)
            ml_confidence: Confianza del modelo ML (0-100)
            recent_win_rate: Win rate de últimos trades (0-100)

        Returns:
            Tuple[threshold, explicación]
        """
        threshold = self.DEFAULT_THRESHOLD
        reasons = []

        # 🚀 MODO EXPLORACIÓN: Sin bloqueos, solo ajustes informativos
        # El Trade Manager protege durante el trade, no necesitamos restringir entrada

        # ============== AJUSTES POR FEAR & GREED ==============
        if fear_greed_index < 20:
            # Ajuste suave, NO bloqueante
            threshold += self.ADJUSTMENTS['fear_greed_low']
            reasons.append(f"F&G={fear_greed_index} (bajo) -> +0.1")
        elif fear_greed_index >= 40:
            # Más agresivo en mercado positivo
            threshold += self.ADJUSTMENTS['fear_greed_good']
            reasons.append(f"F&G={fear_greed_index} (positivo) -> -0.4")
        elif fear_greed_index >= 25:
            threshold += self.ADJUSTMENTS['fear_greed_moderate']
            reasons.append(f"F&G={fear_greed_index} (moderado) -> -0.2")

        # ============== AJUSTES POR SESIÓN ==============
        session_upper = current_session.upper()
        if session_upper in ['US', 'EUROPE', 'US_OPEN', 'EU_OPEN', 'EU_US_OVERLAP', 'LONDON', 'NEW_YORK']:
            threshold += self.ADJUSTMENTS['session_us_eu']
            reasons.append(f"Sesión {session_upper} (alta liquidez) -> -0.7")
        elif session_upper in ['ASIA', 'ASIAN', 'TOKYO', 'SYDNEY']:
            threshold += self.ADJUSTMENTS['session_asia']
            reasons.append(f"Sesión {session_upper} -> +0.2")

        # ============== AJUSTES POR VOLATILIDAD ==============
        vol_lower = volatility.lower()
        if vol_lower in ['low', 'baja']:
            threshold += self.ADJUSTMENTS['volatility_low']
            reasons.append(f"Volatilidad baja -> -0.3")
        elif vol_lower in ['high', 'alta', 'extreme', 'extrema']:
            threshold += self.ADJUSTMENTS['volatility_high']
            reasons.append(f"Volatilidad alta -> +0.2")

        # ============== AJUSTES POR ML CONFIDENCE ==============
        if ml_confidence > 60:
            threshold += self.ADJUSTMENTS['confidence_high']
            reasons.append(f"ML conf={ml_confidence:.0f}% (alta) -> -0.4")
        elif ml_confidence < 40:
            threshold += self.ADJUSTMENTS['confidence_low']
            reasons.append(f"ML conf={ml_confidence:.0f}% (baja) -> +0.1")

        # ============== AJUSTES POR WIN RATE ==============
        if recent_win_rate is not None:
            if recent_win_rate > 55:
                threshold += self.ADJUSTMENTS['winrate_good']
                reasons.append(f"Win rate={recent_win_rate:.0f}% (bueno) -> -0.3")
            elif recent_win_rate < 40:
                threshold += self.ADJUSTMENTS['winrate_bad']
                reasons.append(f"Win rate={recent_win_rate:.0f}% (malo) -> +0.2")

        # ============== AJUSTES POR RACHA ==============
        # Racha de pérdidas: ajuste SUAVE, no bloqueante
        if self.consecutive_losses >= 3:
            threshold += self.ADJUSTMENTS['consecutive_losses']
            reasons.append(f"{self.consecutive_losses} pérdidas -> +0.2")
            logger.info(f"⚠️ Racha de {self.consecutive_losses} pérdidas detectada (ajuste suave, NO bloqueo)")

        # Racha de ganancias: aprovechar momentum
        if self.consecutive_wins >= 3:
            threshold += self.ADJUSTMENTS['consecutive_wins']
            reasons.append(f"{self.consecutive_wins} wins -> -0.3 (momentum)")
            logger.info(f"🔥 Racha de {self.consecutive_wins} wins - aprovechando momentum")

        # ============== APLICAR LÍMITES ==============
        threshold = max(self.MIN_THRESHOLD, min(self.MAX_THRESHOLD, threshold))

        # Determinar modo
        if threshold <= 4.0:
            mode = "EXPLORACIÓN"
        elif threshold >= 5.0:
            mode = "CAUTELOSO"
        else:
            mode = "BALANCEADO"

        self.current_threshold = threshold
        self.last_adjustment_reasons = reasons

        explanation = self._build_explanation(threshold, reasons, mode)

        logger.info(
            f"🎚️ Threshold: {threshold:.2f} ({mode}) | "
            f"F&G={fear_greed_index} | Session={current_session} | Vol={volatility}"
        )

        return threshold, explanation

    def _build_explanation(self, threshold: float, reasons: list, mode: str) -> str:
        """Construye explicación legible del threshold."""
        base_msg = f"Threshold: {threshold:.2f} ({mode})"
        if reasons:
            adjustments = " | ".join(reasons[:3])  # Máximo 3 razones
            return f"{base_msg} | {adjustments}"
        return base_msg

    def record_trade_result(self, won: bool, threshold_used: float, market_conditions: Dict):
        """
        Registra resultado de un trade para aprendizaje.

        Args:
            won: Si el trade fue ganador
            threshold_used: Threshold que se usó
            market_conditions: Condiciones del mercado cuando se tomó el trade
        """
        # Actualizar rachas
        if won:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0

        # Agregar a historial reciente
        self.recent_trades.append({
            'won': won,
            'threshold': threshold_used,
            'timestamp': datetime.now().isoformat()
        })

        # Mantener solo últimos 20 trades
        if len(self.recent_trades) > 20:
            self.recent_trades = self.recent_trades[-20:]

        # Guardar experiencia para RL
        experience = {
            'timestamp': datetime.now().isoformat(),
            'threshold_used': threshold_used,
            'won': won,
            'market_conditions': {
                'fear_greed': market_conditions.get('fear_greed_index', 50),
                'session': market_conditions.get('current_session', 'UNKNOWN'),
                'volatility': market_conditions.get('volatility', 'medium'),
                'ml_confidence': market_conditions.get('confidence', 50)
            },
            'consecutive_wins_before': self.consecutive_wins - (1 if won else 0),
            'consecutive_losses_before': self.consecutive_losses - (0 if won else 1)
        }

        self.experiences.append(experience)
        self._save_experiences()

        result_emoji = "✅" if won else "❌"
        logger.info(
            f"{result_emoji} Trade registrado | Won={won} | Threshold={threshold_used:.2f} | "
            f"Racha: {self.consecutive_wins}W / {self.consecutive_losses}L"
        )

    def get_recent_win_rate(self) -> Optional[float]:
        """Calcula win rate de trades recientes."""
        if len(self.recent_trades) < 5:
            return None  # No suficientes datos

        wins = sum(1 for t in self.recent_trades if t['won'])
        return (wins / len(self.recent_trades)) * 100

    def get_status(self) -> Dict:
        """Retorna estado actual del manager."""
        return {
            'mode': 'EXPLORACIÓN',
            'current_threshold': self.current_threshold,
            'default_threshold': self.DEFAULT_THRESHOLD,
            'min_threshold': self.MIN_THRESHOLD,
            'max_threshold': self.MAX_THRESHOLD,
            'consecutive_wins': self.consecutive_wins,
            'consecutive_losses': self.consecutive_losses,
            'recent_win_rate': self.get_recent_win_rate(),
            'total_experiences': len(self.experiences),
            'last_adjustments': self.last_adjustment_reasons
        }

    def _load_experiences(self):
        """Carga experiencias previas del archivo."""
        try:
            if self.experience_file.exists():
                with open(self.experience_file, 'r') as f:
                    data = json.load(f)
                    self.experiences = data.get('experiences', [])
                    self.recent_trades = data.get('recent_trades', [])
                    self.consecutive_wins = data.get('consecutive_wins', 0)
                    self.consecutive_losses = data.get('consecutive_losses', 0)

                    logger.info(
                        f"📚 Cargadas {len(self.experiences)} experiencias | "
                        f"Win rate reciente: {self.get_recent_win_rate() or 'N/A'}%"
                    )
        except Exception as e:
            logger.warning(f"No se pudieron cargar experiencias: {e}")
            self.experiences = []

    def _save_experiences(self):
        """Guarda experiencias al archivo."""
        try:
            data = {
                'experiences': self.experiences[-1000:],  # Últimas 1000
                'recent_trades': self.recent_trades,
                'consecutive_wins': self.consecutive_wins,
                'consecutive_losses': self.consecutive_losses,
                'last_updated': datetime.now().isoformat()
            }

            with open(self.experience_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Error guardando experiencias: {e}")

    def get_experiences_for_rl(self) -> list:
        """
        Retorna experiencias formateadas para entrenamiento RL.

        Returns:
            Lista de experiencias para RL Agent
        """
        return [
            {
                'state': {
                    'fear_greed': exp['market_conditions']['fear_greed'],
                    'session': exp['market_conditions']['session'],
                    'volatility': exp['market_conditions']['volatility'],
                    'threshold': exp['threshold_used']
                },
                'action': 'TRADE',
                'reward': 1.0 if exp['won'] else -1.0,
                'next_state': None  # Episódico
            }
            for exp in self.experiences[-100:]  # Últimas 100 para entrenamiento
        ]


# Singleton global para uso en todo el sistema
_dynamic_threshold_manager: Optional[DynamicThresholdManager] = None


def get_dynamic_threshold_manager() -> DynamicThresholdManager:
    """Obtiene instancia singleton del DynamicThresholdManager."""
    global _dynamic_threshold_manager
    if _dynamic_threshold_manager is None:
        _dynamic_threshold_manager = DynamicThresholdManager()
    return _dynamic_threshold_manager
