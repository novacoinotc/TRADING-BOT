"""
Dynamic Threshold Manager - Sistema de threshold adaptativo
Permite experimentar con thresholds más bajos durante condiciones favorables
mientras mantiene seguridad durante condiciones adversas.
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

    Filosofía:
    - Threshold BASE (conservador): 5.5
    - Threshold EXPERIMENTAL (buenas condiciones): 4.0 - 4.5
    - El sistema aprende de sus experiencias y ajusta automáticamente

    Condiciones para BAJAR threshold (más trades):
    - Sesión US/European (mejor liquidez)
    - Fear & Greed > 25 (mercado no en pánico)
    - Volatilidad moderada (no extrema)
    - ML confidence > 50%
    - Win rate reciente > 45%

    Condiciones para SUBIR threshold (más conservador):
    - Sesión asiática (menor liquidez)
    - Fear & Greed < 20 (miedo extremo)
    - Alta volatilidad
    - 3+ pérdidas consecutivas
    - Win rate reciente < 35%

    Reglas de SEGURIDAD:
    - NUNCA menos de 4.0
    - NUNCA bajar si F&G < 20
    - NUNCA bajar si 3+ pérdidas consecutivas
    """

    # Thresholds límites
    MIN_THRESHOLD = 4.0  # NUNCA menos que esto
    MAX_THRESHOLD = 6.0  # Máximo conservador
    DEFAULT_THRESHOLD = 5.5  # Base conservador

    # Ajustes por factor
    ADJUSTMENTS = {
        'session_us_eu': -0.5,      # US/EU: bajar 0.5
        'session_asia': +0.3,       # Asia: subir 0.3
        'fear_greed_good': -0.3,    # F&G > 40: bajar 0.3
        'fear_greed_moderate': -0.1, # F&G 25-40: bajar 0.1
        'fear_greed_extreme': +0.5,  # F&G < 20: subir 0.5
        'volatility_low': -0.2,      # Baja volatilidad: bajar 0.2
        'volatility_high': +0.4,     # Alta volatilidad: subir 0.4
        'confidence_high': -0.3,     # ML conf > 60%: bajar 0.3
        'confidence_low': +0.2,      # ML conf < 40%: subir 0.2
        'winrate_good': -0.2,        # Win rate > 55%: bajar 0.2
        'winrate_bad': +0.3,         # Win rate < 40%: subir 0.3
        'consecutive_losses': +0.5,  # 3+ pérdidas: subir 0.5
        'consecutive_wins': -0.2,    # 3+ ganancias: bajar 0.2
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

        logger.info(
            f"🎚️ DynamicThresholdManager inicializado | "
            f"Base={self.DEFAULT_THRESHOLD} | Min={self.MIN_THRESHOLD} | Max={self.MAX_THRESHOLD}"
        )

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

        # ============== REGLAS DE SEGURIDAD (PRIMERO) ==============

        # REGLA 1: Si Fear & Greed < 20, NO experimentar
        if fear_greed_index < 20:
            threshold = max(self.DEFAULT_THRESHOLD, threshold)
            threshold += self.ADJUSTMENTS['fear_greed_extreme']
            reasons.append(f"SEGURIDAD: F&G={fear_greed_index} (miedo extremo) -> +0.5")
            self.current_threshold = min(self.MAX_THRESHOLD, threshold)
            self.last_adjustment_reasons = reasons
            explanation = self._build_explanation(threshold, reasons, "CONSERVADOR")
            logger.warning(f"🔒 Threshold BLOQUEADO en modo conservador: {threshold:.2f} | F&G={fear_greed_index}")
            return threshold, explanation

        # REGLA 2: Si 3+ pérdidas consecutivas, NO experimentar
        if self.consecutive_losses >= 3:
            threshold = max(self.DEFAULT_THRESHOLD, threshold)
            threshold += self.ADJUSTMENTS['consecutive_losses']
            reasons.append(f"SEGURIDAD: {self.consecutive_losses} pérdidas consecutivas -> +0.5")
            self.current_threshold = min(self.MAX_THRESHOLD, threshold)
            self.last_adjustment_reasons = reasons
            explanation = self._build_explanation(threshold, reasons, "CONSERVADOR")
            logger.warning(f"🔒 Threshold BLOQUEADO por pérdidas: {threshold:.2f} | Losses={self.consecutive_losses}")
            return threshold, explanation

        # ============== AJUSTES POR SESIÓN ==============
        session_upper = current_session.upper()
        if session_upper in ['US', 'EUROPE', 'US_OPEN', 'EU_OPEN', 'LONDON', 'NEW_YORK']:
            threshold += self.ADJUSTMENTS['session_us_eu']
            reasons.append(f"Sesión {session_upper} (alta liquidez) -> -0.5")
        elif session_upper in ['ASIA', 'TOKYO', 'SYDNEY', 'ASIAN']:
            threshold += self.ADJUSTMENTS['session_asia']
            reasons.append(f"Sesión {session_upper} (baja liquidez) -> +0.3")

        # ============== AJUSTES POR FEAR & GREED ==============
        if fear_greed_index > 40:
            threshold += self.ADJUSTMENTS['fear_greed_good']
            reasons.append(f"F&G={fear_greed_index} (optimista) -> -0.3")
        elif fear_greed_index >= 25:
            threshold += self.ADJUSTMENTS['fear_greed_moderate']
            reasons.append(f"F&G={fear_greed_index} (moderado) -> -0.1")
        # < 20 ya se manejó arriba como regla de seguridad

        # ============== AJUSTES POR VOLATILIDAD ==============
        vol_lower = volatility.lower()
        if vol_lower in ['low', 'baja']:
            threshold += self.ADJUSTMENTS['volatility_low']
            reasons.append(f"Volatilidad baja -> -0.2")
        elif vol_lower in ['high', 'alta', 'extreme', 'extrema']:
            threshold += self.ADJUSTMENTS['volatility_high']
            reasons.append(f"Volatilidad alta -> +0.4")

        # ============== AJUSTES POR ML CONFIDENCE ==============
        if ml_confidence > 60:
            threshold += self.ADJUSTMENTS['confidence_high']
            reasons.append(f"ML confidence={ml_confidence:.0f}% (alta) -> -0.3")
        elif ml_confidence < 40:
            threshold += self.ADJUSTMENTS['confidence_low']
            reasons.append(f"ML confidence={ml_confidence:.0f}% (baja) -> +0.2")

        # ============== AJUSTES POR WIN RATE ==============
        if recent_win_rate is not None:
            if recent_win_rate > 55:
                threshold += self.ADJUSTMENTS['winrate_good']
                reasons.append(f"Win rate={recent_win_rate:.0f}% (bueno) -> -0.2")
            elif recent_win_rate < 40:
                threshold += self.ADJUSTMENTS['winrate_bad']
                reasons.append(f"Win rate={recent_win_rate:.0f}% (malo) -> +0.3")

        # ============== AJUSTES POR RACHA ==============
        if self.consecutive_wins >= 3:
            threshold += self.ADJUSTMENTS['consecutive_wins']
            reasons.append(f"{self.consecutive_wins} wins consecutivos -> -0.2")

        # ============== APLICAR LÍMITES ==============
        threshold = max(self.MIN_THRESHOLD, min(self.MAX_THRESHOLD, threshold))

        # Determinar modo
        if threshold <= 4.5:
            mode = "EXPERIMENTAL"
        elif threshold >= 5.5:
            mode = "CONSERVADOR"
        else:
            mode = "BALANCEADO"

        self.current_threshold = threshold
        self.last_adjustment_reasons = reasons

        explanation = self._build_explanation(threshold, reasons, mode)

        logger.info(
            f"🎚️ Threshold calculado: {threshold:.2f} | Modo: {mode} | "
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
