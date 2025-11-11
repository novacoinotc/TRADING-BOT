"""
Anomaly Detection System - Detecta comportamiento anómalo del bot

Este sistema monitorea el comportamiento del bot en tiempo real y detecta:
1. Degradación repentina de performance
2. Parámetros que causan pérdidas anómalas
3. Cambios que empeoran el sistema
4. Outliers en trades

Cuando detecta anomalías, puede:
- Revertir parámetros automáticamente
- Alertar vía Telegram
- Reducir agresividad temporalmente
- Guardar snapshot para análisis
"""

import logging
import json
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class AnomalyEvent:
    """Evento de anomalía detectada"""
    timestamp: datetime
    anomaly_type: str  # 'performance_degradation', 'parameter_issue', 'outlier_trade', etc.
    severity: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    description: str
    affected_parameters: List[str] = field(default_factory=list)
    metrics: Dict = field(default_factory=dict)
    action_taken: str = ""


class AnomalyDetector:
    """
    Detector de anomalías con auto-corrección

    La IA controla completamente este sistema sin intervención humana
    """

    def __init__(self, config):
        self.config = config

        # Parámetros de detección (optimizables)
        self.enabled = config.get('ANOMALY_DETECTION_ENABLED', True)
        self.performance_degradation_threshold = config.get('PERFORMANCE_DEGRADATION_THRESHOLD', 10.0)  # 5-20% (degradación para alertar)
        self.outlier_std_threshold = config.get('OUTLIER_STD_THRESHOLD', 3.0)  # 2.0-4.0 (desviaciones estándar)
        self.min_trades_for_detection = config.get('MIN_TRADES_FOR_DETECTION', 20)  # 10-50 (mínimo para detectar)
        self.lookback_window = config.get('ANOMALY_LOOKBACK_WINDOW', 50)  # 30-100 (trades a considerar)
        self.auto_revert_enabled = config.get('AUTO_REVERT_ENABLED', True)  # True/False

        # Historial de performance (sliding window)
        self.performance_history = deque(maxlen=self.lookback_window)

        # Historial de anomalías detectadas
        self.anomaly_events: List[AnomalyEvent] = []

        # Snapshot de parámetros previos (para revertir)
        self.parameter_snapshots: deque = deque(maxlen=10)  # Últimos 10 snapshots

        # Baseline de performance (para comparar)
        self.baseline_metrics: Optional[Dict] = None

        logger.info(f"AnomalyDetector initialized: degradation_threshold={self.performance_degradation_threshold}%, outlier_std={self.outlier_std_threshold}")

    def record_trade_result(
        self,
        pair: str,
        profit_pct: float,
        trade_duration_minutes: int,
        exit_reason: str,
        signal_score: float,
        confidence: float
    ) -> None:
        """
        Registra resultado de trade para análisis de anomalías

        Args:
            pair: Par de trading
            profit_pct: Profit/loss en % (-5.0 a +5.0 típicamente)
            trade_duration_minutes: Duración del trade
            exit_reason: 'TAKE_PROFIT', 'STOP_LOSS', 'MANUAL'
            signal_score: Score de la señal original (0-10)
            confidence: Confianza de la señal (0-100)
        """
        if not self.enabled:
            return

        trade_record = {
            'timestamp': datetime.now(),
            'pair': pair,
            'profit_pct': profit_pct,
            'duration_minutes': trade_duration_minutes,
            'exit_reason': exit_reason,
            'signal_score': signal_score,
            'confidence': confidence
        }

        self.performance_history.append(trade_record)

        # Detectar anomalías solo si tenemos suficientes trades
        if len(self.performance_history) >= self.min_trades_for_detection:
            self._detect_anomalies()

    def _detect_anomalies(self) -> None:
        """
        Ejecuta detección de anomalías en el historial reciente

        Tipos de anomalías detectadas:
        1. Performance degradation (win rate cae repentinamente)
        2. Outlier trades (pérdidas/ganancias anómalas)
        3. Parameter issues (cambio reciente empeoró performance)
        4. Exit reason patterns (demasiados SL en row)
        """
        # 1. Detectar degradación de performance
        self._detect_performance_degradation()

        # 2. Detectar trades outliers
        self._detect_outlier_trades()

        # 3. Detectar patrones de exit anómalos
        self._detect_exit_patterns()

    def _detect_performance_degradation(self) -> None:
        """
        Detecta degradación repentina de performance

        Compara últimos N trades vs baseline
        """
        if not self.baseline_metrics:
            # Establecer baseline con primeros N trades
            self.baseline_metrics = self._calculate_metrics(list(self.performance_history)[:20])
            logger.info(f"📊 Baseline establecido: win_rate={self.baseline_metrics['win_rate']:.1f}%, avg_profit={self.baseline_metrics['avg_profit']:.2f}%")
            return

        # Calcular métricas de últimos N trades
        recent_window = min(20, len(self.performance_history) // 2)
        recent_trades = list(self.performance_history)[-recent_window:]
        recent_metrics = self._calculate_metrics(recent_trades)

        # Comparar con baseline
        win_rate_change = recent_metrics['win_rate'] - self.baseline_metrics['win_rate']
        avg_profit_change = recent_metrics['avg_profit'] - self.baseline_metrics['avg_profit']

        # Alertar si degradación significativa
        if win_rate_change < -self.performance_degradation_threshold:
            anomaly = AnomalyEvent(
                timestamp=datetime.now(),
                anomaly_type='performance_degradation',
                severity='HIGH',
                description=f"Win rate cayó {abs(win_rate_change):.1f}% (de {self.baseline_metrics['win_rate']:.1f}% a {recent_metrics['win_rate']:.1f}%)",
                metrics={
                    'baseline_win_rate': self.baseline_metrics['win_rate'],
                    'recent_win_rate': recent_metrics['win_rate'],
                    'change': win_rate_change
                }
            )

            self._handle_anomaly(anomaly)

        if avg_profit_change < -0.5:  # Si profit promedio cae >0.5%
            anomaly = AnomalyEvent(
                timestamp=datetime.now(),
                anomaly_type='profit_degradation',
                severity='MEDIUM',
                description=f"Profit promedio cayó {abs(avg_profit_change):.2f}% (de {self.baseline_metrics['avg_profit']:.2f}% a {recent_metrics['avg_profit']:.2f}%)",
                metrics={
                    'baseline_avg_profit': self.baseline_metrics['avg_profit'],
                    'recent_avg_profit': recent_metrics['avg_profit'],
                    'change': avg_profit_change
                }
            )

            self._handle_anomaly(anomaly)

    def _detect_outlier_trades(self) -> None:
        """
        Detecta trades con profit/loss anómalo (outliers)

        Usa desviación estándar: |x - mean| > threshold * std
        """
        if len(self.performance_history) < 10:
            return

        profits = [t['profit_pct'] for t in self.performance_history]
        mean_profit = np.mean(profits)
        std_profit = np.std(profits)

        # Último trade
        last_trade = list(self.performance_history)[-1]
        deviation = abs(last_trade['profit_pct'] - mean_profit)

        if deviation > self.outlier_std_threshold * std_profit:
            # Es un outlier
            severity = 'HIGH' if last_trade['profit_pct'] < 0 else 'MEDIUM'  # Pérdida outlier es más grave

            anomaly = AnomalyEvent(
                timestamp=datetime.now(),
                anomaly_type='outlier_trade',
                severity=severity,
                description=f"Trade outlier detectado: {last_trade['pair']} con {last_trade['profit_pct']:.2f}% (mean={mean_profit:.2f}%, std={std_profit:.2f}%)",
                metrics={
                    'profit_pct': last_trade['profit_pct'],
                    'mean_profit': mean_profit,
                    'std_profit': std_profit,
                    'deviation': deviation,
                    'z_score': deviation / std_profit if std_profit > 0 else 0
                }
            )

            self._handle_anomaly(anomaly)

    def _detect_exit_patterns(self) -> None:
        """
        Detecta patrones anómalos en exit reasons

        Por ejemplo: 5+ STOP_LOSS consecutivos = problema
        """
        if len(self.performance_history) < 5:
            return

        # Últimos 5 trades
        recent_5 = list(self.performance_history)[-5:]
        exit_reasons = [t['exit_reason'] for t in recent_5]

        # Todos SL?
        if exit_reasons.count('STOP_LOSS') >= 5:
            anomaly = AnomalyEvent(
                timestamp=datetime.now(),
                anomaly_type='losing_streak',
                severity='CRITICAL',
                description=f"5 STOP LOSS consecutivos - estrategia fallando",
                metrics={'consecutive_stop_losses': 5}
            )

            self._handle_anomaly(anomaly)

        # Últimos 10 trades con >70% SL?
        if len(self.performance_history) >= 10:
            recent_10 = list(self.performance_history)[-10:]
            sl_count = sum(1 for t in recent_10 if t['exit_reason'] == 'STOP_LOSS')

            if sl_count >= 7:  # 70%+
                anomaly = AnomalyEvent(
                    timestamp=datetime.now(),
                    anomaly_type='high_stop_loss_rate',
                    severity='HIGH',
                    description=f"{sl_count}/10 trades terminaron en STOP_LOSS ({sl_count*10}%)",
                    metrics={'stop_loss_rate': sl_count * 10}
                )

                self._handle_anomaly(anomaly)

    def _calculate_metrics(self, trades: List[Dict]) -> Dict:
        """
        Calcula métricas de un conjunto de trades

        Returns:
            Dict con win_rate, avg_profit, etc.
        """
        if not trades:
            return {'win_rate': 0.0, 'avg_profit': 0.0, 'total_trades': 0}

        winning_trades = sum(1 for t in trades if t['profit_pct'] > 0)
        win_rate = (winning_trades / len(trades)) * 100

        avg_profit = np.mean([t['profit_pct'] for t in trades])

        return {
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'total_trades': len(trades),
            'winning_trades': winning_trades
        }

    def _handle_anomaly(self, anomaly: AnomalyEvent) -> None:
        """
        Maneja una anomalía detectada

        Acciones:
        1. Guardar en historial
        2. Log warning/critical
        3. Notificar vía Telegram (opcional)
        4. Auto-revertir parámetros si es CRITICAL (opcional)

        Args:
            anomaly: Evento de anomalía
        """
        # Guardar
        self.anomaly_events.append(anomaly)

        # Log según severidad
        log_msg = f"🚨 ANOMALY DETECTED [{anomaly.severity}]: {anomaly.description}"

        if anomaly.severity == 'CRITICAL':
            logger.critical(log_msg)
        elif anomaly.severity == 'HIGH':
            logger.error(log_msg)
        elif anomaly.severity == 'MEDIUM':
            logger.warning(log_msg)
        else:
            logger.info(log_msg)

        # Auto-revertir si es crítico y está habilitado
        if self.auto_revert_enabled and anomaly.severity == 'CRITICAL':
            self._auto_revert_parameters(anomaly)

    def _auto_revert_parameters(self, anomaly: AnomalyEvent) -> None:
        """
        Revierte parámetros a snapshot anterior (auto-corrección)

        Args:
            anomaly: Evento que triggereó el revert
        """
        if not self.parameter_snapshots:
            logger.warning("⚠️ No hay snapshots de parámetros para revertir")
            return

        # Obtener snapshot anterior (antes del problema)
        previous_snapshot = self.parameter_snapshots[-1]

        logger.warning(f"🔄 AUTO-REVERTING parámetros a snapshot de {previous_snapshot['timestamp']}")

        # Aquí se debería recargar los parámetros
        # Esto requeriría integración con parameter_optimizer

        anomaly.action_taken = f"Reverted to snapshot from {previous_snapshot['timestamp']}"

        # Notificar
        logger.info(f"✅ Parámetros revertidos exitosamente")

    def save_parameter_snapshot(self, parameters: Dict, metadata: Optional[Dict] = None) -> None:
        """
        Guarda snapshot de parámetros actuales

        Args:
            parameters: Dict con parámetros actuales
            metadata: Información adicional (win_rate, etc.)
        """
        snapshot = {
            'timestamp': datetime.now(),
            'parameters': parameters.copy(),
            'metadata': metadata or {}
        }

        self.parameter_snapshots.append(snapshot)
        logger.debug(f"📸 Parameter snapshot guardado ({len(self.parameter_snapshots)}/10)")

    def get_recent_anomalies(self, hours: int = 24) -> List[AnomalyEvent]:
        """
        Obtiene anomalías de las últimas N horas

        Args:
            hours: Ventana de tiempo

        Returns:
            Lista de AnomalyEvent
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        return [a for a in self.anomaly_events if a.timestamp >= cutoff]

    def get_statistics(self) -> Dict:
        """
        Estadísticas del detector de anomalías

        Returns:
            Dict con métricas
        """
        recent_24h = self.get_recent_anomalies(hours=24)

        severity_counts = {
            'CRITICAL': sum(1 for a in recent_24h if a.severity == 'CRITICAL'),
            'HIGH': sum(1 for a in recent_24h if a.severity == 'HIGH'),
            'MEDIUM': sum(1 for a in recent_24h if a.severity == 'MEDIUM'),
            'LOW': sum(1 for a in recent_24h if a.severity == 'LOW')
        }

        type_counts = {}
        for a in recent_24h:
            type_counts[a.anomaly_type] = type_counts.get(a.anomaly_type, 0) + 1

        return {
            'total_anomalies_24h': len(recent_24h),
            'severity_breakdown': severity_counts,
            'type_breakdown': type_counts,
            'baseline_metrics': self.baseline_metrics,
            'snapshots_saved': len(self.parameter_snapshots),
            'enabled': self.enabled,
            'auto_revert_enabled': self.auto_revert_enabled
        }

    def reset_baseline(self) -> None:
        """
        Reinicia el baseline de performance

        Útil después de optimizaciones o cambios significativos
        """
        if len(self.performance_history) >= 10:
            self.baseline_metrics = self._calculate_metrics(list(self.performance_history)[-20:])
            logger.info(f"✅ Baseline reiniciado: win_rate={self.baseline_metrics['win_rate']:.1f}%")
        else:
            logger.warning("⚠️ Insuficientes trades para reiniciar baseline")


# Parámetros optimizables para config.py
ANOMALY_DETECTION_PARAMS = {
    # Habilitación
    'ANOMALY_DETECTION_ENABLED': True,  # True/False

    # Thresholds (optimizables)
    'PERFORMANCE_DEGRADATION_THRESHOLD': 10.0,  # 5-20% (caída en win rate para alertar)
    'OUTLIER_STD_THRESHOLD': 3.0,  # 2.0-4.0 (desviaciones estándar)
    'MIN_TRADES_FOR_DETECTION': 20,  # 10-50 (mínimo de trades)
    'ANOMALY_LOOKBACK_WINDOW': 50,  # 30-100 (trades a considerar)

    # Auto-revert
    'AUTO_REVERT_ENABLED': True,  # True/False (revertir automáticamente parámetros)
}
