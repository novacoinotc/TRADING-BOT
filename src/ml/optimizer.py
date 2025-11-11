"""
Auto-Optimizer - Ajusta parámetros del bot automáticamente basado en performance
La IA aprende y optimiza sus propios parámetros
"""
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict
from config import config

logger = logging.getLogger(__name__)


class AutoOptimizer:
    """
    Optimiza parámetros automáticamente basado en resultados
    - Ajusta umbrales de señales
    - Modifica pesos de indicadores
    - Adapta risk management
    - Aprende de los errores
    """

    def __init__(self):
        # Parámetros optimizables
        self.params = {
            'flash_threshold': config.FLASH_THRESHOLD,
            'flash_min_confidence': config.FLASH_MIN_CONFIDENCE,
            'conservative_threshold': config.CONSERVATIVE_THRESHOLD,
            'position_size_pct': 5.0,
            'max_positions': 10,
        }

        # Histórico de ajustes
        self.adjustments_history = []

        # Archivos
        self.data_dir = Path('data/optimization')
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.params_file = self.data_dir / 'optimized_params.json'
        self.history_file = self.data_dir / 'optimization_history.json'

        # Cargar parámetros si existen
        self._load_params()

        # Contadores para decisiones
        self.optimization_count = 0
        self.last_stats = {}

    def optimize(self, stats: Dict) -> Dict:
        """
        Optimiza parámetros basado en estadísticas de performance

        Args:
            stats: Estadísticas del portfolio

        Returns:
            Dict con ajustes realizados
        """
        self.optimization_count += 1
        adjustments = {}

        # Solo optimizar si hay suficientes datos
        if stats['total_trades'] < 20:
            logger.debug(f"Insuficientes trades para optimizar: {stats['total_trades']} < 20")
            return adjustments

        win_rate = stats['win_rate']
        roi = stats['roi']
        profit_factor = stats['profit_factor']
        drawdown = stats['max_drawdown']

        logger.info("🤖 Auto-optimizer analizando performance...")

        # === OPTIMIZACIÓN 1: FLASH THRESHOLD ===
        adjustments.update(self._optimize_flash_threshold(win_rate, profit_factor, stats))

        # === OPTIMIZACIÓN 2: CONFIDENCE THRESHOLD ===
        adjustments.update(self._optimize_confidence(win_rate, stats))

        # === OPTIMIZACIÓN 3: POSITION SIZING ===
        adjustments.update(self._optimize_position_size(roi, drawdown, win_rate))

        # === OPTIMIZACIÓN 4: MAX POSITIONS ===
        adjustments.update(self._optimize_max_positions(stats))

        # Guardar ajustes
        if adjustments:
            self._record_adjustment(adjustments, stats)
            self._save_params()

            logger.info(f"✅ Optimización completada: {len(adjustments)} parámetros ajustados")
            for param, change in adjustments.items():
                logger.info(f"   {param}: {change['from']} → {change['to']} ({change['reason']})")

        self.last_stats = stats.copy()

        return adjustments

    def _optimize_flash_threshold(self, win_rate: float, profit_factor: float, stats: Dict) -> Dict:
        """Optimiza umbral de señales flash"""
        adjustments = {}
        current = self.params['flash_threshold']

        # Si win rate es bajo, AUMENTAR umbral (ser más selectivos)
        if win_rate < 45 and stats['total_trades'] > 30:
            new_value = min(current + 0.5, 7.0)
            if new_value != current:
                adjustments['flash_threshold'] = {
                    'from': current,
                    'to': new_value,
                    'reason': f'Win rate bajo ({win_rate:.1f}%), aumentando selectividad'
                }
                self.params['flash_threshold'] = new_value

        # Si win rate es alto y profit factor alto, REDUCIR umbral (más señales)
        elif win_rate > 65 and profit_factor > 2.0 and stats['total_trades'] > 50:
            new_value = max(current - 0.3, 4.0)
            if new_value != current:
                adjustments['flash_threshold'] = {
                    'from': current,
                    'to': new_value,
                    'reason': f'Win rate alto ({win_rate:.1f}%), generando más señales'
                }
                self.params['flash_threshold'] = new_value

        return adjustments

    def _optimize_confidence(self, win_rate: float, stats: Dict) -> Dict:
        """Optimiza umbral de confianza"""
        adjustments = {}
        current = self.params['flash_min_confidence']

        # Si win rate bajo, AUMENTAR confianza requerida
        if win_rate < 40 and stats['total_trades'] > 30:
            new_value = min(current + 5, 70)
            if new_value != current:
                adjustments['flash_min_confidence'] = {
                    'from': current,
                    'to': new_value,
                    'reason': f'Win rate muy bajo ({win_rate:.1f}%), aumentando confianza mínima'
                }
                self.params['flash_min_confidence'] = new_value

        # Si win rate muy alto, REDUCIR confianza (más señales)
        elif win_rate > 70 and stats['total_trades'] > 50:
            new_value = max(current - 5, 40)
            if new_value != current:
                adjustments['flash_min_confidence'] = {
                    'from': current,
                    'to': new_value,
                    'reason': f'Win rate excelente ({win_rate:.1f}%), reduciendo confianza mínima'
                }
                self.params['flash_min_confidence'] = new_value

        return adjustments

    def _optimize_position_size(self, roi: float, drawdown: float, win_rate: float) -> Dict:
        """Optimiza tamaño de posiciones"""
        adjustments = {}
        current = self.params['position_size_pct']

        # Si drawdown alto, REDUCIR tamaño
        if drawdown > 15:
            new_value = max(current - 1.0, 2.0)
            if new_value != current:
                adjustments['position_size_pct'] = {
                    'from': current,
                    'to': new_value,
                    'reason': f'Drawdown alto ({drawdown:.1f}%), reduciendo riesgo'
                }
                self.params['position_size_pct'] = new_value

        # Si ROI muy positivo y drawdown bajo, AUMENTAR tamaño
        elif roi > 10 and drawdown < 5 and win_rate > 60:
            new_value = min(current + 0.5, 8.0)
            if new_value != current:
                adjustments['position_size_pct'] = {
                    'from': current,
                    'to': new_value,
                    'reason': f'Performance excelente (ROI {roi:.1f}%), aumentando tamaño'
                }
                self.params['position_size_pct'] = new_value

        return adjustments

    def _optimize_max_positions(self, stats: Dict) -> Dict:
        """Optimiza número máximo de posiciones simultáneas"""
        adjustments = {}
        current = self.params['max_positions']

        # Si muchas posiciones abiertas consistentemente, podría aumentar
        open_positions = stats.get('open_positions', 0)

        # Por ahora, mantener conservador
        return adjustments

    def _record_adjustment(self, adjustments: Dict, stats: Dict):
        """Registra ajustes en historial"""
        record = {
            'timestamp': datetime.now().isoformat(),
            'optimization_number': self.optimization_count,
            'adjustments': adjustments,
            'stats_before': {
                'total_trades': stats['total_trades'],
                'win_rate': stats['win_rate'],
                'roi': stats['roi'],
                'max_drawdown': stats['max_drawdown'],
                'profit_factor': stats['profit_factor']
            }
        }

        self.adjustments_history.append(record)

        # Guardar solo últimas 50 optimizaciones
        if len(self.adjustments_history) > 50:
            self.adjustments_history = self.adjustments_history[-50:]

        # Guardar en disco
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.adjustments_history, f, indent=2)
        except Exception as e:
            logger.error(f"Error guardando historial de optimización: {e}")

    def _save_params(self):
        """Guarda parámetros optimizados"""
        data = {
            'params': self.params,
            'last_updated': datetime.now().isoformat(),
            'optimization_count': self.optimization_count
        }

        try:
            with open(self.params_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"💾 Parámetros optimizados guardados")

        except Exception as e:
            logger.error(f"Error guardando parámetros: {e}")

    def _load_params(self):
        """Carga parámetros optimizados si existen"""
        if not self.params_file.exists():
            return

        try:
            with open(self.params_file, 'r') as f:
                data = json.load(f)

            self.params = data.get('params', self.params)
            self.optimization_count = data.get('optimization_count', 0)

            logger.info(f"📊 Parámetros optimizados cargados")
            logger.info(f"   Flash Threshold: {self.params['flash_threshold']}")
            logger.info(f"   Min Confidence: {self.params['flash_min_confidence']}%")
            logger.info(f"   Position Size: {self.params['position_size_pct']}%")

        except Exception as e:
            logger.error(f"Error cargando parámetros: {e}")

    def get_current_params(self) -> Dict:
        """Retorna parámetros actuales optimizados"""
        return self.params.copy()

    def get_optimization_history(self, limit: int = 10) -> list:
        """Retorna historial de optimizaciones"""
        return self.adjustments_history[-limit:]

    def should_optimize(self, new_trades_count: int, trades_since_last_optimization: int) -> bool:
        """
        Determina si es momento de optimizar

        Args:
            new_trades_count: Total de trades
            trades_since_last_optimization: Trades desde última optimización

        Returns:
            True si debe optimizar
        """
        # Optimizar cada 20 trades
        if trades_since_last_optimization >= 20:
            return True

        # Optimizar si hay suficientes trades iniciales
        if new_trades_count >= 30 and self.optimization_count == 0:
            return True

        return False
