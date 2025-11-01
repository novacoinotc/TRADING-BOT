"""
ML Integration - Capa de integración que coordina todos los componentes ML
Conecta Predictor, Trainer, Optimizer y Paper Trader
"""
import logging
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
import json

from src.ml.predictor import MLPredictor
from src.ml.model_trainer import ModelTrainer
from src.ml.feature_engineer import FeatureEngineer
from src.ml.optimizer import AutoOptimizer
from src.trading.paper_trader import PaperTrader

logger = logging.getLogger(__name__)


class MLIntegration:
    """
    Sistema completo de ML + Paper Trading
    - Predice señales con ML
    - Ejecuta trades en paper trading
    - Entrena modelos automáticamente
    - Optimiza parámetros continuamente
    """

    def __init__(self, initial_balance: float = 50000.0, enable_ml: bool = True):
        """
        Args:
            initial_balance: Balance inicial en USDT
            enable_ml: Habilitar predicciones ML (False = solo paper trading sin ML)
        """
        # Componentes ML
        self.predictor = MLPredictor()
        self.trainer = ModelTrainer(min_samples_for_training=50)
        self.feature_engineer = FeatureEngineer()
        self.optimizer = AutoOptimizer()

        # Paper Trading
        self.paper_trader = PaperTrader(initial_balance=initial_balance)

        # Estado
        self.enable_ml = enable_ml
        self.trades_since_last_training = 0
        self.trades_since_last_optimization = 0
        self.last_trained_samples = 0
        self.last_optimization_trades = 0

        # Almacenamiento de features y señales para entrenamiento
        self.training_buffer = []
        self.data_dir = Path('data/ml')
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.buffer_file = self.data_dir / 'training_buffer.json'

        # Cargar buffer si existe
        self._load_buffer()

        logger.info("🚀 ML Integration System inicializado")
        logger.info(f"   ML Enabled: {enable_ml}")
        logger.info(f"   Initial Balance: ${initial_balance:,.2f} USDT")

    def process_signal(
        self,
        pair: str,
        signal: Dict,
        indicators: Dict,
        current_price: float,
        mtf_indicators: Dict = None,
        sentiment_features: Dict = None,
        orderbook_features: Dict = None,
        regime_features: Dict = None
    ) -> Optional[Dict]:
        """
        Procesa señal completa con ML + Paper Trading

        Args:
            pair: Par de trading
            signal: Señal original
            indicators: Indicadores técnicos
            current_price: Precio actual
            mtf_indicators: Indicadores multi-timeframe
            sentiment_features: Features de sentiment analysis
            orderbook_features: Features de order book analysis
            regime_features: Features de market regime detection

        Returns:
            Resultado del trade o None
        """
        # 1. Enriquecer señal con predicción ML
        if self.enable_ml:
            enhanced_signal = self.predictor.enhance_signal(
                signal=signal,
                indicators=indicators,
                mtf_indicators=mtf_indicators,
                sentiment_features=sentiment_features,
                orderbook_features=orderbook_features,
                regime_features=regime_features
            )

            # Verificar si ML recomienda tradear
            if not self.predictor.should_trade(enhanced_signal):
                logger.info(f"⚠️ Trade en {pair} bloqueado por ML")
                return None
        else:
            enhanced_signal = signal.copy()

        # 2. Aplicar parámetros optimizados
        optimized_params = self.optimizer.get_current_params()
        enhanced_signal['optimized_params'] = optimized_params

        # 3. Ejecutar en paper trading
        trade_result = self.paper_trader.process_signal(
            pair=pair,
            signal=enhanced_signal,
            current_price=current_price
        )

        # 4. Si se abrió un trade, guardar features para entrenamiento futuro
        if trade_result and trade_result.get('status') == 'OPEN':
            self._save_signal_for_training(
                pair=pair,
                signal=enhanced_signal,
                indicators=indicators,
                mtf_indicators=mtf_indicators,
                sentiment_features=sentiment_features,
                orderbook_features=orderbook_features,
                regime_features=regime_features,
                entry_price=current_price,
                trade_id=trade_result.get('trade_id')
            )

        # 5. Si se cerró un trade, actualizar contadores y verificar si reentrenar
        if trade_result and trade_result.get('status') == 'CLOSED':
            self.trades_since_last_training += 1
            self.trades_since_last_optimization += 1

            # Verificar si reentrenar modelo
            stats = self.paper_trader.get_statistics()
            total_trades = stats['total_trades']

            if self.trainer.should_retrain(total_trades, self.last_trained_samples):
                logger.info("🧠 Iniciando reentrenamiento de modelo ML...")
                self._retrain_model()

            # Verificar si optimizar parámetros
            if self.optimizer.should_optimize(total_trades, self.trades_since_last_optimization):
                logger.info("🤖 Iniciando optimización de parámetros...")
                self._optimize_parameters()

        return trade_result

    def update_position(self, pair: str, current_price: float) -> Optional[Dict]:
        """
        Actualiza posición existente

        Args:
            pair: Par de trading
            current_price: Precio actual

        Returns:
            Trade cerrado si alcanzó SL/TP, None otherwise
        """
        result = self.paper_trader.update_position(pair, current_price)

        # Si se cerró trade, incrementar contadores
        if result and result.get('status') == 'CLOSED':
            self.trades_since_last_training += 1
            self.trades_since_last_optimization += 1

            # Verificar reentrenamiento y optimización
            stats = self.paper_trader.get_statistics()
            total_trades = stats['total_trades']

            if self.trainer.should_retrain(total_trades, self.last_trained_samples):
                self._retrain_model()

            if self.optimizer.should_optimize(total_trades, self.trades_since_last_optimization):
                self._optimize_parameters()

        return result

    def _retrain_model(self):
        """Reentrena modelo ML con trades cerrados"""
        try:
            # Obtener trades cerrados
            closed_trades = self.paper_trader.get_closed_trades(limit=500)

            if len(closed_trades) < self.trainer.min_samples:
                logger.warning(f"Insuficientes trades para reentrenar: {len(closed_trades)}")
                return

            # Cargar features correspondientes desde buffer
            features_list = self._get_features_for_trades(closed_trades)

            if len(features_list) != len(closed_trades):
                logger.error(f"Mismatch: {len(closed_trades)} trades vs {len(features_list)} features")
                return

            # Preparar datos
            X, y = self.trainer.prepare_training_data(closed_trades, features_list)

            # Entrenar
            metrics = self.trainer.train(X, y)

            if metrics:
                logger.info("✅ Modelo reentrenado exitosamente!")
                logger.info(f"   Test Accuracy: {metrics['test_accuracy']:.3f}")
                logger.info(f"   Test Precision: {metrics['test_precision']:.3f}")
                logger.info(f"   Test F1: {metrics['test_f1']:.3f}")

                # Actualizar contador
                self.last_trained_samples = len(closed_trades)
                self.trades_since_last_training = 0

        except Exception as e:
            logger.error(f"Error reentrenando modelo: {e}")

    def _optimize_parameters(self):
        """Optimiza parámetros del bot"""
        try:
            stats = self.paper_trader.get_statistics()

            adjustments = self.optimizer.optimize(stats)

            if adjustments:
                logger.info(f"✅ Parámetros optimizados: {len(adjustments)} ajustes")

                # Actualizar contador
                self.trades_since_last_optimization = 0
                self.last_optimization_trades = stats['total_trades']

        except Exception as e:
            logger.error(f"Error optimizando parámetros: {e}")

    def _save_signal_for_training(
        self,
        pair: str,
        signal: Dict,
        indicators: Dict,
        mtf_indicators: Dict,
        sentiment_features: Dict,
        orderbook_features: Dict,
        regime_features: Dict,
        entry_price: float,
        trade_id: str
    ):
        """Guarda señal y features para entrenamiento futuro"""
        try:
            # Crear features (ahora incluye sentiment + orderbook + regime)
            features = self.feature_engineer.create_features(
                indicators=indicators,
                signals=signal,
                mtf_indicators=mtf_indicators,
                sentiment_features=sentiment_features,
                orderbook_features=orderbook_features,
                regime_features=regime_features
            )

            # Guardar en buffer
            record = {
                'trade_id': trade_id,
                'pair': pair,
                'timestamp': datetime.now().isoformat(),
                'entry_price': entry_price,
                'features': features
            }

            self.training_buffer.append(record)

            # Guardar a disco cada 10 nuevas señales
            if len(self.training_buffer) % 10 == 0:
                self._save_buffer()

        except Exception as e:
            logger.error(f"Error guardando señal para entrenamiento: {e}")

    def _get_features_for_trades(self, trades: List[Dict]) -> List[Dict]:
        """Recupera features correspondientes a trades cerrados"""
        features_list = []

        for trade in trades:
            trade_id = trade.get('trade_id')

            # Buscar features en buffer
            for record in self.training_buffer:
                if record.get('trade_id') == trade_id:
                    features_list.append(record['features'])
                    break
            else:
                # Si no se encuentra, crear features vacías (esto no debería pasar)
                logger.warning(f"Features no encontradas para trade {trade_id}")

        return features_list

    def _save_buffer(self):
        """Guarda buffer de entrenamiento a disco"""
        try:
            # Mantener solo últimas 1000 señales
            if len(self.training_buffer) > 1000:
                self.training_buffer = self.training_buffer[-1000:]

            with open(self.buffer_file, 'w') as f:
                json.dump(self.training_buffer, f, indent=2)

            logger.debug(f"Buffer guardado: {len(self.training_buffer)} señales")

        except Exception as e:
            logger.error(f"Error guardando buffer: {e}")

    def _load_buffer(self):
        """Carga buffer de entrenamiento desde disco"""
        if not self.buffer_file.exists():
            return

        try:
            with open(self.buffer_file, 'r') as f:
                self.training_buffer = json.load(f)

            logger.info(f"📊 Buffer cargado: {len(self.training_buffer)} señales históricas")

        except Exception as e:
            logger.error(f"Error cargando buffer: {e}")

    def get_comprehensive_stats(self) -> Dict:
        """Retorna estadísticas completas del sistema"""
        # Stats de paper trading
        trading_stats = self.paper_trader.get_statistics()

        # Stats de ML
        model_info = self.trainer.get_model_info()

        # Parámetros optimizados
        optimized_params = self.optimizer.get_current_params()
        optimization_history = self.optimizer.get_optimization_history(limit=5)

        return {
            'trading': trading_stats,
            'ml_model': model_info,
            'optimized_params': optimized_params,
            'recent_optimizations': optimization_history,
            'training_buffer_size': len(self.training_buffer),
            'trades_since_last_training': self.trades_since_last_training,
            'trades_since_last_optimization': self.trades_since_last_optimization,
            'ml_enabled': self.enable_ml
        }

    def get_performance_report(self) -> str:
        """Genera reporte completo de performance"""
        stats = self.get_comprehensive_stats()

        trading = stats['trading']
        ml_model = stats['ml_model']
        params = stats['optimized_params']

        report = f"""
🤖 **ML TRADING BOT - PERFORMANCE REPORT**

💰 **PAPER TRADING**
Balance Inicial: ${trading['initial_balance']:,.2f} USDT
Balance Actual: ${trading['current_balance']:,.2f} USDT
Equity Total: ${trading['equity']:,.2f} USDT
P&L Neto: ${trading['net_pnl']:,.2f} ({trading['roi']:+.2f}%)

📊 **ESTADÍSTICAS DE TRADING**
Total Trades: {trading['total_trades']}
Posiciones Abiertas: {trading['open_positions']}
Win Rate: {trading['win_rate']:.1f}%
Profit Factor: {trading['profit_factor']:.2f}
Sharpe Ratio: {trading['sharpe_ratio']:.2f}
Max Drawdown: {trading['max_drawdown']:.2f}%

🧠 **MACHINE LEARNING**
Modelo Disponible: {'✅ Sí' if ml_model.get('available') else '❌ No'}
"""

        if ml_model.get('available'):
            metrics = ml_model.get('metrics', {})
            report += f"""Test Accuracy: {metrics.get('test_accuracy', 0):.3f}
Test Precision: {metrics.get('test_precision', 0):.3f}
Test F1 Score: {metrics.get('test_f1', 0):.3f}
Samples Entrenados: {metrics.get('samples_total', 0)}
"""

        report += f"""
⚙️ **PARÁMETROS OPTIMIZADOS**
Flash Threshold: {params.get('flash_threshold', 0):.1f}
Min Confidence: {params.get('flash_min_confidence', 0)}%
Position Size: {params.get('position_size_pct', 0):.1f}%
Max Positions: {params.get('max_positions', 0)}

📈 **ACTIVIDAD RECIENTE**
Trades desde último entrenamiento: {stats['trades_since_last_training']}
Trades desde última optimización: {stats['trades_since_last_optimization']}
Señales en buffer: {stats['training_buffer_size']}
"""

        return report.strip()

    def enable_ml_predictions(self):
        """Habilita predicciones ML"""
        self.enable_ml = True
        self.predictor.enable()
        logger.info("✅ Predicciones ML habilitadas")

    def disable_ml_predictions(self):
        """Deshabilita predicciones ML (solo paper trading)"""
        self.enable_ml = False
        self.predictor.disable()
        logger.info("❌ Predicciones ML deshabilitadas")

    def force_retrain(self):
        """Fuerza reentrenamiento inmediato"""
        logger.info("🔄 Forzando reentrenamiento de modelo...")
        self._retrain_model()

    def force_optimize(self):
        """Fuerza optimización inmediata"""
        logger.info("🔄 Forzando optimización de parámetros...")
        self._optimize_parameters()
