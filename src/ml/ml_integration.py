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

logger = logging.getLogger(__name__)


class MLIntegration:
    """
    Sistema completo de ML (v2.0: con o sin Paper Trading)
    - Predice señales con ML
    - Mejora/valida señales con ML Predictor
    - v2.0: Ejecuta trades en Binance Futures (o Paper Trading si initial_balance != None)
    - Entrena modelos automáticamente
    - Optimiza parámetros continuamente
    """

    def __init__(self, initial_balance: float = None, enable_ml: bool = True, telegram_notifier=None):
        """
        Args:
            initial_balance: Balance inicial en USDT (None = sin paper trading, solo ML en v2.0)
            enable_ml: Habilitar predicciones ML (False = solo paper trading sin ML)
            telegram_notifier: Notificador de Telegram para enviar alertas de trades
        """
        # Componentes ML
        self.predictor = MLPredictor()
        self.trainer = ModelTrainer(min_samples_for_training=30)  # Reducido de 50 a 30
        self.feature_engineer = FeatureEngineer()
        self.optimizer = AutoOptimizer()

        # Paper Trading (opcional en v2.0 - import condicional)
        if initial_balance is not None:
            try:
                from src.trading.paper_trader import PaperTrader
                self.paper_trader = PaperTrader(initial_balance=initial_balance)
                logger.info(f"   Paper Trading habilitado: ${initial_balance:,.2f} USDT")
            except ImportError:
                logger.warning("⚠️ PaperTrader no disponible (eliminado en v2.0), usando solo Binance")
                self.paper_trader = None
        else:
            self.paper_trader = None
            logger.info("   Paper Trading deshabilitado (v2.0: usando Binance real)")

        # Telegram notifier para alertas de trades
        self.telegram_notifier = telegram_notifier

        # Estado
        self.enable_ml = enable_ml
        self.trades_since_last_training = 0
        self.trades_since_last_optimization = 0
        self.last_trained_samples = 0
        self.last_optimization_trades = 0

        # Almacenamiento de features y señales para entrenamiento
        self.training_buffer = []
        self.imported_features = {}  # Features de trades importados (fallback)
        self.data_dir = Path('data/ml')
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.buffer_file = self.data_dir / 'training_buffer.json'

        # Cargar buffer si existe
        self._load_buffer()

        logger.info("🚀 ML Integration System inicializado")
        logger.info(f"   ML Enabled: {enable_ml}")

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

        # 3. Ejecutar en paper trading (v2.0: solo si paper_trader está disponible)
        trade_result = None
        if self.paper_trader:
            trade_result = self.paper_trader.process_signal(
                pair=pair,
                signal=enhanced_signal,
                current_price=current_price
            )
        else:
            # v2.0: Sin paper trader, retornar señal mejorada para que market_monitor
            # la ejecute con Futures Trader
            logger.debug(f"📊 ML enhanced signal for {pair} (no paper trading)")

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

            # Notificar apertura de trade por Telegram
            if self.telegram_notifier:
                try:
                    import asyncio
                    asyncio.create_task(
                        self.telegram_notifier.send_trade_opened(trade_result)
                    )
                except Exception as e:
                    logger.error(f"Error enviando notificación de trade abierto: {e}")

        # 5. Si se cerró un trade, actualizar contadores y verificar si reentrenar
        if trade_result and trade_result.get('status') == 'CLOSED':
            self.trades_since_last_training += 1
            self.trades_since_last_optimization += 1

            # Notificar cierre de trade por Telegram
            if self.telegram_notifier:
                try:
                    import asyncio
                    asyncio.create_task(
                        self.telegram_notifier.send_trade_closed(trade_result)
                    )
                except Exception as e:
                    logger.error(f"Error enviando notificación de trade cerrado: {e}")

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
        Actualiza posición existente (v2.0: solo si paper_trader está disponible)

        Args:
            pair: Par de trading
            current_price: Precio actual

        Returns:
            Trade cerrado si alcanzó SL/TP, None otherwise
        """
        if not self.paper_trader:
            # v2.0: Sin paper trader, las posiciones son manejadas por PositionMonitor
            return None

        result = self.paper_trader.update_position(pair, current_price)

        # Si se cerró trade, incrementar contadores
        if result and result.get('status') == 'CLOSED':
            self.trades_since_last_training += 1
            self.trades_since_last_optimization += 1

            # Notificar cierre de trade por Telegram
            if self.telegram_notifier:
                try:
                    import asyncio
                    asyncio.create_task(
                        self.telegram_notifier.send_trade_closed(result)
                    )
                except Exception as e:
                    logger.error(f"Error enviando notificación de trade cerrado: {e}")

            # Verificar reentrenamiento y optimización
            stats = self.paper_trader.get_statistics()
            total_trades = stats['total_trades']

            if self.trainer.should_retrain(total_trades, self.last_trained_samples):
                self._retrain_model()

            if self.optimizer.should_optimize(total_trades, self.trades_since_last_optimization):
                self._optimize_parameters()

        return result

    def _retrain_model(self, external_paper_trader=None):
        """
        Reentrena modelo ML con trades cerrados

        Args:
            external_paper_trader: Paper trader externo para usar sus trades
                                  Si es None, usa self.paper_trader
        """
        try:
            # Intentar obtener datos de múltiples fuentes
            training_data = []

            # Opción 1: ml_training_data (estándar)
            if hasattr(self, 'ml_training_data') and self.ml_training_data:
                training_data = self.ml_training_data
                logger.info(f"📊 Usando ml_training_data: {len(training_data)} muestras")

            # Opción 2: training_buffer
            elif hasattr(self, 'training_buffer') and self.training_buffer:
                training_data = self.training_buffer
                logger.info(f"📊 Usando training_buffer: {len(training_data)} muestras")

            # Opción 3: Desde autonomy_controller via RL memory
            elif hasattr(self, 'autonomy_controller') and self.autonomy_controller:
                rl_agent = getattr(self.autonomy_controller, 'rl_agent', None)
                if rl_agent:
                    rl_memory = getattr(rl_agent, 'memory', [])
                    if rl_memory:
                        # Convertir RL memory a formato ML
                        for exp in rl_memory:
                            training_data.append({
                                'features': exp.get('state', ''),
                                'reward': exp.get('reward', 0),
                                'success': exp.get('reward', 0) > 0,
                                'action': exp.get('action', 'SKIP')
                            })
                        logger.info(f"📊 Convertido de RL memory: {len(training_data)} muestras")

            # Opción 4 (fallback): Paper trader trades
            if not training_data:
                # Usar paper_trader externo si se proporcionó, sino usar el interno
                paper_trader = external_paper_trader if external_paper_trader else self.paper_trader

                # DEBUG: Verificar estado del portfolio
                if hasattr(paper_trader, 'portfolio'):
                    portfolio_stats = paper_trader.portfolio.get_statistics()
                    logger.info(f"📊 Portfolio stats: total_trades={portfolio_stats.get('total_trades', 0)}")
                    logger.info(f"📊 Portfolio closed_trades length: {len(paper_trader.portfolio.closed_trades)}")

                # Obtener trades cerrados
                training_data = paper_trader.get_closed_trades(limit=500)
                logger.info(f"📊 Trades obtenidos para entrenamiento: {len(training_data)}")

            if not training_data:
                logger.error("❌ No se encontraron datos en ninguna fuente")
                return

            if len(training_data) < self.trainer.min_samples:
                logger.warning(f"Insuficientes trades para reentrenar: {len(training_data)}")
                logger.warning(f"  Requerido: {self.trainer.min_samples}")
                return

            logger.info(f"✅ Datos encontrados: {len(training_data)} muestras para entrenar ML")

            # Cargar features correspondientes desde buffer
            features_list = self._get_features_for_trades(training_data)

            if len(features_list) != len(training_data):
                logger.error(f"Mismatch: {len(training_data)} trades vs {len(features_list)} features")
                return

            # Preparar datos
            X, y = self.trainer.prepare_training_data(training_data, features_list)

            # Entrenar
            metrics = self.trainer.train(X, y)

            if metrics:
                logger.info("✅ Modelo reentrenado exitosamente!")
                logger.info(f"   Test Accuracy: {metrics['test_accuracy']:.3f}")
                logger.info(f"   Test Precision: {metrics['test_precision']:.3f}")
                logger.info(f"   Test F1: {metrics['test_f1']:.3f}")

                # Actualizar contador
                self.last_trained_samples = len(training_data)
                self.trades_since_last_training = 0

        except Exception as e:
            logger.error(f"Error reentrenando modelo: {e}")

    def _optimize_parameters(self):
        """Optimiza parámetros del bot (v2.0: requiere stats externos si no hay paper_trader)"""
        try:
            if not self.paper_trader:
                logger.warning("⚠️ No se puede optimizar sin paper_trader (TODO: usar stats de Binance)")
                return

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
        """
        Recupera features correspondientes a trades cerrados
        Con sistema de fallback robusto para máxima compatibilidad
        """
        features_list = []

        for trade in trades:
            trade_id = trade.get('trade_id')
            features = None

            # FALLBACK 1: Buscar en training_buffer
            for record in self.training_buffer:
                if record.get('trade_id') == trade_id:
                    features = record['features']
                    logger.debug(f"✓ Features encontradas para {trade_id} en training_buffer")
                    break

            # FALLBACK 2: Buscar directamente en el trade (features embebidas)
            if features is None and 'features' in trade:
                features = trade['features']
                logger.debug(f"✓ Features encontradas para {trade_id} embebidas en trade")

            # FALLBACK 3: Buscar en imported_features (datos importados)
            if features is None and hasattr(self, 'imported_features') and trade_id in self.imported_features:
                features = self.imported_features[trade_id]
                logger.debug(f"✓ Features encontradas para {trade_id} en imported_features")

            # FALLBACK 4: Intentar reconstruir features básicas desde el trade
            if features is None:
                logger.warning(f"⚠️ Features no encontradas para trade {trade_id} - intentando reconstruir")
                features = self._reconstruct_basic_features(trade)

            if features:
                features_list.append(features)
            else:
                logger.error(f"❌ No se pudieron obtener features para trade {trade_id}")

        logger.info(f"📊 Features recuperadas: {len(features_list)}/{len(trades)} trades")
        return features_list

    def _reconstruct_basic_features(self, trade: Dict) -> Optional[Dict]:
        """
        Intenta reconstruir features básicas desde la información del trade
        Usado como último recurso cuando no hay features guardadas
        """
        try:
            # Extraer información básica del trade
            side = trade.get('side', 'BUY')
            pnl = trade.get('pnl', 0)
            pnl_pct = trade.get('pnl_pct', 0)

            # Crear features básicas (placeholder - no son tan precisas como las originales)
            basic_features = {
                'price': trade.get('entry_price', 0),
                'volume_24h': 0,  # No disponible
                'rsi_1h': 50,  # Neutral
                'rsi_4h': 50,
                'macd_1h': 0,
                'macd_4h': 0,
                'volatility': abs(pnl_pct) if pnl_pct else 1.0,
                'side_numeric': 1 if side == 'BUY' else -1,
                'regime_numeric': 0,  # Neutral
                'sentiment_score': 0,  # Neutral
                'ml_confidence': 0.5,  # Neutral
                # Agregar más features básicas según sea necesario
            }

            logger.debug(f"⚠️ Features básicas reconstruidas para trade (precisión limitada)")
            return basic_features

        except Exception as e:
            logger.error(f"Error reconstruyendo features: {e}")
            return None

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
        """Retorna estadísticas completas del sistema (v2.0: stats mínimas si no hay paper_trader)"""
        # Stats de paper trading (si está disponible)
        trading_stats = {}
        if self.paper_trader:
            trading_stats = self.paper_trader.get_statistics()
        else:
            # v2.0: Stats placeholder (TODO: obtener de Binance PositionMonitor)
            trading_stats = {
                'total_trades': 0,
                'win_rate': 0.0,
                'roi': 0.0,
                'net_pnl': 0.0,
                'current_balance': 0.0,
                'initial_balance': 0.0
            }

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

    def force_retrain(self, min_samples_override: int = None, external_paper_trader=None):
        """
        Fuerza reentrenamiento inmediato, opcionalmente con threshold reducido

        Args:
            min_samples_override: Si se proporciona, usa este threshold temporalmente
                                 Útil para entrenar con menos datos (ej: 25 en lugar de 30)
            external_paper_trader: Paper trader externo para usar sus trades
                                  (útil cuando el ML System tiene portfolio vacío pero
                                   autonomy_controller tiene portfolio restaurado)
        """
        logger.info("🔄 Forzando reentrenamiento de modelo...")

        # Guardar threshold original
        original_min_samples = self.trainer.min_samples

        try:
            # Aplicar override temporal si se proporcionó
            if min_samples_override is not None:
                logger.info(f"   Reduciendo threshold temporalmente: {original_min_samples} → {min_samples_override}")
                self.trainer.min_samples = min_samples_override

            # Ejecutar reentrenamiento
            self._retrain_model(external_paper_trader=external_paper_trader)

        finally:
            # Restaurar threshold original SIEMPRE
            self.trainer.min_samples = original_min_samples
            if min_samples_override is not None:
                logger.info(f"   Threshold restaurado a {original_min_samples}")

    def force_optimize(self):
        """Fuerza optimización inmediata"""
        logger.info("🔄 Forzando optimización de parámetros...")
        self._optimize_parameters()

    # ========================================
    # 🧠 MÉTODOS DE APRENDIZAJE CONTINUO
    # ========================================

    def add_trade_result(self, trade_result: Dict):
        """
        🧠 Agrega resultado de trade al buffer de entrenamiento

        Args:
            trade_result: Dict con datos del trade cerrado
                - symbol: Par trading
                - side: BUY/SELL
                - pnl_pct: % de ganancia/pérdida
                - entry_price, exit_price
                - duration: duración en segundos
                - features: features de mercado al momento de entrada (opcional)
        """
        try:
            # Crear entrada para buffer
            entry = {
                'timestamp': datetime.now().isoformat(),
                'symbol': trade_result.get('symbol', 'UNKNOWN'),
                'side': trade_result.get('side', 'BUY'),
                'pnl_pct': trade_result.get('pnl_pct', 0),
                'profitable': trade_result.get('pnl_pct', 0) > 0,
                'duration': trade_result.get('duration', 0),
                'leverage': trade_result.get('leverage', 1),
                'features': trade_result.get('features', {})
            }

            # Agregar al buffer
            self.training_buffer.append(entry)

            # Limitar tamaño del buffer
            if len(self.training_buffer) > 1000:
                self.training_buffer = self.training_buffer[-1000:]

            # Incrementar contador
            self.trades_since_last_training += 1

            # Guardar buffer a disco periódicamente
            if self.trades_since_last_training % 10 == 0:
                self._save_buffer()

            logger.debug(
                f"🧠 Trade añadido al buffer: {entry['symbol']} "
                f"{'+' if entry['profitable'] else ''}{entry['pnl_pct']:.2f}% "
                f"(Buffer: {len(self.training_buffer)} samples)"
            )

            # Auto-reentrenar si acumulamos suficientes trades
            if self.should_retrain_from_buffer():
                logger.info("🔄 Auto-reentrenamiento activado por buffer lleno")
                self.retrain_with_new_data()

        except Exception as e:
            logger.error(f"❌ Error añadiendo trade a buffer: {e}")

    def should_retrain_from_buffer(self) -> bool:
        """
        Decide si es momento de reentrenar basado en el buffer

        Returns:
            True si hay suficientes datos nuevos para reentrenar
        """
        # Reentrenar cada 50 trades nuevos
        if self.trades_since_last_training >= 50:
            return True

        # O si el buffer tiene muchos datos sin procesar
        if len(self.training_buffer) >= 100 and self.trades_since_last_training >= 20:
            return True

        return False

    def retrain_with_new_data(self):
        """
        🧠 Reentrena modelo con datos nuevos del buffer

        Usa los resultados de trades recientes para mejorar el modelo.
        """
        logger.info("🧠 ML System: Reentrenando con datos nuevos del buffer...")

        try:
            # Calcular métricas del buffer
            if not self.training_buffer:
                logger.warning("⚠️ Buffer vacío, no se puede reentrenar")
                return

            profitable_trades = sum(1 for t in self.training_buffer if t.get('profitable', False))
            total_trades = len(self.training_buffer)
            win_rate = (profitable_trades / total_trades) * 100 if total_trades > 0 else 0

            logger.info(
                f"   Buffer stats: {total_trades} trades, {win_rate:.1f}% win rate, "
                f"{sum(t.get('pnl_pct', 0) for t in self.training_buffer):.2f}% total P&L"
            )

            # Llamar al reentrenamiento
            self._retrain_model()

            # Resetear contador
            self.trades_since_last_training = 0

            logger.info("✅ Reentrenamiento completado exitosamente")

        except Exception as e:
            logger.error(f"❌ Error en reentrenamiento: {e}")

    def get_learning_stats(self) -> Dict:
        """Obtiene estadísticas de aprendizaje continuo"""
        return {
            'buffer_size': len(self.training_buffer),
            'trades_since_training': self.trades_since_last_training,
            'trades_since_optimization': self.trades_since_last_optimization,
            'last_trained_samples': self.last_trained_samples,
            'predictor_loaded': self.predictor.is_model_loaded() if hasattr(self.predictor, 'is_model_loaded') else False
        }

    # ========================================
    # 🔮 PREDICCIÓN PARA DECISION BRAIN
    # ========================================

    def get_prediction(self, symbol: str, market_data: Dict) -> Dict:
        """
        🔮 Obtiene predicción ML para un símbolo

        Args:
            symbol: Par de trading (ej: BTCUSDT)
            market_data: Datos de mercado con indicadores

        Returns:
            Dict con predicción:
            - direction: 'LONG', 'SHORT', 'NEUTRAL'
            - confidence: 0-100
            - features_used: lista de features
        """
        try:
            # Crear features desde market_data
            indicators = market_data.get('indicators', {})
            mtf_indicators = market_data.get('mtf_indicators', {})
            sentiment_features = market_data.get('sentiment', {})
            orderbook_features = market_data.get('orderbook', {})
            regime_features = market_data.get('regime', {})

            # Usar feature_engineer para crear features
            features = self.feature_engineer.create_features(
                indicators=indicators,
                signals={},  # Sin señal previa
                mtf_indicators=mtf_indicators,
                sentiment_features=sentiment_features,
                orderbook_features=orderbook_features,
                regime_features=regime_features
            )

            # Verificar si el predictor está disponible
            if not self.predictor or not self.enable_ml:
                return {
                    'direction': 'NEUTRAL',
                    'confidence': 50,
                    'features_used': list(features.keys()) if features else [],
                    'reason': 'ML predictor not available'
                }

            # Obtener predicción del modelo
            prediction = self.predictor.predict(features)

            if prediction:
                # Determinar dirección basada en la predicción
                prob_long = prediction.get('prob_long', 0.5)
                prob_short = prediction.get('prob_short', 0.5)

                if prob_long > 0.6:
                    direction = 'LONG'
                    confidence = int(prob_long * 100)
                elif prob_short > 0.6:
                    direction = 'SHORT'
                    confidence = int(prob_short * 100)
                else:
                    direction = 'NEUTRAL'
                    confidence = 50

                return {
                    'direction': direction,
                    'confidence': confidence,
                    'prob_long': prob_long,
                    'prob_short': prob_short,
                    'features_used': list(features.keys()) if features else [],
                    'model_version': self.predictor.get_model_info().get('version', 'unknown')
                }
            else:
                return {
                    'direction': 'NEUTRAL',
                    'confidence': 50,
                    'features_used': [],
                    'reason': 'No prediction available'
                }

        except Exception as e:
            logger.error(f"❌ Error en get_prediction: {e}")
            return {
                'direction': 'NEUTRAL',
                'confidence': 50,
                'features_used': [],
                'error': str(e)
            }
