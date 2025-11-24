"""
Backtester - Simula el bot corriendo en datos históricos
Genera señales históricas con resultado conocido (WIN/LOSS) para entrenar ML
"""
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import json

from src.advanced_technical_analysis import AdvancedTechnicalAnalyzer
from src.flash_signal_analyzer import FlashSignalAnalyzer
from src.ml.feature_engineer import FeatureEngineer
from config import config

logger = logging.getLogger(__name__)


class Backtester:
    """
    Simula trading en datos históricos para generar datos de entrenamiento ML

    Características:
    - Simula ejecución cronológica (evita look-ahead bias)
    - Usa mismos analizadores que bot en vivo
    - Calcula resultado real de cada señal (WIN/LOSS)
    - Incluye comisiones y slippage
    - Genera features ML para cada señal
    """

    def __init__(
        self,
        initial_balance: float = 50000.0,
        commission_rate: float = 0.001,  # 0.1% comisión
        slippage_rate: float = 0.0005,   # 0.05% slippage
        telegram_bot=None  # NUEVO: Para enviar notificaciones
    ):
        self.initial_balance = initial_balance
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.telegram_bot = telegram_bot  # NUEVO

        # Analizadores (mismos que en vivo)
        self.analyzer = AdvancedTechnicalAnalyzer()
        self.flash_analyzer = FlashSignalAnalyzer()
        self.feature_engineer = FeatureEngineer()

        # Resultados
        self.backtest_results = []
        self.trades_simulated = 0
        self.wins = 0
        self.losses = 0

    async def _send_telegram_notification(self, message: str):
        """Envía notificación a Telegram si telegram_bot está disponible"""
        if self.telegram_bot:
            try:
                await self.telegram_bot.send_status_message(message)
            except Exception as e:
                logger.warning(f"No se pudo enviar notificación a Telegram: {e}")

    async def run_backtest(
        self,
        historical_data: Dict[str, Dict[str, pd.DataFrame]],
        signal_type: str = 'both'  # 'conservative', 'flash', 'both'
    ) -> List[Dict]:
        """
        Corre backtest en datos históricos

        Args:
            historical_data: Dict con {pair: {timeframe: DataFrame}}
            signal_type: Tipo de señales a generar

        Returns:
            Lista de señales con features y resultado (WIN/LOSS)
        """
        logger.info("🔄 Iniciando backtest histórico...")
        logger.info(f"   Pares: {len(historical_data)}")
        logger.info(f"   Tipo de señales: {signal_type}")

        # NUEVO: Notificación inicial a Telegram
        await self._send_telegram_notification(
            f"📊 **INICIANDO BACKTEST**\n\n"
            f"Pares: {len(historical_data)}\n"
            f"Periodo: {config.HISTORICAL_START_DATE} a {config.HISTORICAL_END_DATE}\n"
            f"Tipo: {signal_type}"
        )

        self.backtest_results = []
        total_pairs = len(historical_data)

        for idx, (pair, timeframe_data) in enumerate(historical_data.items(), 1):
            logger.info(f"\n[{idx}/{total_pairs}] Backtesting {pair}...")

            # NUEVO: Notificación de progreso cada 5 pares
            if idx % 5 == 1 or idx == total_pairs:
                await self._send_telegram_notification(
                    f"⏳ Backtest progreso: {idx}/{total_pairs} pares\n"
                    f"Procesando: {pair}"
                )

            # Backtest conservativo (multi-timeframe)
            conservative_count = 0
            if signal_type in ['conservative', 'both']:
                if '1h' in timeframe_data and '4h' in timeframe_data and '1d' in timeframe_data:
                    conservative_signals = self._backtest_conservative(pair, timeframe_data)
                    self.backtest_results.extend(conservative_signals)
                    conservative_count = len(conservative_signals)
                    logger.info(f"   Conservative: {conservative_count} señales")

            # Backtest flash (15m)
            flash_count = 0
            if signal_type in ['flash', 'both']:
                if '15m' in timeframe_data or '1h' in timeframe_data:
                    flash_timeframe = '15m' if '15m' in timeframe_data else '1h'
                    flash_signals = self._backtest_flash(pair, timeframe_data[flash_timeframe])
                    self.backtest_results.extend(flash_signals)
                    flash_count = len(flash_signals)
                    logger.info(f"   Flash: {flash_count} señales")

        # Calcular estadísticas
        self.wins = sum(1 for s in self.backtest_results if s['result'] == 'WIN')
        self.losses = sum(1 for s in self.backtest_results if s['result'] == 'LOSS')
        self.trades_simulated = len(self.backtest_results)
        win_rate = self.wins/max(self.trades_simulated,1)*100

        logger.info(f"\n✅ Backtest completado!")
        logger.info(f"   Total señales: {self.trades_simulated}")
        logger.info(f"   WIN: {self.wins} ({win_rate:.1f}%)")
        logger.info(f"   LOSS: {self.losses} ({self.losses/max(self.trades_simulated,1)*100:.1f}%)")

        # NUEVO: Notificación final a Telegram con resultados
        await self._send_telegram_notification(
            f"✅ **BACKTEST COMPLETADO**\n\n"
            f"📊 Total señales: {self.trades_simulated}\n"
            f"✅ Ganadas: {self.wins} ({win_rate:.1f}%)\n"
            f"❌ Perdidas: {self.losses}\n"
            f"📅 Periodo: {config.HISTORICAL_START_DATE} - {config.HISTORICAL_END_DATE}"
        )

        # NUEVO: Guardar resultados para reusar
        self._save_backtest_results()

        return self.backtest_results

    def _save_backtest_results(self):
        """Guarda resultados del backtest para reusar en futuros deploys"""
        try:
            # Crear directorio si no existe
            data_dir = Path('data/ml')
            data_dir.mkdir(parents=True, exist_ok=True)

            # Guardar resultados
            results_file = data_dir / 'backtest_results.json'
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'start_date': config.HISTORICAL_START_DATE,
                'end_date': config.HISTORICAL_END_DATE,
                'total_signals': self.trades_simulated,
                'wins': self.wins,
                'losses': self.losses,
                'win_rate': self.wins/max(self.trades_simulated,1)*100,
                'results_count': len(self.backtest_results)
            }

            with open(results_file, 'w') as f:
                json.dump({
                    'metadata': metadata,
                    'signals': self.backtest_results
                }, f)

            logger.info(f"💾 Resultados guardados en: {results_file}")

        except Exception as e:
            logger.warning(f"No se pudieron guardar resultados del backtest: {e}")

    @classmethod
    def load_backtest_results(cls, expected_date_range: Tuple[str, str] = None) -> Optional[List[Dict]]:
        """
        Carga resultados previos del backtest si existen

        Args:
            expected_date_range: Tuple (start_date, end_date) esperado

        Returns:
            Lista de señales o None si no hay resultados válidos
        """
        try:
            results_file = Path('data/ml/backtest_results.json')

            if not results_file.exists():
                return None

            with open(results_file, 'r') as f:
                data = json.load(f)

            metadata = data['metadata']

            # Verificar si el rango de fechas coincide
            if expected_date_range:
                start_expected, end_expected = expected_date_range
                if (metadata['start_date'] != start_expected or
                    metadata['end_date'] != end_expected):
                    logger.info(f"📅 Backtest guardado es de diferente periodo, se descartará")
                    return None

            logger.info(f"✅ Backtest previo encontrado:")
            logger.info(f"   Fecha: {metadata['timestamp']}")
            logger.info(f"   Señales: {metadata['total_signals']}")
            logger.info(f"   Win Rate: {metadata['win_rate']:.1f}%")

            return data['signals']

        except Exception as e:
            logger.warning(f"No se pudo cargar backtest previo: {e}")
            return None

    def _backtest_conservative(
        self,
        pair: str,
        timeframe_data: Dict[str, pd.DataFrame]
    ) -> List[Dict]:
        """
        Backtest de señales conservadoras (multi-timeframe) - OPTIMIZADO

        Args:
            pair: Par de trading
            timeframe_data: Dict con DataFrames por timeframe

        Returns:
            Lista de señales con resultado
        """
        signals = []

        # Usar timeframe 1h como base
        df_1h = timeframe_data['1h']

        # OPTIMIZACIÓN: Analizar cada 10 velas en lugar de cada vela (10x más rápido)
        step = 10

        # Iterar por cada N velas (en orden cronológico)
        for i in range(50, len(df_1h), step):  # Necesitamos al menos 50 velas para indicadores
            try:
                # Obtener datos hasta este punto (evitar look-ahead bias)
                dfs = {
                    '1h': df_1h.iloc[:i+1],
                    '4h': timeframe_data['4h'][timeframe_data['4h'].index <= df_1h.index[i]],
                    '1d': timeframe_data['1d'][timeframe_data['1d'].index <= df_1h.index[i]]
                }

                # Verificar datos suficientes
                if len(dfs['4h']) < 50 or len(dfs['1d']) < 50:
                    continue

                # Analizar
                analysis = self.analyzer.analyze_multi_timeframe(dfs)

                if not analysis:
                    continue

                signal = analysis['signals']
                indicators = analysis['indicators']

                # Solo señales fuertes (BUY o SELL)
                if signal['action'] == 'HOLD':
                    continue

                # Verificar threshold
                if signal.get('score', 0) < config.CONSERVATIVE_THRESHOLD:
                    continue

                current_price = indicators['current_price']

                # Simular trade
                trade_result = self._simulate_trade(
                    pair=pair,
                    signal=signal,
                    entry_price=current_price,
                    future_data=df_1h.iloc[i+1:],  # Datos futuros para verificar SL/TP
                    signal_type='conservative'
                )

                if trade_result:
                    # Crear features
                    features = self.feature_engineer.create_features(
                        indicators=indicators,
                        signals=signal,
                        mtf_indicators=analysis.get('mtf_indicators')
                    )

                    # Agregar a resultados
                    signal_data = {
                        'pair': pair,
                        'timestamp': df_1h.index[i],
                        'signal_type': 'conservative',
                        'action': signal['action'],
                        'entry_price': current_price,
                        'score': signal.get('score', 0),
                        'confidence': signal.get('confidence', 0),
                        'features': features,
                        'result': trade_result['result'],  # WIN o LOSS
                        'pnl_pct': trade_result['pnl_pct'],
                        'exit_price': trade_result['exit_price'],
                        'exit_reason': trade_result['exit_reason'],
                        'bars_held': trade_result['bars_held']
                    }

                    signals.append(signal_data)

            except Exception as e:
                logger.debug(f"Error en vela {i} de {pair}: {e}")
                continue

        return signals

    def _backtest_flash(
        self,
        pair: str,
        df: pd.DataFrame
    ) -> List[Dict]:
        """
        Backtest de señales flash (timeframe corto) - OPTIMIZADO

        Args:
            pair: Par de trading
            df: DataFrame con datos históricos

        Returns:
            Lista de señales con resultado
        """
        signals = []

        # OPTIMIZACIÓN: Analizar cada 5 velas en lugar de cada vela (5x más rápido)
        step = 5

        # Iterar por cada N velas
        for i in range(30, len(df), step):
            try:
                # Datos hasta este punto
                df_until_now = df.iloc[:i+1]

                if len(df_until_now) < 30:
                    continue

                # Analizar
                analysis = self.flash_analyzer.analyze_flash(df_until_now)

                if not analysis:
                    continue

                signal = analysis['signals']
                indicators = analysis['indicators']

                # Solo señales fuertes
                if signal['action'] == 'HOLD':
                    continue

                # Verificar threshold y confidence
                if signal.get('score', 0) < config.FLASH_THRESHOLD:
                    continue
                if signal.get('confidence', 0) < config.FLASH_MIN_CONFIDENCE:
                    continue

                current_price = indicators['current_price']

                # Simular trade
                trade_result = self._simulate_trade(
                    pair=pair,
                    signal=signal,
                    entry_price=current_price,
                    future_data=df.iloc[i+1:],
                    signal_type='flash'
                )

                if trade_result:
                    # Crear features
                    features = self.feature_engineer.create_features(
                        indicators=indicators,
                        signals=signal,
                        mtf_indicators=None
                    )

                    # Agregar a resultados
                    signal_data = {
                        'pair': pair,
                        'timestamp': df.index[i],
                        'signal_type': 'flash',
                        'action': signal['action'],
                        'entry_price': current_price,
                        'score': signal.get('score', 0),
                        'confidence': signal.get('confidence', 0),
                        'features': features,
                        'result': trade_result['result'],
                        'pnl_pct': trade_result['pnl_pct'],
                        'exit_price': trade_result['exit_price'],
                        'exit_reason': trade_result['exit_reason'],
                        'bars_held': trade_result['bars_held']
                    }

                    signals.append(signal_data)

            except Exception as e:
                logger.debug(f"Error en vela {i} de {pair}: {e}")
                continue

        return signals

    def _simulate_trade(
        self,
        pair: str,
        signal: Dict,
        entry_price: float,
        future_data: pd.DataFrame,
        signal_type: str
    ) -> Optional[Dict]:
        """
        Simula un trade y determina si fue WIN o LOSS

        Args:
            pair: Par
            signal: Señal generada
            entry_price: Precio de entrada
            future_data: Datos futuros para verificar SL/TP
            signal_type: Tipo de señal

        Returns:
            Dict con resultado o None si no se puede simular
        """
        if len(future_data) < 2:
            return None

        # Aplicar slippage y comisión a entrada
        if signal['action'] == 'BUY':
            actual_entry = entry_price * (1 + self.slippage_rate)
        else:  # SELL
            actual_entry = entry_price * (1 - self.slippage_rate)

        # Comisión de entrada
        entry_commission = actual_entry * self.commission_rate

        # Stop Loss y Take Profit desde señal
        stop_loss = signal.get('stop_loss')
        take_profit = signal.get('take_profit', {})

        if not stop_loss or not take_profit:
            return None

        # Tomar TP2 como target (balance entre TP1 y TP3)
        tp_target = take_profit.get('tp2', take_profit.get('tp1'))

        if not tp_target:
            return None

        # Simular evolución del precio
        for idx, (timestamp, row) in enumerate(future_data.iterrows(), 1):
            high = row['high']
            low = row['low']
            close = row['close']

            if signal['action'] == 'BUY':
                # Verificar Stop Loss
                if low <= stop_loss:
                    exit_price = stop_loss * (1 - self.slippage_rate)
                    exit_commission = exit_price * self.commission_rate
                    pnl_pct = ((exit_price - actual_entry) / actual_entry) * 100
                    pnl_pct -= (entry_commission + exit_commission) / actual_entry * 100

                    return {
                        'result': 'LOSS',
                        'pnl_pct': pnl_pct,
                        'exit_price': exit_price,
                        'exit_reason': 'STOP_LOSS',
                        'bars_held': idx
                    }

                # Verificar Take Profit
                if high >= tp_target:
                    exit_price = tp_target * (1 - self.slippage_rate)
                    exit_commission = exit_price * self.commission_rate
                    pnl_pct = ((exit_price - actual_entry) / actual_entry) * 100
                    pnl_pct -= (entry_commission + exit_commission) / actual_entry * 100

                    return {
                        'result': 'WIN',
                        'pnl_pct': pnl_pct,
                        'exit_price': exit_price,
                        'exit_reason': 'TAKE_PROFIT',
                        'bars_held': idx
                    }

            else:  # SELL
                # Verificar Stop Loss (inverso)
                if high >= stop_loss:
                    exit_price = stop_loss * (1 + self.slippage_rate)
                    exit_commission = exit_price * self.commission_rate
                    pnl_pct = ((actual_entry - exit_price) / actual_entry) * 100
                    pnl_pct -= (entry_commission + exit_commission) / actual_entry * 100

                    return {
                        'result': 'LOSS',
                        'pnl_pct': pnl_pct,
                        'exit_price': exit_price,
                        'exit_reason': 'STOP_LOSS',
                        'bars_held': idx
                    }

                # Verificar Take Profit (inverso)
                if low <= tp_target:
                    exit_price = tp_target * (1 + self.slippage_rate)
                    exit_commission = exit_price * self.commission_rate
                    pnl_pct = ((actual_entry - exit_price) / actual_entry) * 100
                    pnl_pct -= (entry_commission + exit_commission) / actual_entry * 100

                    return {
                        'result': 'WIN',
                        'pnl_pct': pnl_pct,
                        'exit_price': exit_price,
                        'exit_reason': 'TAKE_PROFIT',
                        'bars_held': idx
                    }

            # Límite de 100 velas (evitar trades eternos)
            if idx >= 100:
                # Cerrar en precio actual
                exit_price = close
                exit_commission = exit_price * self.commission_rate

                if signal['action'] == 'BUY':
                    pnl_pct = ((exit_price - actual_entry) / actual_entry) * 100
                else:
                    pnl_pct = ((actual_entry - exit_price) / actual_entry) * 100

                pnl_pct -= (entry_commission + exit_commission) / actual_entry * 100

                return {
                    'result': 'WIN' if pnl_pct > 0 else 'LOSS',
                    'pnl_pct': pnl_pct,
                    'exit_price': exit_price,
                    'exit_reason': 'TIMEOUT',
                    'bars_held': idx
                }

        return None

    def save_results(self, filename: str = 'backtest_results.json'):
        """Guarda resultados de backtest a disco"""
        output_dir = Path('data/training')
        output_dir.mkdir(parents=True, exist_ok=True)

        filepath = output_dir / filename

        # Convertir timestamps a string para JSON
        results_serializable = []
        for result in self.backtest_results:
            result_copy = result.copy()
            result_copy['timestamp'] = result['timestamp'].isoformat()
            results_serializable.append(result_copy)

        with open(filepath, 'w') as f:
            json.dump(results_serializable, f, indent=2)

        logger.info(f"💾 Resultados guardados: {filepath}")

    def load_results(self, filename: str = 'backtest_results.json') -> List[Dict]:
        """Carga resultados de backtest desde disco"""
        filepath = Path('data/training') / filename

        if not filepath.exists():
            logger.warning(f"Archivo no encontrado: {filepath}")
            return []

        with open(filepath, 'r') as f:
            results = json.load(f)

        # Convertir timestamps de vuelta a datetime
        for result in results:
            result['timestamp'] = pd.to_datetime(result['timestamp'])

        self.backtest_results = results
        self.trades_simulated = len(results)
        self.wins = sum(1 for r in results if r['result'] == 'WIN')
        self.losses = sum(1 for r in results if r['result'] == 'LOSS')

        logger.info(f"📂 Resultados cargados: {len(results)} señales")

        return results

    def get_statistics(self) -> Dict:
        """Retorna estadísticas del backtest"""
        if not self.backtest_results:
            return {}

        win_rate = (self.wins / max(self.trades_simulated, 1)) * 100

        winning_trades = [r for r in self.backtest_results if r['result'] == 'WIN']
        losing_trades = [r for r in self.backtest_results if r['result'] == 'LOSS']

        avg_win = sum(r['pnl_pct'] for r in winning_trades) / max(len(winning_trades), 1)
        avg_loss = sum(r['pnl_pct'] for r in losing_trades) / max(len(losing_trades), 1)

        total_pnl = sum(r['pnl_pct'] for r in self.backtest_results)

        return {
            'total_signals': self.trades_simulated,
            'wins': self.wins,
            'losses': self.losses,
            'win_rate': round(win_rate, 2),
            'avg_win_pct': round(avg_win, 2),
            'avg_loss_pct': round(avg_loss, 2),
            'total_pnl_pct': round(total_pnl, 2),
            'avg_trade_pnl_pct': round(total_pnl / max(self.trades_simulated, 1), 2)
        }
