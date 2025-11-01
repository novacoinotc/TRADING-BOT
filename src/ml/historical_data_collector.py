"""
Historical Data Collector
Descarga datos históricos de Binance para pre-entrenamiento del modelo ML
"""
import ccxt
import pandas as pd
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import asyncio
import time
from config import config

logger = logging.getLogger(__name__)


class HistoricalDataCollector:
    """
    Descarga y almacena datos históricos de exchange para entrenamiento ML

    Características:
    - Descarga múltiples pares en paralelo
    - Maneja rate limits de exchange
    - Guarda datos en CSV para reutilización
    - Soporta múltiples timeframes
    """

    def __init__(self, exchange_name: str = 'binance'):
        self.exchange_name = exchange_name
        self.exchange = self._initialize_exchange()
        self.data_dir = Path('data/historical')
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _initialize_exchange(self) -> ccxt.Exchange:
        """Inicializa conexión con exchange"""
        try:
            exchange_class = getattr(ccxt, self.exchange_name)

            exchange_config = {
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            }

            # Agregar proxy si está configurado
            if config.USE_PROXY and config.PROXY_HOST and config.PROXY_PORT:
                if config.PROXY_USERNAME and config.PROXY_PASSWORD:
                    proxy_url = f"http://{config.PROXY_USERNAME}:{config.PROXY_PASSWORD}@{config.PROXY_HOST}:{config.PROXY_PORT}"
                else:
                    proxy_url = f"http://{config.PROXY_HOST}:{config.PROXY_PORT}"

                exchange_config['proxies'] = {
                    'http': proxy_url,
                    'https': proxy_url,
                }

            exchange = exchange_class(exchange_config)
            return exchange

        except Exception as e:
            logger.error(f"Error inicializando exchange: {e}")
            raise

    def download_pair_data(
        self,
        pair: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        force_download: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Descarga datos históricos para un par

        Args:
            pair: Par de trading (ej: 'BTC/USDT')
            timeframe: Timeframe (ej: '1h', '4h', '1d')
            start_date: Fecha inicial
            end_date: Fecha final
            force_download: Forzar descarga aunque exista cache

        Returns:
            DataFrame con datos OHLCV o None si error
        """
        # Verificar si ya existe en cache
        cache_file = self._get_cache_filename(pair, timeframe, start_date, end_date)

        if cache_file.exists() and not force_download:
            logger.info(f"✅ Usando cache para {pair} {timeframe}")
            return pd.read_csv(cache_file, index_col='timestamp', parse_dates=True)

        try:
            logger.info(f"📥 Descargando {pair} {timeframe} desde {start_date.date()} hasta {end_date.date()}...")

            # Convertir fechas a timestamps
            since = int(start_date.timestamp() * 1000)
            end_ts = int(end_date.timestamp() * 1000)

            all_ohlcv = []
            current_since = since

            # Descargar en chunks (máximo 1000 velas por request)
            while current_since < end_ts:
                try:
                    ohlcv = self.exchange.fetch_ohlcv(
                        pair,
                        timeframe=timeframe,
                        since=current_since,
                        limit=1000
                    )

                    if not ohlcv:
                        break

                    all_ohlcv.extend(ohlcv)

                    # Actualizar timestamp para siguiente chunk
                    current_since = ohlcv[-1][0] + 1

                    # Evitar rate limits
                    time.sleep(self.exchange.rateLimit / 1000)

                    # Log progreso
                    current_date = datetime.fromtimestamp(current_since / 1000)
                    if len(all_ohlcv) % 5000 == 0:
                        logger.debug(f"   Descargadas {len(all_ohlcv)} velas hasta {current_date.date()}")

                except ccxt.BadSymbol:
                    logger.warning(f"Par {pair} no disponible en {self.exchange_name}")
                    return None
                except Exception as e:
                    logger.error(f"Error descargando chunk de {pair}: {e}")
                    break

            if not all_ohlcv:
                logger.warning(f"No se descargaron datos para {pair}")
                return None

            # Convertir a DataFrame
            df = pd.DataFrame(
                all_ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')

            # Filtrar por rango de fechas
            df = df[(df.index >= start_date) & (df.index <= end_date)]

            # Eliminar duplicados
            df = df[~df.index.duplicated(keep='first')]

            # Guardar en cache
            df.to_csv(cache_file)

            logger.info(f"✅ {pair} {timeframe}: {len(df)} velas descargadas")

            return df

        except Exception as e:
            logger.error(f"Error descargando {pair} {timeframe}: {e}")
            return None

    def download_all_pairs(
        self,
        pairs: List[str],
        timeframes: List[str],
        start_date: datetime,
        end_date: datetime,
        force_download: bool = False
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Descarga datos para múltiples pares y timeframes

        Args:
            pairs: Lista de pares
            timeframes: Lista de timeframes
            start_date: Fecha inicial
            end_date: Fecha final
            force_download: Forzar descarga

        Returns:
            Dict con estructura {pair: {timeframe: DataFrame}}
        """
        logger.info(f"📊 Descargando datos históricos para {len(pairs)} pares y {len(timeframes)} timeframes")
        logger.info(f"   Periodo: {start_date.date()} hasta {end_date.date()}")

        all_data = {}
        total_pairs = len(pairs)

        for idx, pair in enumerate(pairs, 1):
            logger.info(f"\n[{idx}/{total_pairs}] Procesando {pair}...")

            pair_data = {}

            for timeframe in timeframes:
                df = self.download_pair_data(
                    pair=pair,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    force_download=force_download
                )

                if df is not None and len(df) >= 50:
                    pair_data[timeframe] = df
                else:
                    logger.warning(f"⚠️ Datos insuficientes para {pair} {timeframe}")

            if pair_data:
                all_data[pair] = pair_data

            # Pausa entre pares para evitar rate limits
            if idx < total_pairs:
                time.sleep(1)

        logger.info(f"\n✅ Descarga completa: {len(all_data)} pares con datos válidos")

        return all_data

    def _get_cache_filename(
        self,
        pair: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> Path:
        """Genera nombre de archivo cache"""
        pair_clean = pair.replace('/', '_')
        start_str = start_date.strftime('%Y%m%d')
        end_str = end_date.strftime('%Y%m%d')

        filename = f"{pair_clean}_{timeframe}_{start_str}_{end_str}.csv"
        return self.data_dir / filename

    def get_cache_info(self) -> Dict:
        """Retorna información sobre datos en cache"""
        cache_files = list(self.data_dir.glob('*.csv'))

        total_size = sum(f.stat().st_size for f in cache_files)
        total_size_mb = total_size / (1024 * 1024)

        return {
            'total_files': len(cache_files),
            'total_size_mb': round(total_size_mb, 2),
            'data_dir': str(self.data_dir)
        }

    def clear_cache(self):
        """Elimina todos los datos en cache"""
        cache_files = list(self.data_dir.glob('*.csv'))

        for file in cache_files:
            file.unlink()

        logger.info(f"🗑️ Cache limpiado: {len(cache_files)} archivos eliminados")


def download_historical_data_from_config() -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Función helper para descargar datos usando configuración del config.py

    Returns:
        Dict con datos históricos
    """
    collector = HistoricalDataCollector(exchange_name=config.EXCHANGE_NAME)

    # Fechas desde config
    start_date = datetime.strptime(config.HISTORICAL_START_DATE, '%Y-%m-%d')
    end_date = datetime.strptime(config.HISTORICAL_END_DATE, '%Y-%m-%d')

    # Timeframes desde config
    timeframes = config.HISTORICAL_TIMEFRAMES

    # Pares desde config
    pairs = config.TRADING_PAIRS

    # Descargar
    all_data = collector.download_all_pairs(
        pairs=pairs,
        timeframes=timeframes,
        start_date=start_date,
        end_date=end_date,
        force_download=config.FORCE_HISTORICAL_DOWNLOAD
    )

    return all_data
