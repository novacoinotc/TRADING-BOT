"""
Binance Futures REST API Client
Cliente completo para interactuar con Binance USD-M Futures API
"""

import logging
import time
import requests
from typing import Dict, List, Optional, Union
from decimal import Decimal

from .utils import (
    get_timestamp,
    build_signed_request,
    validate_price,
    validate_quantity,
    parse_binance_error,
    get_error_description
)

logger = logging.getLogger(__name__)


class BinanceClientError(Exception):
    """Error del cliente de Binance"""
    pass


class BinanceAPIError(Exception):
    """Error de la API de Binance"""
    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message
        super().__init__(f"[{code}] {message}")


class BinanceClient:
    """
    Cliente para Binance USD-M Futures API

    Maneja:
    - Autenticación con API key/secret
    - Firma HMAC SHA256 de requests
    - Rate limiting
    - Manejo de errores
    - Endpoints públicos y privados
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str = "https://fapi.binance.com",
        timeout: int = 30,  # Aumentado de 10s a 30s para proxy lento de Railway
        proxies: Optional[Dict] = None
    ):
        """
        Args:
            api_key: Binance API Key
            api_secret: Binance API Secret
            base_url: Base URL de la API (producción por defecto, testnet opcional)
            timeout: Timeout de requests en segundos
            proxies: Proxies opcionales para requests
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.proxies = proxies

        # Cache para exchange info
        self._exchange_info_cache = None
        self._exchange_info_timestamp = 0
        self._exchange_info_ttl = 3600  # 1 hora

        # Server time offset y sincronización automática
        self._time_offset = 0
        self._last_sync = 0  # Timestamp de última sincronización
        self._sync_interval = 1800  # Re-sync cada 30 minutos
        self._sync_time()

        logger.info(f"✅ Binance Client inicializado: {base_url} (timeout={timeout}s para Railway proxy)")

    def _sync_time(self):
        """
        Sincroniza tiempo con servidor de Binance.
        Previene error -1021 (timestamp outside recvWindow).
        """
        try:
            server_time = self.get_server_time()
            local_time = int(time.time() * 1000)
            self._time_offset = server_time - local_time
            self._last_sync = time.time()

            logger.info(f"⏰ Tiempo sincronizado con Binance: offset={self._time_offset}ms")

            if abs(self._time_offset) > 1000:
                logger.warning(f"⚠️ Offset de tiempo alto: {self._time_offset}ms (>1s) - Railway proxy puede causar latencia")

        except Exception as e:
            logger.error(f"❌ Error sincronizando tiempo con Binance: {e}")
            self._time_offset = 0
            self._last_sync = time.time()  # Evitar retry loop

    def _get_adjusted_timestamp(self) -> int:
        """
        Obtiene timestamp corregido con offset de Binance.
        Re-sincroniza automáticamente cada 30 minutos.
        """
        # Re-sincronizar si ha pasado el intervalo
        current_time = time.time()
        if current_time - self._last_sync > self._sync_interval:
            logger.info(f"⏰ Re-sincronizando tiempo (han pasado {int(current_time - self._last_sync)}s)")
            self._sync_time()

        return get_timestamp() + self._time_offset

    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        signed: bool = False
    ) -> Dict:
        """
        Hace request a la API de Binance con retry logic y exponential backoff

        Args:
            method: HTTP method (GET, POST, DELETE, PUT)
            endpoint: Endpoint de la API (ej: /fapi/v1/ticker/price)
            params: Parámetros de la request
            signed: Si requiere firma HMAC

        Returns:
            Dict: Respuesta de la API

        Raises:
            BinanceAPIError: Si la API retorna error
            BinanceClientError: Si hay error de conexión
        """
        max_retries = 5  # Aumentado de 3 a 5 para Railway proxy lento
        base_delay = 1  # segundos (exponential: 1s, 2s, 4s, 8s, 16s)

        for attempt in range(max_retries):
            try:
                url = f"{self.base_url}{endpoint}"
                headers = {'X-MBX-APIKEY': self.api_key} if signed or self.api_key else {}

                params = params or {}

                if signed:
                    # Construir query string firmado
                    query_string = build_signed_request(
                        params,
                        self.api_secret,
                        timestamp=self._get_adjusted_timestamp()
                    )
                    url = f"{url}?{query_string}"
                    response = requests.request(
                        method,
                        url,
                        headers=headers,
                        timeout=self.timeout,
                        proxies=self.proxies
                    )
                else:
                    # Request pública
                    response = requests.request(
                        method,
                        url,
                        params=params,
                        headers=headers,
                        timeout=self.timeout,
                        proxies=self.proxies
                    )

                # Parsear respuesta
                data = response.json()

                # Verificar errores
                if response.status_code != 200:
                    error_code = data.get('code', response.status_code)
                    error_msg = data.get('msg', 'Unknown error')
                    error_desc = get_error_description(error_code)

                    # Error -4046 es informativo (margin type ya configurado), no es un error real
                    if error_code == -4046:
                        logger.info(
                            f"ℹ️ Binance Info (Code {error_code}): {error_msg}"
                        )
                    else:
                        # Otros errores sí son verdaderos errores
                        logger.error(
                            f"❌ Binance API Error:\n"
                            f"   Code: {error_code}\n"
                            f"   Message: {error_msg}\n"
                            f"   Description: {error_desc}\n"
                            f"   Endpoint: {endpoint}"
                        )

                    # Manejar errores específicos
                    if error_code == -1021:  # Timestamp out of sync
                        logger.error(f"🕐 ERROR TIMESTAMP (-1021): Request expiró")
                        logger.info(f"   └─ Forzando re-sincronización de tiempo...")
                        self._sync_time()
                        # Retry para timestamp errors
                        if attempt < max_retries - 1:
                            logger.warning(f"   └─ ⏳ Retry inmediato con timestamp corregido (attempt {attempt+1}/{max_retries})")
                            time.sleep(1)  # Solo 1s de espera para -1021
                            continue
                        else:
                            logger.error(f"❌ Error -1021 persistente después de {max_retries} intentos")

                    raise BinanceAPIError(error_code, error_msg)

                return data

            except requests.exceptions.ProxyError as pe:
                # Error específico de proxy (Railway)
                delay = min(base_delay * (2 ** attempt), 30)  # Cap en 30s
                logger.error(f"🌐 PROXY ERROR detectado:")
                logger.error(f"   └─ Proxy: Railway (posible timeout de conexión)")
                logger.error(f"   └─ Endpoint: {endpoint}")
                logger.error(f"   └─ Intento: {attempt + 1}/{max_retries}")

                if attempt < max_retries - 1:
                    logger.warning(f"   └─ ⏳ Retry en {delay}s...")
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"❌ PROXY: Todos los retries fallaron después de {max_retries} intentos")
                    raise BinanceClientError(f"Proxy error después de {max_retries} intentos: {pe}")

            except requests.exceptions.Timeout as te:
                # Timeout específico
                delay = min(base_delay * (2 ** attempt), 30)
                logger.error(f"⏱️ TIMEOUT después de esperar {self.timeout}s:")
                logger.error(f"   └─ Endpoint: {endpoint}")
                logger.error(f"   └─ Intento: {attempt + 1}/{max_retries}")

                if attempt < max_retries - 1:
                    logger.warning(f"   └─ ⏳ Retry en {delay}s...")
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"❌ TIMEOUT: Max retries alcanzado")
                    raise BinanceClientError(f"Timeout después de {max_retries} intentos: {te}")

            except (requests.exceptions.ConnectTimeout,
                    requests.exceptions.ConnectionError) as e:
                # Otros errores de conexión: aplicar exponential backoff
                delay = min(base_delay * (2 ** attempt), 30)  # Cap en 30s
                if attempt < max_retries - 1:
                    logger.warning(
                        f"⚠️ Connection error on {endpoint}:\n"
                        f"   └─ Error: {type(e).__name__}\n"
                        f"   └─ Intento: {attempt+1}/{max_retries}\n"
                        f"   └─ ⏳ Retry en {delay}s..."
                    )
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"❌ Max retries exceeded for {endpoint}: {e}")
                    raise BinanceClientError(f"Max retries exceeded: {e}")

            except requests.exceptions.RequestException as e:
                # Otros errores de requests: fallar inmediatamente
                logger.error(f"❌ Request error on {endpoint}: {e}")
                raise BinanceClientError(f"Request error: {e}")

    # ========== ENDPOINTS PÚBLICOS ==========

    def get_server_time(self) -> int:
        """
        Obtiene timestamp del servidor

        Returns:
            int: Server timestamp en milisegundos
        """
        response = self._make_request('GET', '/fapi/v1/time')
        return response['serverTime']

    def get_exchange_info(self, symbol: Optional[str] = None) -> Dict:
        """
        Obtiene información de exchange (límites, precision, etc.)

        Args:
            symbol: Símbolo opcional para filtrar

        Returns:
            Dict: Exchange info completo o del símbolo
        """
        # Usar cache si es válido
        current_time = time.time()
        if self._exchange_info_cache and (current_time - self._exchange_info_timestamp) < self._exchange_info_ttl:
            data = self._exchange_info_cache
        else:
            params = {'symbol': symbol} if symbol else {}
            data = self._make_request('GET', '/fapi/v1/exchangeInfo', params=params)
            self._exchange_info_cache = data
            self._exchange_info_timestamp = current_time

        return data

    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """
        Obtiene información de un símbolo específico

        Args:
            symbol: Símbolo (ej: BTCUSDT)

        Returns:
            Dict: Info del símbolo o None si no existe
        """
        exchange_info = self.get_exchange_info()
        for s in exchange_info['symbols']:
            if s['symbol'] == symbol:
                return s
        return None

    def get_price(self, symbol: str) -> float:
        """
        Obtiene precio actual de un símbolo

        Args:
            symbol: Símbolo (ej: BTCUSDT)

        Returns:
            float: Precio actual
        """
        response = self._make_request('GET', '/fapi/v1/ticker/price', params={'symbol': symbol})
        return float(response['price'])

    def get_orderbook(self, symbol: str, limit: int = 20) -> Dict:
        """
        Obtiene order book

        Args:
            symbol: Símbolo
            limit: Número de niveles (5, 10, 20, 50, 100, 500, 1000)

        Returns:
            Dict: Order book con bids y asks
        """
        params = {'symbol': symbol, 'limit': limit}
        return self._make_request('GET', '/fapi/v1/depth', params=params)

    def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[List]:
        """
        Obtiene velas/klines

        Args:
            symbol: Símbolo
            interval: Intervalo (1m, 5m, 15m, 1h, 4h, 1d, etc.)
            limit: Número de velas (max 1500)
            start_time: Timestamp de inicio (opcional)
            end_time: Timestamp de fin (opcional)

        Returns:
            List: Lista de velas
        """
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': min(limit, 1500)
        }
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time

        return self._make_request('GET', '/fapi/v1/klines', params=params)

    def get_mark_price(self, symbol: str) -> Dict:
        """
        Obtiene mark price y funding rate

        Args:
            symbol: Símbolo

        Returns:
            Dict: Mark price, funding rate, next funding time
        """
        params = {'symbol': symbol}
        return self._make_request('GET', '/fapi/v1/premiumIndex', params=params)

    def get_open_interest(self, symbol: str) -> Dict:
        """
        Obtiene interés abierto

        Args:
            symbol: Símbolo

        Returns:
            Dict: Open interest
        """
        params = {'symbol': symbol}
        return self._make_request('GET', '/fapi/v1/openInterest', params=params)

    # ========== ENDPOINTS PRIVADOS (Trading) ==========

    def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        position_side: str = 'BOTH',
        time_in_force: str = 'GTC',
        reduce_only: bool = False
    ) -> Dict:
        """
        Crea nueva orden

        Args:
            symbol: Símbolo (ej: BTCUSDT)
            side: BUY o SELL
            order_type: MARKET, LIMIT, STOP_MARKET, TAKE_PROFIT_MARKET
            quantity: Cantidad
            price: Precio (solo para LIMIT)
            stop_price: Stop price (para STOP_MARKET y TAKE_PROFIT_MARKET)
            position_side: LONG, SHORT, BOTH (default BOTH para one-way mode)
            time_in_force: GTC, IOC, FOK (solo para LIMIT)
            reduce_only: Si solo reduce posición

        Returns:
            Dict: Información de la orden creada
        """
        # Validaciones
        if not validate_quantity(quantity, symbol):
            raise ValueError(f"Invalid quantity: {quantity}")

        if price and not validate_price(price, symbol):
            raise ValueError(f"Invalid price: {price}")

        if stop_price and not validate_price(stop_price, symbol):
            raise ValueError(f"Invalid stop price: {stop_price}")

        # Construir parámetros
        params = {
            'symbol': symbol,
            'side': side,
            'type': order_type,
            'quantity': quantity,
            'positionSide': position_side,
        }

        if order_type == 'LIMIT':
            if not price:
                raise ValueError("Price required for LIMIT order")
            params['price'] = price
            params['timeInForce'] = time_in_force

        if order_type in ['STOP_MARKET', 'TAKE_PROFIT_MARKET']:
            if not stop_price:
                raise ValueError(f"Stop price required for {order_type}")
            params['stopPrice'] = stop_price

        if reduce_only:
            params['reduceOnly'] = 'true'

        logger.info(
            f"📤 Creating {order_type} order: {symbol} {side} "
            f"{quantity} @ {price or stop_price or 'MARKET'}"
        )

        return self._make_request('POST', '/fapi/v1/order', params=params, signed=True)

    def cancel_order(self, symbol: str, order_id: int) -> Dict:
        """
        Cancela orden

        Args:
            symbol: Símbolo
            order_id: ID de la orden

        Returns:
            Dict: Info de orden cancelada
        """
        # 🔍 LOGGING CRÍTICO: Rastrear cada cancelación de orden
        import traceback
        logger.warning(f"⚠️ CANCEL_ORDER LLAMADO: symbol={symbol}, order_id={order_id}")
        logger.warning(f"⚠️ TRACEBACK: {''.join(traceback.format_stack()[-6:])}")

        params = {'symbol': symbol, 'orderId': order_id}
        logger.info(f"❌ Canceling order {order_id} for {symbol}")
        return self._make_request('DELETE', '/fapi/v1/order', params=params, signed=True)

    def cancel_all_orders(self, symbol: str) -> Dict:
        """
        Cancela todas las órdenes de un símbolo

        Args:
            symbol: Símbolo

        Returns:
            Dict: Confirmación
        """
        # 🔍 LOGGING CRÍTICO: Rastrear cada cancelación masiva de órdenes
        import traceback
        logger.warning(f"⚠️ CANCEL_ALL_ORDERS LLAMADO: symbol={symbol}")
        logger.warning(f"⚠️ TRACEBACK: {''.join(traceback.format_stack()[-6:])}")

        params = {'symbol': symbol}
        logger.warning(f"⚠️ Canceling ALL orders for {symbol}")
        return self._make_request('DELETE', '/fapi/v1/allOpenOrders', params=params, signed=True)

    def get_order(self, symbol: str, order_id: int) -> Dict:
        """
        Consulta estado de una orden

        Args:
            symbol: Símbolo
            order_id: ID de la orden

        Returns:
            Dict: Info de la orden
        """
        params = {'symbol': symbol, 'orderId': order_id}
        return self._make_request('GET', '/fapi/v1/order', params=params, signed=True)

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Obtiene órdenes abiertas

        Args:
            symbol: Símbolo opcional (si no se especifica, retorna todas)

        Returns:
            List: Lista de órdenes abiertas
        """
        params = {'symbol': symbol} if symbol else {}
        return self._make_request('GET', '/fapi/v1/openOrders', params=params, signed=True)

    def get_all_orders(self, symbol: str, limit: int = 10) -> List[Dict]:
        """
        Obtiene historial de órdenes (abiertas y cerradas)

        Args:
            symbol: Símbolo
            limit: Número máximo de órdenes a retornar (default 10, max 1000)

        Returns:
            List: Lista de órdenes ordenadas por tiempo (más recientes primero)
        """
        params = {'symbol': symbol, 'limit': limit}
        return self._make_request('GET', '/fapi/v1/allOrders', params=params, signed=True)

    # ========== ENDPOINTS PRIVADOS (Cuenta y Posiciones) ==========

    def change_leverage(self, symbol: str, leverage: int) -> Dict:
        """
        Cambia leverage de un símbolo

        Args:
            symbol: Símbolo
            leverage: Leverage (1-125 según el símbolo)

        Returns:
            Dict: Confirmación con leverage aplicado
        """
        if not (1 <= leverage <= 125):
            raise ValueError(f"Leverage debe estar entre 1-125, got {leverage}")

        params = {'symbol': symbol, 'leverage': leverage}
        logger.info(f"⚙️ Setting leverage {leverage}x for {symbol}")
        return self._make_request('POST', '/fapi/v1/leverage', params=params, signed=True)

    def change_margin_type(self, symbol: str, margin_type: str) -> Dict:
        """
        Cambia tipo de margen (ISOLATED o CROSS)

        Args:
            symbol: Símbolo
            margin_type: 'ISOLATED' o 'CROSS'

        Returns:
            Dict: Confirmación
        """
        if margin_type not in ['ISOLATED', 'CROSS']:
            raise ValueError(f"Margin type debe ser ISOLATED o CROSS, got {margin_type}")

        params = {'symbol': symbol, 'marginType': margin_type}
        logger.info(f"⚙️ Setting margin type {margin_type} for {symbol}")
        return self._make_request('POST', '/fapi/v1/marginType', params=params, signed=True)

    def get_position_risk(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Obtiene información de posiciones (v3)

        Args:
            symbol: Símbolo opcional

        Returns:
            List: Lista de todas las posiciones - SIEMPRE devuelve lista, nunca string
        """
        try:
            params = {'symbol': symbol} if symbol else {}
            result = self._make_request('GET', '/fapi/v3/positionRisk', params=params, signed=True)

            # Validar que result es una lista
            if not isinstance(result, list):
                logger.error(f"❌ get_position_risk devolvió tipo inesperado: {type(result)}")
                return []

            return result

        except Exception as e:
            logger.error(f"❌ Error en get_position_risk: {e}")
            return []  # SIEMPRE devolver lista vacía en error, NO string

    def get_balance(self) -> List[Dict]:
        """
        Obtiene balance de la cuenta (v3)

        Returns:
            List: Lista de balances por asset
        """
        return self._make_request('GET', '/fapi/v3/balance', signed=True)

    def get_account_info(self) -> Dict:
        """
        Obtiene información completa de la cuenta (v3)

        Returns:
            Dict: Info completa incluyendo balance, posiciones, etc.
        """
        return self._make_request('GET', '/fapi/v3/account', signed=True)

    def get_user_trades(
        self,
        symbol: str,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[Dict]:
        """
        Obtiene historial de trades del usuario

        Args:
            symbol: Símbolo
            limit: Número de trades (max 1000)
            start_time: Timestamp de inicio (opcional)
            end_time: Timestamp de fin (opcional)

        Returns:
            List: Lista de trades ejecutados
        """
        params = {'symbol': symbol, 'limit': min(limit, 1000)}
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time

        return self._make_request('GET', '/fapi/v1/userTrades', params=params, signed=True)

    def get_income_history(
        self,
        symbol: Optional[str] = None,
        income_type: Optional[str] = None,
        limit: int = 100,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[Dict]:
        """
        Obtiene historial de ingresos (P&L, funding, comisiones)

        Args:
            symbol: Símbolo opcional
            income_type: 'REALIZED_PNL', 'FUNDING_FEE', 'COMMISSION', etc.
            limit: Número de registros (max 1000)
            start_time: Timestamp de inicio (opcional)
            end_time: Timestamp de fin (opcional)

        Returns:
            List: Lista de ingresos
        """
        params = {'limit': min(limit, 1000)}
        if symbol:
            params['symbol'] = symbol
        if income_type:
            params['incomeType'] = income_type
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time

        return self._make_request('GET', '/fapi/v1/income', params=params, signed=True)

    def get_commission_rate(self, symbol: str) -> Dict:
        """
        Obtiene tasa de comisión del usuario

        Args:
            symbol: Símbolo

        Returns:
            Dict: Tasas maker/taker
        """
        params = {'symbol': symbol}
        return self._make_request('GET', '/fapi/v1/commissionRate', params=params, signed=True)

    # ========== WEBSOCKET HELPERS ==========

    def create_listen_key(self) -> str:
        """
        Crea listen key para User Data Stream

        Returns:
            str: Listen key
        """
        response = self._make_request('POST', '/fapi/v1/listenKey', signed=False)
        listen_key = response['listenKey']
        logger.info(f"🔑 Listen key created: {listen_key[:10]}...")
        return listen_key

    def keep_alive_listen_key(self, listen_key: str) -> Dict:
        """
        Mantiene vivo el listen key (cada 30 min)

        Args:
            listen_key: Listen key

        Returns:
            Dict: Confirmación
        """
        return self._make_request('PUT', '/fapi/v1/listenKey', signed=False)

    def close_listen_key(self, listen_key: str) -> Dict:
        """
        Cierra listen key

        Args:
            listen_key: Listen key

        Returns:
            Dict: Confirmación
        """
        logger.info("🔒 Closing listen key")
        return self._make_request('DELETE', '/fapi/v1/listenKey', signed=False)
