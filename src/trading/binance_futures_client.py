"""
Binance Futures Client - Cliente para la API de Binance USDT-M Futures
Maneja autenticacion, ordenes, posiciones y balance real

Documentacion API: https://developers.binance.com/docs/derivatives/usds-margined-futures
"""

import hmac
import hashlib
import time
import logging
import requests
from typing import Dict, List, Optional, Literal
from urllib.parse import urlencode
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    STOP = "STOP"
    STOP_MARKET = "STOP_MARKET"
    TAKE_PROFIT = "TAKE_PROFIT"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"
    TRAILING_STOP_MARKET = "TRAILING_STOP_MARKET"


class PositionSide(Enum):
    BOTH = "BOTH"  # One-way mode
    LONG = "LONG"  # Hedge mode
    SHORT = "SHORT"  # Hedge mode


class MarginType(Enum):
    ISOLATED = "ISOLATED"
    CROSSED = "CROSSED"


class TimeInForce(Enum):
    GTC = "GTC"  # Good Till Cancel
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill
    GTX = "GTX"  # Good Till Crossing (Post Only)


@dataclass
class OrderResult:
    """Resultado de una orden ejecutada"""
    order_id: int
    client_order_id: str
    symbol: str
    side: str
    type: str
    status: str
    price: float
    avg_price: float
    orig_qty: float
    executed_qty: float
    cum_quote: float
    time_in_force: str
    reduce_only: bool
    position_side: str
    working_type: str
    update_time: int
    raw_response: Dict


@dataclass
class PositionInfo:
    """Informacion de una posicion abierta"""
    symbol: str
    position_side: str
    position_amt: float
    entry_price: float
    mark_price: float
    unrealized_profit: float
    liquidation_price: float
    leverage: int
    margin_type: str
    isolated_margin: float
    notional: float
    update_time: int


@dataclass
class AccountBalance:
    """Balance de la cuenta"""
    asset: str
    balance: float
    available_balance: float
    cross_wallet_balance: float
    cross_unrealized_pnl: float
    max_withdraw_amount: float


class BinanceFuturesClient:
    """
    Cliente para Binance USDT-M Futures API

    Soporta:
    - Autenticacion HMAC-SHA256
    - Ordenes (MARKET, LIMIT, STOP_MARKET, TAKE_PROFIT_MARKET)
    - Gestion de posiciones
    - Consulta de balance
    - Cambio de leverage y margin type

    Uso:
        client = BinanceFuturesClient(api_key, api_secret)
        client.place_market_order("BTCUSDT", "BUY", 0.01)
    """

    # URLs de produccion y testnet
    BASE_URL_PRODUCTION = "https://fapi.binance.com"
    BASE_URL_TESTNET = "https://testnet.binancefuture.com"

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = False,
        proxy: Optional[Dict] = None,
        recv_window: int = 5000
    ):
        """
        Inicializa el cliente de Binance Futures

        Args:
            api_key: API Key de Binance
            api_secret: API Secret de Binance
            testnet: True para usar testnet, False para produccion
            proxy: Configuracion de proxy {'http': url, 'https': url}
            recv_window: Ventana de tiempo para validacion de requests (ms)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.base_url = self.BASE_URL_TESTNET if testnet else self.BASE_URL_PRODUCTION
        self.proxy = proxy
        self.recv_window = recv_window

        # Session para reutilizar conexiones
        self.session = requests.Session()
        self.session.headers.update({
            'X-MBX-APIKEY': self.api_key,
            'Content-Type': 'application/x-www-form-urlencoded'
        })

        if self.proxy:
            self.session.proxies.update(self.proxy)

        # Cache de informacion del exchange
        self._exchange_info_cache = None
        self._exchange_info_timestamp = 0

        logger.info(f"BinanceFuturesClient initialized - {'TESTNET' if testnet else 'PRODUCTION'}")

    def _get_timestamp(self) -> int:
        """Retorna timestamp actual en milliseconds"""
        return int(time.time() * 1000)

    def _sign_request(self, params: Dict) -> str:
        """
        Genera firma HMAC-SHA256 para los parametros

        Args:
            params: Parametros de la request

        Returns:
            Signature string
        """
        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature

    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        signed: bool = False
    ) -> Dict:
        """
        Realiza request a la API de Binance

        Args:
            method: HTTP method (GET, POST, DELETE)
            endpoint: API endpoint
            params: Parametros de la request
            signed: True si requiere firma

        Returns:
            Response JSON

        Raises:
            BinanceAPIError: Si la API retorna error
        """
        url = f"{self.base_url}{endpoint}"
        params = params or {}

        if signed:
            params['timestamp'] = self._get_timestamp()
            params['recvWindow'] = self.recv_window
            params['signature'] = self._sign_request(params)

        try:
            if method == 'GET':
                response = self.session.get(url, params=params, timeout=30)
            elif method == 'POST':
                response = self.session.post(url, data=params, timeout=30)
            elif method == 'DELETE':
                response = self.session.delete(url, params=params, timeout=30)
            elif method == 'PUT':
                response = self.session.put(url, data=params, timeout=30)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            # Log rate limit info
            used_weight = response.headers.get('X-MBX-USED-WEIGHT-1M', 'N/A')
            order_count = response.headers.get('X-MBX-ORDER-COUNT-1M', 'N/A')
            logger.debug(f"API Request: {method} {endpoint} | Weight: {used_weight} | Orders: {order_count}")

            # Handle response
            if response.status_code == 200:
                return response.json()
            else:
                error_data = response.json() if response.text else {}
                error_code = error_data.get('code', response.status_code)
                error_msg = error_data.get('msg', response.text)

                logger.error(f"Binance API Error: [{error_code}] {error_msg}")
                raise BinanceAPIError(error_code, error_msg)

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise BinanceConnectionError(str(e))

    # ==================== MARKET DATA (Public) ====================

    def get_server_time(self) -> int:
        """Obtiene tiempo del servidor de Binance"""
        response = self._make_request('GET', '/fapi/v1/time')
        return response['serverTime']

    def get_exchange_info(self, use_cache: bool = True) -> Dict:
        """
        Obtiene informacion del exchange (simbolos, reglas, etc)

        Args:
            use_cache: True para usar cache (5 minutos)
        """
        cache_duration = 300000  # 5 minutes in ms

        if use_cache and self._exchange_info_cache:
            if self._get_timestamp() - self._exchange_info_timestamp < cache_duration:
                return self._exchange_info_cache

        response = self._make_request('GET', '/fapi/v1/exchangeInfo')
        self._exchange_info_cache = response
        self._exchange_info_timestamp = self._get_timestamp()

        return response

    def get_ticker_price(self, symbol: Optional[str] = None) -> Dict:
        """Obtiene precio actual de un simbolo o todos"""
        params = {}
        if symbol:
            params['symbol'] = symbol
        return self._make_request('GET', '/fapi/v1/ticker/price', params)

    def get_mark_price(self, symbol: Optional[str] = None) -> Dict:
        """Obtiene mark price de un simbolo o todos"""
        params = {}
        if symbol:
            params['symbol'] = symbol
        return self._make_request('GET', '/fapi/v1/premiumIndex', params)

    def get_orderbook(self, symbol: str, limit: int = 100) -> Dict:
        """Obtiene order book de un simbolo"""
        params = {'symbol': symbol, 'limit': limit}
        return self._make_request('GET', '/fapi/v1/depth', params)

    def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 100,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List:
        """
        Obtiene velas (candlesticks)

        Args:
            symbol: Par de trading (ej. BTCUSDT)
            interval: Intervalo (1m, 5m, 15m, 1h, 4h, 1d, etc)
            limit: Numero de velas (max 1500)
        """
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time

        return self._make_request('GET', '/fapi/v1/klines', params)

    def get_funding_rate(self, symbol: str) -> Dict:
        """Obtiene funding rate actual de un simbolo"""
        return self._make_request('GET', '/fapi/v1/premiumIndex', {'symbol': symbol})

    # ==================== ACCOUNT DATA (Private) ====================

    def get_account_balance(self) -> List[AccountBalance]:
        """
        Obtiene balance de la cuenta

        Returns:
            Lista de AccountBalance para cada asset
        """
        response = self._make_request('GET', '/fapi/v2/balance', signed=True)

        balances = []
        for item in response:
            balances.append(AccountBalance(
                asset=item['asset'],
                balance=float(item['balance']),
                available_balance=float(item['availableBalance']),
                cross_wallet_balance=float(item['crossWalletBalance']),
                cross_unrealized_pnl=float(item['crossUnPnl']),
                max_withdraw_amount=float(item['maxWithdrawAmount'])
            ))

        return balances

    def get_usdt_balance(self) -> AccountBalance:
        """Obtiene balance de USDT especificamente"""
        balances = self.get_account_balance()
        for balance in balances:
            if balance.asset == 'USDT':
                return balance
        raise ValueError("USDT balance not found")

    def get_account_info(self) -> Dict:
        """Obtiene informacion completa de la cuenta (V2)"""
        return self._make_request('GET', '/fapi/v2/account', signed=True)

    def get_positions(self, symbol: Optional[str] = None) -> List[PositionInfo]:
        """
        Obtiene posiciones abiertas

        Args:
            symbol: Filtrar por simbolo (opcional)

        Returns:
            Lista de PositionInfo
        """
        params = {}
        if symbol:
            params['symbol'] = symbol

        response = self._make_request('GET', '/fapi/v2/positionRisk', params, signed=True)

        positions = []
        for item in response:
            # Solo incluir posiciones con cantidad != 0
            position_amt = float(item['positionAmt'])
            if position_amt != 0 or symbol:  # Incluir todas si se especifica symbol
                positions.append(PositionInfo(
                    symbol=item['symbol'],
                    position_side=item['positionSide'],
                    position_amt=position_amt,
                    entry_price=float(item['entryPrice']),
                    mark_price=float(item['markPrice']),
                    unrealized_profit=float(item['unRealizedProfit']),
                    liquidation_price=float(item['liquidationPrice']) if item['liquidationPrice'] != '0' else 0,
                    leverage=int(item['leverage']),
                    margin_type=item['marginType'],
                    isolated_margin=float(item['isolatedMargin']),
                    notional=float(item['notional']),
                    update_time=int(item['updateTime'])
                ))

        return positions

    def get_open_positions(self) -> List[PositionInfo]:
        """Obtiene solo posiciones con cantidad != 0"""
        all_positions = self.get_positions()
        return [p for p in all_positions if p.position_amt != 0]

    def get_position_mode(self) -> bool:
        """
        Obtiene el modo de posicion actual

        Returns:
            True = Hedge Mode (LONG/SHORT), False = One-way Mode (BOTH)
        """
        response = self._make_request('GET', '/fapi/v1/positionSide/dual', signed=True)
        return response['dualSidePosition']

    def set_position_mode(self, hedge_mode: bool) -> bool:
        """
        Configura el modo de posicion

        Args:
            hedge_mode: True = Hedge Mode, False = One-way Mode

        Returns:
            True si exitoso
        """
        params = {'dualSidePosition': 'true' if hedge_mode else 'false'}
        try:
            self._make_request('POST', '/fapi/v1/positionSide/dual', params, signed=True)
            logger.info(f"Position mode set to: {'Hedge' if hedge_mode else 'One-way'}")
            return True
        except BinanceAPIError as e:
            if e.code == -4059:  # No need to change
                logger.info("Position mode already set correctly")
                return True
            raise

    # ==================== LEVERAGE & MARGIN ====================

    def set_leverage(self, symbol: str, leverage: int) -> Dict:
        """
        Configura el leverage para un simbolo

        Args:
            symbol: Par de trading (ej. BTCUSDT)
            leverage: Nivel de leverage (1-125)

        Returns:
            Response con nuevo leverage
        """
        params = {
            'symbol': symbol,
            'leverage': leverage
        }
        response = self._make_request('POST', '/fapi/v1/leverage', params, signed=True)
        logger.info(f"Leverage set for {symbol}: {leverage}x")
        return response

    def set_margin_type(self, symbol: str, margin_type: MarginType) -> bool:
        """
        Configura el tipo de margen (ISOLATED o CROSSED)

        Args:
            symbol: Par de trading
            margin_type: ISOLATED o CROSSED
        """
        params = {
            'symbol': symbol,
            'marginType': margin_type.value
        }
        try:
            self._make_request('POST', '/fapi/v1/marginType', params, signed=True)
            logger.info(f"Margin type set for {symbol}: {margin_type.value}")
            return True
        except BinanceAPIError as e:
            if e.code == -4046:  # No need to change
                logger.info(f"Margin type for {symbol} already set to {margin_type.value}")
                return True
            raise

    # ==================== ORDERS ====================

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: Optional[float] = None,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        position_side: PositionSide = PositionSide.BOTH,
        time_in_force: Optional[TimeInForce] = None,
        reduce_only: bool = False,
        close_position: bool = False,
        working_type: str = "CONTRACT_PRICE",
        new_client_order_id: Optional[str] = None
    ) -> OrderResult:
        """
        Coloca una orden en el mercado

        Args:
            symbol: Par de trading (ej. BTCUSDT)
            side: BUY o SELL
            order_type: MARKET, LIMIT, STOP_MARKET, etc
            quantity: Cantidad a operar
            price: Precio (para LIMIT orders)
            stop_price: Precio de activacion (para STOP orders)
            position_side: BOTH, LONG, SHORT
            time_in_force: GTC, IOC, FOK, GTX
            reduce_only: True para solo reducir posicion
            close_position: True para cerrar toda la posicion
            working_type: CONTRACT_PRICE o MARK_PRICE
            new_client_order_id: ID personalizado para la orden

        Returns:
            OrderResult con detalles de la orden
        """
        params = {
            'symbol': symbol,
            'side': side.value,
            'type': order_type.value,
            'positionSide': position_side.value
        }

        # Agregar parametros opcionales segun tipo de orden
        if quantity and not close_position:
            params['quantity'] = self._format_quantity(symbol, quantity)

        if price:
            params['price'] = self._format_price(symbol, price)

        if stop_price:
            params['stopPrice'] = self._format_price(symbol, stop_price)

        if time_in_force:
            params['timeInForce'] = time_in_force.value
        elif order_type == OrderType.LIMIT:
            params['timeInForce'] = TimeInForce.GTC.value

        if reduce_only:
            params['reduceOnly'] = 'true'

        if close_position:
            params['closePosition'] = 'true'

        params['workingType'] = working_type

        if new_client_order_id:
            params['newClientOrderId'] = new_client_order_id

        response = self._make_request('POST', '/fapi/v1/order', params, signed=True)

        result = OrderResult(
            order_id=response['orderId'],
            client_order_id=response['clientOrderId'],
            symbol=response['symbol'],
            side=response['side'],
            type=response['type'],
            status=response['status'],
            price=float(response['price']),
            avg_price=float(response['avgPrice']),
            orig_qty=float(response['origQty']),
            executed_qty=float(response['executedQty']),
            cum_quote=float(response['cumQuote']),
            time_in_force=response.get('timeInForce', 'N/A'),
            reduce_only=response.get('reduceOnly', False),
            position_side=response['positionSide'],
            working_type=response.get('workingType', 'CONTRACT_PRICE'),
            update_time=response['updateTime'],
            raw_response=response
        )

        logger.info(
            f"Order placed: {result.symbol} {result.side} {result.type} "
            f"qty={result.orig_qty} price={result.price} status={result.status}"
        )

        return result

    def place_market_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        position_side: PositionSide = PositionSide.BOTH,
        reduce_only: bool = False
    ) -> OrderResult:
        """
        Coloca una orden de mercado (ejecucion inmediata)

        Args:
            symbol: Par de trading
            side: BUY o SELL
            quantity: Cantidad
            position_side: BOTH, LONG, SHORT
            reduce_only: True para solo reducir posicion
        """
        return self.place_order(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            position_side=position_side,
            reduce_only=reduce_only
        )

    def place_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        position_side: PositionSide = PositionSide.BOTH,
        time_in_force: TimeInForce = TimeInForce.GTC,
        reduce_only: bool = False
    ) -> OrderResult:
        """
        Coloca una orden limitada

        Args:
            symbol: Par de trading
            side: BUY o SELL
            quantity: Cantidad
            price: Precio limite
            position_side: BOTH, LONG, SHORT
            time_in_force: GTC, IOC, FOK, GTX
            reduce_only: True para solo reducir posicion
        """
        return self.place_order(
            symbol=symbol,
            side=side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=price,
            position_side=position_side,
            time_in_force=time_in_force,
            reduce_only=reduce_only
        )

    def place_stop_market_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        stop_price: float,
        position_side: PositionSide = PositionSide.BOTH,
        reduce_only: bool = True,
        close_position: bool = False
    ) -> OrderResult:
        """
        Coloca una orden STOP MARKET (para stop loss)

        Args:
            symbol: Par de trading
            side: BUY o SELL
            quantity: Cantidad (ignorada si close_position=True)
            stop_price: Precio de activacion
            position_side: BOTH, LONG, SHORT
            reduce_only: True para solo reducir posicion
            close_position: True para cerrar toda la posicion
        """
        return self.place_order(
            symbol=symbol,
            side=side,
            order_type=OrderType.STOP_MARKET,
            quantity=quantity if not close_position else None,
            stop_price=stop_price,
            position_side=position_side,
            reduce_only=reduce_only,
            close_position=close_position
        )

    def place_take_profit_market_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        stop_price: float,
        position_side: PositionSide = PositionSide.BOTH,
        reduce_only: bool = True,
        close_position: bool = False
    ) -> OrderResult:
        """
        Coloca una orden TAKE PROFIT MARKET

        Args:
            symbol: Par de trading
            side: BUY o SELL
            quantity: Cantidad
            stop_price: Precio de activacion
            position_side: BOTH, LONG, SHORT
            reduce_only: True para solo reducir posicion
            close_position: True para cerrar toda la posicion
        """
        return self.place_order(
            symbol=symbol,
            side=side,
            order_type=OrderType.TAKE_PROFIT_MARKET,
            quantity=quantity if not close_position else None,
            stop_price=stop_price,
            position_side=position_side,
            reduce_only=reduce_only,
            close_position=close_position
        )

    def cancel_order(self, symbol: str, order_id: Optional[int] = None,
                     client_order_id: Optional[str] = None) -> Dict:
        """
        Cancela una orden abierta

        Args:
            symbol: Par de trading
            order_id: ID de la orden (o usar client_order_id)
            client_order_id: ID de cliente de la orden
        """
        params = {'symbol': symbol}
        if order_id:
            params['orderId'] = order_id
        elif client_order_id:
            params['origClientOrderId'] = client_order_id
        else:
            raise ValueError("Must provide either order_id or client_order_id")

        response = self._make_request('DELETE', '/fapi/v1/order', params, signed=True)
        logger.info(f"Order cancelled: {symbol} orderId={order_id or client_order_id}")
        return response

    def cancel_all_orders(self, symbol: str) -> Dict:
        """Cancela todas las ordenes abiertas de un simbolo"""
        params = {'symbol': symbol}
        response = self._make_request('DELETE', '/fapi/v1/allOpenOrders', params, signed=True)
        logger.info(f"All orders cancelled for {symbol}")
        return response

    def get_order(self, symbol: str, order_id: Optional[int] = None,
                  client_order_id: Optional[str] = None) -> Dict:
        """Obtiene informacion de una orden"""
        params = {'symbol': symbol}
        if order_id:
            params['orderId'] = order_id
        elif client_order_id:
            params['origClientOrderId'] = client_order_id

        return self._make_request('GET', '/fapi/v1/order', params, signed=True)

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Obtiene todas las ordenes abiertas"""
        params = {}
        if symbol:
            params['symbol'] = symbol
        return self._make_request('GET', '/fapi/v1/openOrders', params, signed=True)

    def get_all_orders(self, symbol: str, limit: int = 500) -> List[Dict]:
        """Obtiene historial de ordenes de un simbolo"""
        params = {'symbol': symbol, 'limit': limit}
        return self._make_request('GET', '/fapi/v1/allOrders', params, signed=True)

    # ==================== POSITION CLOSE HELPERS ====================

    def close_position(
        self,
        symbol: str,
        position_side: PositionSide = PositionSide.BOTH
    ) -> Optional[OrderResult]:
        """
        Cierra completamente una posicion abierta

        Args:
            symbol: Par de trading
            position_side: Lado de la posicion a cerrar

        Returns:
            OrderResult si habia posicion, None si no habia
        """
        # Obtener posicion actual
        positions = self.get_positions(symbol)

        for pos in positions:
            if pos.position_side == position_side.value and pos.position_amt != 0:
                # Determinar side para cerrar
                if pos.position_amt > 0:
                    close_side = OrderSide.SELL
                    qty = abs(pos.position_amt)
                else:
                    close_side = OrderSide.BUY
                    qty = abs(pos.position_amt)

                return self.place_market_order(
                    symbol=symbol,
                    side=close_side,
                    quantity=qty,
                    position_side=position_side,
                    reduce_only=True
                )

        logger.info(f"No open position found for {symbol} {position_side.value}")
        return None

    def close_all_positions(self) -> List[OrderResult]:
        """Cierra todas las posiciones abiertas"""
        results = []
        open_positions = self.get_open_positions()

        for pos in open_positions:
            try:
                position_side = PositionSide(pos.position_side)
                result = self.close_position(pos.symbol, position_side)
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"Error closing position {pos.symbol}: {e}")

        return results

    # ==================== UTILITY METHODS ====================

    def _get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Obtiene informacion de un simbolo del cache"""
        exchange_info = self.get_exchange_info()
        for s in exchange_info['symbols']:
            if s['symbol'] == symbol:
                return s
        return None

    def _format_quantity(self, symbol: str, quantity: float) -> str:
        """Formatea cantidad segun precision del simbolo"""
        symbol_info = self._get_symbol_info(symbol)
        if symbol_info:
            for filter in symbol_info['filters']:
                if filter['filterType'] == 'LOT_SIZE':
                    step_size = float(filter['stepSize'])
                    precision = len(str(step_size).split('.')[-1].rstrip('0'))
                    return f"{quantity:.{precision}f}"
        return str(quantity)

    def _format_price(self, symbol: str, price: float) -> str:
        """Formatea precio segun precision del simbolo"""
        symbol_info = self._get_symbol_info(symbol)
        if symbol_info:
            for filter in symbol_info['filters']:
                if filter['filterType'] == 'PRICE_FILTER':
                    tick_size = float(filter['tickSize'])
                    precision = len(str(tick_size).split('.')[-1].rstrip('0'))
                    return f"{price:.{precision}f}"
        return str(price)

    # Mapeo de símbolos que usan formato 1000x en Binance Futures
    # Estas memecoins tienen precio muy bajo, así que Binance usa 1000 unidades por contrato
    FUTURES_SYMBOL_MAPPING = {
        'SHIB': '1000SHIB',
        'PEPE': '1000PEPE',
        'BONK': '1000BONK',
        'FLOKI': '1000FLOKI',
        'LUNC': '1000LUNC',
        'XEC': '1000XEC',
        'SATS': '1000SATS',
        'RATS': '1000RATS',
        'CAT': '1000CAT',
        'CHEEMS': '1000CHEEMS',
        'APU': '1000APU',
        'X': '1000X',
        'WHY': '1000WHY',
        'TURBO': '1000TURBO',
    }

    def convert_pair_format(self, pair: str) -> str:
        """
        Convierte formato de par (BTC/USDT -> BTCUSDT)
        Maneja símbolos especiales que usan formato 1000x en Futures

        Args:
            pair: Par en formato con slash

        Returns:
            Par en formato Binance Futures (sin slash, con prefijo 1000 si aplica)
        """
        # Separar base y quote
        if '/' in pair:
            base, quote = pair.split('/')
        else:
            # Ya está en formato Binance
            return pair

        # Verificar si necesita mapeo 1000x
        if base.upper() in self.FUTURES_SYMBOL_MAPPING:
            mapped_base = self.FUTURES_SYMBOL_MAPPING[base.upper()]
            result = f"{mapped_base}{quote}"
            logger.debug(f"Symbol mapping: {pair} -> {result}")
            return result

        return f"{base}{quote}"

    def convert_symbol_to_pair(self, symbol: str) -> str:
        """
        Convierte formato Binance Futures a par (BTCUSDT -> BTC/USDT)
        Maneja símbolos especiales que usan formato 1000x

        Args:
            symbol: Símbolo en formato Binance (ej. 1000SHIBUSDT)

        Returns:
            Par en formato con slash (ej. SHIB/USDT)
        """
        if not symbol.endswith('USDT'):
            return symbol

        base = symbol[:-4]  # Remove USDT

        # Verificar si tiene prefijo 1000
        if base.startswith('1000'):
            original_base = base[4:]  # Remove '1000' prefix
            # Verificar que está en nuestro mapeo
            if original_base in self.FUTURES_SYMBOL_MAPPING.values() or \
               f"1000{original_base}" == self.FUTURES_SYMBOL_MAPPING.get(original_base, ''):
                base = original_base

        return f"{base}/USDT"

    def validate_connection(self) -> bool:
        """
        Valida la conexion con la API

        Returns:
            True si la conexion es valida
        """
        try:
            # Test public endpoint
            self.get_server_time()

            # Test private endpoint (requires valid API key)
            self.get_account_balance()

            logger.info("Binance Futures connection validated successfully")
            return True

        except Exception as e:
            logger.error(f"Connection validation failed: {e}")
            return False

    def get_commission_rate(self, symbol: str) -> Dict:
        """Obtiene tasa de comision para un simbolo"""
        params = {'symbol': symbol}
        return self._make_request('GET', '/fapi/v1/commissionRate', params, signed=True)


class BinanceAPIError(Exception):
    """Error de la API de Binance"""
    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message
        super().__init__(f"Binance API Error [{code}]: {message}")


class BinanceConnectionError(Exception):
    """Error de conexion con Binance"""
    pass
