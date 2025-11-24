"""
Binance Endpoints Validator
Valida TODOS los endpoints necesarios sin ejecutar trades reales
"""

import time
import hmac
import hashlib
import requests
import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class BinanceEndpointValidator:
    """
    Validador completo de endpoints de Binance USD-M Futures
    """

    def __init__(self, api_key: str, api_secret: str, base_url: str):
        """
        Args:
            api_key: Binance API Key
            api_secret: Binance API Secret
            base_url: Base URL de la API (testnet o live)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip('/')

        self.results = {
            'passed': [],
            'failed': [],
            'warnings': []
        }

    def _generate_signature(self, params: str) -> str:
        """Generar firma HMAC SHA256"""
        return hmac.new(
            self.api_secret.encode('utf-8'),
            params.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Dict = None,
        signed: bool = False
    ) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        Hacer request a Binance

        Returns:
            (success, data, error_message)
        """
        url = f"{self.base_url}{endpoint}"
        headers = {"X-MBX-APIKEY": self.api_key} if self.api_key else {}

        if params is None:
            params = {}

        if signed:
            timestamp = int(time.time() * 1000)
            params['timestamp'] = timestamp
            query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
            signature = self._generate_signature(query_string)
            params['signature'] = signature

        try:
            if method == 'GET':
                response = requests.get(url, params=params, headers=headers, timeout=10)
            elif method == 'POST':
                response = requests.post(url, params=params, headers=headers, timeout=10)
            elif method == 'PUT':
                response = requests.put(url, params=params, headers=headers, timeout=10)
            elif method == 'DELETE':
                response = requests.delete(url, params=params, headers=headers, timeout=10)
            else:
                return False, None, f"Método HTTP inválido: {method}"

            if response.status_code == 200:
                return True, response.json(), None
            else:
                error_data = response.json() if response.text else {}
                error_msg = error_data.get('msg', response.text)
                return False, None, f"HTTP {response.status_code}: {error_msg}"

        except requests.exceptions.Timeout:
            return False, None, "Request timeout (>10s)"
        except requests.exceptions.ConnectionError:
            return False, None, "Connection error"
        except Exception as e:
            return False, None, f"Exception: {str(e)}"

    def _log_test(self, test_name: str, success: bool, details: str = ""):
        """Log resultado de un test"""
        if success:
            self.results['passed'].append(test_name)
            logger.info(f"✅ {test_name} - {details}")
        else:
            self.results['failed'].append(test_name)
            logger.error(f"❌ {test_name} - {details}")

    # =====================================================
    # TESTS DE CONECTIVIDAD Y AUTENTICACIÓN
    # =====================================================

    def test_1_connectivity(self) -> bool:
        """Test 1: Conectividad básica (ping)"""
        success, data, error = self._make_request('GET', '/fapi/v1/ping')

        if success:
            self._log_test("1. Connectivity (Ping)", True, "Server respondió correctamente")
        else:
            self._log_test("1. Connectivity (Ping)", False, error)

        return success

    def test_2_server_time(self) -> bool:
        """Test 2: Server time sync"""
        success, data, error = self._make_request('GET', '/fapi/v1/time')

        if success:
            server_time = data.get('serverTime', 0)
            local_time = int(time.time() * 1000)
            offset = abs(server_time - local_time)

            if offset > 5000:
                self._log_test("2. Server Time Sync", False, f"Offset muy grande: {offset}ms")
                return False

            self._log_test("2. Server Time Sync", True, f"Offset: {offset}ms")
        else:
            self._log_test("2. Server Time Sync", False, error)

        return success

    def test_3_authentication(self) -> bool:
        """Test 3: API Key authentication"""
        success, data, error = self._make_request('GET', '/fapi/v3/balance', signed=True)

        if success:
            self._log_test("3. API Authentication", True, "API Key válida y autorizada")
        else:
            self._log_test("3. API Authentication", False, error)

        return success

    # =====================================================
    # TESTS DE MARKET DATA
    # =====================================================

    def test_4_exchange_info(self) -> bool:
        """Test 4: Exchange info (symbols, filters)"""
        success, data, error = self._make_request('GET', '/fapi/v1/exchangeInfo')

        if success:
            symbols = len(data.get('symbols', []))
            self._log_test("4. Exchange Info", True, f"Obtenidos {symbols} símbolos")
        else:
            self._log_test("4. Exchange Info", False, error)

        return success

    def test_5_ticker_price(self) -> bool:
        """Test 5: Ticker price (BTCUSDT)"""
        success, data, error = self._make_request('GET', '/fapi/v2/ticker/price', {'symbol': 'BTCUSDT'})

        if success:
            price = data.get('price', 0)
            self._log_test("5. Ticker Price", True, f"BTCUSDT: ${float(price):,.2f}")
        else:
            self._log_test("5. Ticker Price", False, error)

        return success

    def test_6_orderbook(self) -> bool:
        """Test 6: Order book depth"""
        success, data, error = self._make_request('GET', '/fapi/v1/depth', {'symbol': 'BTCUSDT', 'limit': 100})

        if success:
            bids = len(data.get('bids', []))
            asks = len(data.get('asks', []))
            self._log_test("6. Order Book", True, f"{bids} bids, {asks} asks")
        else:
            self._log_test("6. Order Book", False, error)

        return success

    def test_7_klines(self) -> bool:
        """Test 7: Candlestick data (klines)"""
        success, data, error = self._make_request(
            'GET', '/fapi/v1/klines',
            {'symbol': 'BTCUSDT', 'interval': '1h', 'limit': 100}
        )

        if success:
            candles = len(data) if isinstance(data, list) else 0
            self._log_test("7. Klines", True, f"{candles} velas obtenidas")
        else:
            self._log_test("7. Klines", False, error)

        return success

    def test_8_funding_rate(self) -> bool:
        """Test 8: Funding rate"""
        success, data, error = self._make_request('GET', '/fapi/v1/premiumIndex', {'symbol': 'BTCUSDT'})

        if success:
            funding_rate = float(data.get('lastFundingRate', 0)) * 100
            self._log_test("8. Funding Rate", True, f"{funding_rate:.4f}%")
        else:
            self._log_test("8. Funding Rate", False, error)

        return success

    # =====================================================
    # TESTS DE ACCOUNT & BALANCE
    # =====================================================

    def test_9_account_balance(self) -> bool:
        """Test 9: Account balance"""
        success, data, error = self._make_request('GET', '/fapi/v3/balance', signed=True)

        if success:
            usdt_balance = None
            for asset in data:
                if asset.get('asset') == 'USDT':
                    usdt_balance = float(asset.get('availableBalance', 0))
                    break

            if usdt_balance is not None:
                self._log_test("9. Account Balance", True, f"USDT: ${usdt_balance:,.2f}")
            else:
                self._log_test("9. Account Balance", True, "Balance obtenido")
        else:
            self._log_test("9. Account Balance", False, error)

        return success

    def test_10_account_info(self) -> bool:
        """Test 10: Account information"""
        success, data, error = self._make_request('GET', '/fapi/v3/account', signed=True)

        if success:
            total_margin = float(data.get('totalMarginBalance', 0))
            self._log_test("10. Account Info", True, f"Margin: ${total_margin:,.2f}")
        else:
            self._log_test("10. Account Info", False, error)

        return success

    def test_11_position_info(self) -> bool:
        """Test 11: Position information"""
        success, data, error = self._make_request('GET', '/fapi/v2/positionRisk', signed=True)

        if success:
            open_positions = [p for p in data if float(p.get('positionAmt', 0)) != 0]
            self._log_test("11. Position Info", True, f"{len(open_positions)} posiciones abiertas")
        else:
            self._log_test("11. Position Info", False, error)

        return success

    # =====================================================
    # TESTS DE TRADING (SIN EJECUTAR)
    # =====================================================

    def test_12_leverage_change(self) -> bool:
        """Test 12: Change leverage (BTCUSDT)"""
        success, data, error = self._make_request(
            'POST', '/fapi/v1/leverage',
            {'symbol': 'BTCUSDT', 'leverage': 2},
            signed=True
        )

        if success:
            leverage = data.get('leverage', 0)
            self._log_test("12. Change Leverage", True, f"Leverage: {leverage}x")
        else:
            self._log_test("12. Change Leverage", False, error)

        return success

    def test_13_margin_type(self) -> bool:
        """Test 13: Change margin type (ISOLATED)"""
        success, data, error = self._make_request(
            'POST', '/fapi/v1/marginType',
            {'symbol': 'BTCUSDT', 'marginType': 'ISOLATED'},
            signed=True
        )

        # Puede fallar si ya está en ISOLATED
        if success:
            self._log_test("13. Change Margin Type", True, "Margin type configurado")
        elif "No need to change margin type" in str(error):
            self._log_test("13. Change Margin Type", True, "Ya estaba en ISOLATED")
        else:
            self._log_test("13. Change Margin Type", False, error)

        return success or "No need to change margin type" in str(error)

    def test_14_order_test(self) -> bool:
        """Test 14: Test order (NO SE EJECUTA)"""
        # Obtener precio actual
        success_price, price_data, _ = self._make_request('GET', '/fapi/v2/ticker/price', {'symbol': 'BTCUSDT'})

        if not success_price:
            self._log_test("14. Test Order", False, "No se pudo obtener precio")
            return False

        current_price = float(price_data.get('price', 0))

        # Test order
        success, data, error = self._make_request(
            'POST', '/fapi/v1/order/test',
            {
                'symbol': 'BTCUSDT',
                'side': 'BUY',
                'type': 'LIMIT',
                'timeInForce': 'GTC',
                'quantity': 0.001,
                'price': round(current_price * 0.95, 2)
            },
            signed=True
        )

        if success:
            self._log_test("14. Test Order", True, "Orden test validada")
        else:
            self._log_test("14. Test Order", False, error)

        return success

    def test_15_open_orders(self) -> bool:
        """Test 15: Get open orders"""
        success, data, error = self._make_request(
            'GET', '/fapi/v1/openOrders',
            {'symbol': 'BTCUSDT'},
            signed=True
        )

        if success:
            open_orders = len(data) if isinstance(data, list) else 0
            self._log_test("15. Open Orders", True, f"{open_orders} órdenes abiertas")
        else:
            self._log_test("15. Open Orders", False, error)

        return success

    # =====================================================
    # TESTS DE USER DATA STREAM (WebSocket)
    # =====================================================

    def test_16_user_data_stream(self) -> bool:
        """Test 16: Create user data stream"""
        success, data, error = self._make_request('POST', '/fapi/v1/listenKey', signed=True)

        if success:
            listen_key = data.get('listenKey', '')
            self._log_test("16. User Data Stream", True, f"Listen key: {listen_key[:20]}...")

            # Test keep-alive
            success_keepalive, _, _ = self._make_request('PUT', '/fapi/v1/listenKey', signed=True)

            if success_keepalive:
                self._log_test("16a. Keep-Alive", True, "Funcionando")
            else:
                self._log_test("16a. Keep-Alive", False, "Keep-alive falló")

            return success_keepalive
        else:
            self._log_test("16. User Data Stream", False, error)

        return success

    # =====================================================
    # TESTS DE RATE LIMITS
    # =====================================================

    def test_17_rate_limits(self) -> bool:
        """Test 17: Rate limit info"""
        headers_captured = None

        for i in range(3):
            url = f"{self.base_url}/fapi/v1/time"
            response = requests.get(url, headers={"X-MBX-APIKEY": self.api_key})

            if response.status_code == 200:
                headers_captured = response.headers
                break

        if headers_captured:
            weight_used = headers_captured.get('X-MBX-USED-WEIGHT-1M', 'N/A')
            order_count = headers_captured.get('X-MBX-ORDER-COUNT-1M', 'N/A')

            self._log_test("17. Rate Limits", True, f"Weight: {weight_used}, Orders: {order_count}")
            return True
        else:
            self._log_test("17. Rate Limits", False, "No se pudieron capturar headers")
            return False

    # =====================================================
    # EJECUTAR TODOS LOS TESTS
    # =====================================================

    def run_all_tests(self) -> Dict:
        """
        Ejecutar todos los tests en secuencia

        Returns:
            Dict con resultados: {
                'total': int,
                'passed': int,
                'failed': int,
                'success_rate': float,
                'ready_for_production': bool,
                'passed_tests': List[str],
                'failed_tests': List[str]
            }
        """
        tests = [
            self.test_1_connectivity,
            self.test_2_server_time,
            self.test_3_authentication,
            self.test_4_exchange_info,
            self.test_5_ticker_price,
            self.test_6_orderbook,
            self.test_7_klines,
            self.test_8_funding_rate,
            self.test_9_account_balance,
            self.test_10_account_info,
            self.test_11_position_info,
            self.test_12_leverage_change,
            self.test_13_margin_type,
            self.test_14_order_test,
            self.test_15_open_orders,
            self.test_16_user_data_stream,
            self.test_17_rate_limits,
        ]

        for test in tests:
            try:
                test()
            except Exception as e:
                logger.error(f"Error ejecutando test {test.__name__}: {e}")
                self.results['failed'].append(test.__name__)

        total = len(self.results['passed']) + len(self.results['failed'])
        passed = len(self.results['passed'])
        failed = len(self.results['failed'])
        success_rate = (passed / total * 100) if total > 0 else 0

        return {
            'total': total,
            'passed': passed,
            'failed': failed,
            'success_rate': success_rate,
            'ready_for_production': failed == 0,
            'passed_tests': self.results['passed'],
            'failed_tests': self.results['failed']
        }
