"""
Utilidades para Binance Futures API
- Firma HMAC SHA256
- Validaciones de precio/cantidad
- Manejo de timestamps
"""

import hashlib
import hmac
import time
import math
import logging
from typing import Dict, Optional, Union
from urllib.parse import urlencode

logger = logging.getLogger(__name__)


def get_timestamp() -> int:
    """
    Obtiene timestamp actual en milisegundos

    Returns:
        int: Timestamp en milisegundos
    """
    return int(time.time() * 1000)


def sign_request(query_string: str, api_secret: str) -> str:
    """
    Firma una request con HMAC SHA256

    Args:
        query_string: Query string completo (ej: "symbol=BTCUSDT&timestamp=1234567890")
        api_secret: API Secret de Binance

    Returns:
        str: Signature en hexadecimal

    Example:
        >>> sign_request("symbol=BTCUSDT&timestamp=1234567890", "my_secret")
        'abc123def456...'
    """
    return hmac.new(
        api_secret.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()


def build_signed_request(params: Dict, api_secret: str, timestamp: Optional[int] = None) -> str:
    """
    Construye query string firmado para requests privados

    Args:
        params: Parámetros de la request
        api_secret: API Secret de Binance
        timestamp: Timestamp opcional (usa actual si no se provee)

    Returns:
        str: Query string firmado con signature

    Example:
        >>> build_signed_request({"symbol": "BTCUSDT"}, "secret")
        'symbol=BTCUSDT&timestamp=1234567890&signature=abc123...'
    """
    # Añadir timestamp si no existe
    if timestamp is None:
        timestamp = get_timestamp()
    params['timestamp'] = timestamp

    # Crear query string
    query_string = urlencode(params)

    # Firmar
    signature = sign_request(query_string, api_secret)

    # Retornar con signature
    return f"{query_string}&signature={signature}"


def validate_price(price: float, symbol: str = "") -> bool:
    """
    Valida que un precio es válido

    Args:
        price: Precio a validar
        symbol: Símbolo (para logging)

    Returns:
        bool: True si es válido, False si no
    """
    if price is None:
        logger.error(f"❌ Precio None para {symbol}")
        return False

    if math.isnan(price) or math.isinf(price):
        logger.error(f"❌ Precio NaN/Inf para {symbol}: {price}")
        return False

    if price <= 0:
        logger.error(f"❌ Precio <= 0 para {symbol}: {price}")
        return False

    return True


def validate_quantity(quantity: float, symbol: str = "") -> bool:
    """
    Valida que una cantidad es válida

    Args:
        quantity: Cantidad a validar
        symbol: Símbolo (para logging)

    Returns:
        bool: True si es válida, False si no
    """
    if quantity is None:
        logger.error(f"❌ Cantidad None para {symbol}")
        return False

    if math.isnan(quantity) or math.isinf(quantity):
        logger.error(f"❌ Cantidad NaN/Inf para {symbol}: {quantity}")
        return False

    if quantity <= 0:
        logger.error(f"❌ Cantidad <= 0 para {symbol}: {quantity}")
        return False

    return True


def round_step_size(quantity: float, step_size: float) -> float:
    """
    Redondea cantidad al stepSize del símbolo

    CRÍTICO: Usa round() y precisión para evitar errores de punto flotante
    que causan error -1111 en Binance.

    Args:
        quantity: Cantidad a redondear
        step_size: Step size del símbolo

    Returns:
        float: Cantidad redondeada exactamente al step_size

    Example:
        >>> round_step_size(0.123456, 0.001)
        0.123
        >>> round_step_size(0.123456, 0.01)
        0.12
    """
    # Calcular número de decimales necesarios basado en step_size
    if step_size >= 1:
        decimals = 0
    else:
        # Contar decimales del step_size
        decimals = len(str(step_size).rstrip('0').split('.')[-1])

    # Redondear hacia abajo (floor) pero con precisión correcta
    # Esto elimina errores de punto flotante
    return round(math.floor(quantity / step_size) * step_size, decimals)


def round_tick_size(price: float, tick_size: float) -> float:
    """
    Redondea precio al tickSize del símbolo

    CRÍTICO: Usa round() en lugar de floor() para evitar errores de precisión
    de punto flotante que causan error -1111 en Binance.

    Args:
        price: Precio a redondear
        tick_size: Tick size del símbolo

    Returns:
        float: Precio redondeado exactamente al tick_size

    Example:
        >>> round_tick_size(1234.5678, 0.01)
        1234.57
        >>> round_tick_size(88360.70000000001, 0.01)
        88360.70  # SIN errores de punto flotante
    """
    # Calcular número de decimales necesarios basado en tick_size
    # Ejemplo: tick_size=0.01 -> 2 decimales, tick_size=0.1 -> 1 decimal
    if tick_size >= 1:
        decimals = 0
    else:
        # Contar decimales del tick_size
        decimals = len(str(tick_size).rstrip('0').split('.')[-1])

    # Redondear a la precisión correcta y devolver float
    # Esto elimina el problema de 88360.70000000001
    return round(round(price / tick_size) * tick_size, decimals)


def format_quantity(quantity: float, precision: int = 8) -> str:
    """
    Formatea cantidad eliminando trailing zeros

    Args:
        quantity: Cantidad a formatear
        precision: Precisión decimal

    Returns:
        str: Cantidad formateada

    Example:
        >>> format_quantity(0.12300000, 8)
        '0.123'
        >>> format_quantity(1.00000000, 8)
        '1'
    """
    return f"{quantity:.{precision}f}".rstrip('0').rstrip('.')


def calculate_quantity(
    usdt_amount: float,
    price: float,
    leverage: int = 1,
    step_size: float = 0.001
) -> float:
    """
    Calcula cantidad de cripto basado en USDT y leverage

    Args:
        usdt_amount: Cantidad en USDT a usar
        price: Precio actual del par
        leverage: Leverage a usar (default 1x)
        step_size: Step size del símbolo

    Returns:
        float: Cantidad de cripto redondeada

    Example:
        >>> calculate_quantity(100, 50000, leverage=3, step_size=0.001)
        0.006  # (100 * 3) / 50000 = 0.006
    """
    # Calcular cantidad raw
    raw_quantity = (usdt_amount * leverage) / price

    # Redondear a step size
    rounded = round_step_size(raw_quantity, step_size)

    return rounded


def parse_binance_error(error_response: Dict) -> str:
    """
    Parsea error de Binance a mensaje legible

    Args:
        error_response: Respuesta de error de Binance API

    Returns:
        str: Mensaje de error formateado

    Example:
        >>> parse_binance_error({"code": -2010, "msg": "Account has insufficient balance"})
        '[-2010] Account has insufficient balance'
    """
    code = error_response.get('code', 'UNKNOWN')
    msg = error_response.get('msg', 'Unknown error')
    return f"[{code}] {msg}"


def get_error_description(error_code: int) -> str:
    """
    Obtiene descripción detallada de código de error de Binance

    Args:
        error_code: Código de error de Binance

    Returns:
        str: Descripción del error
    """
    error_descriptions = {
        -1000: "Unknown error",
        -1001: "Disconnected",
        -1002: "Unauthorized",
        -1003: "Too many requests (rate limit)",
        -1006: "Unexpected response",
        -1007: "Timeout",
        -1014: "Unknown order composition",
        -1015: "Too many orders",
        -1016: "Service shutting down",
        -1020: "Unsupported operation",
        -1021: "Timestamp out of sync (>1000ms)",
        -1022: "Invalid signature",
        -1100: "Illegal characters in parameter",
        -1101: "Too many parameters",
        -1102: "Mandatory parameter missing",
        -1103: "Unknown parameter",
        -1104: "Unread parameters",
        -1105: "Empty parameter",
        -1106: "Parameter not required",
        -1111: "Precision too high (too many decimals in price/quantity)",
        -2010: "Insufficient balance",
        -2011: "Margin call",
        -2013: "No such order",
        -2014: "Invalid API key",
        -2015: "Invalid API key, IP, or permissions for action",
        -4000: "Invalid order status",
        -4001: "Price out of range",
        -4002: "Quantity out of range",
        -4003: "Price less than or equal to stop price",
        -4004: "Price greater than or equal to stop price",
        -4044: "Order would immediately trigger",
        -4046: "No need to change margin type (already set)",
        -4164: "Position side does not match user settings",
    }

    return error_descriptions.get(error_code, f"Unknown error code: {error_code}")
