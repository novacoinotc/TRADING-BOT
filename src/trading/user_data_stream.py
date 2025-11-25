"""
User Data Stream - WebSocket para actualizaciones en tiempo real de la cuenta de Binance Futures

Maneja:
- Actualizaciones de balance y posiciones (ACCOUNT_UPDATE)
- Actualizaciones de ordenes (ORDER_TRADE_UPDATE)
- Margin calls (MARGIN_CALL)
- Expiracion de listenKey

Documentacion: https://developers.binance.com/docs/derivatives/usds-margined-futures/user-data-streams
"""

import asyncio
import json
import logging
import time
import websockets
from typing import Dict, Callable, Optional, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum

if TYPE_CHECKING:
    from src.trading.binance_futures_client import BinanceFuturesClient

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Tipos de eventos del User Data Stream"""
    ACCOUNT_UPDATE = "ACCOUNT_UPDATE"
    ORDER_TRADE_UPDATE = "ORDER_TRADE_UPDATE"
    MARGIN_CALL = "MARGIN_CALL"
    LISTEN_KEY_EXPIRED = "listenKeyExpired"
    ACCOUNT_CONFIG_UPDATE = "ACCOUNT_CONFIG_UPDATE"


@dataclass
class AccountUpdate:
    """Actualizacion de cuenta (balance y posiciones)"""
    event_time: int
    transaction_time: int
    balances: list
    positions: list
    event_reason: str  # DEPOSIT, WITHDRAW, ORDER, FUNDING_FEE, etc


@dataclass
class OrderUpdate:
    """Actualizacion de orden"""
    event_time: int
    transaction_time: int
    symbol: str
    client_order_id: str
    side: str
    order_type: str
    time_in_force: str
    original_qty: float
    original_price: float
    avg_price: float
    stop_price: float
    execution_type: str
    order_status: str
    order_id: int
    last_filled_qty: float
    cumulative_filled_qty: float
    last_filled_price: float
    commission: float
    commission_asset: str
    trade_time: int
    trade_id: int
    position_side: str
    realized_profit: float


@dataclass
class MarginCallEvent:
    """Evento de margin call"""
    event_time: int
    cross_wallet_balance: float
    positions: list  # Lista de posiciones en riesgo


class UserDataStream:
    """
    Cliente WebSocket para User Data Stream de Binance Futures

    Uso:
        stream = UserDataStream(binance_client)
        stream.on_account_update = my_account_handler
        stream.on_order_update = my_order_handler
        await stream.start()
    """

    # WebSocket URLs
    WS_URL_PRODUCTION = "wss://fstream.binance.com/ws/"
    WS_URL_TESTNET = "wss://stream.binancefuture.com/ws/"

    def __init__(
        self,
        binance_client: 'BinanceFuturesClient',
        testnet: bool = False
    ):
        """
        Inicializa el User Data Stream

        Args:
            binance_client: Cliente de Binance Futures
            testnet: True para usar testnet
        """
        self.client = binance_client
        self.testnet = testnet
        self.ws_url_base = self.WS_URL_TESTNET if testnet else self.WS_URL_PRODUCTION

        # WebSocket state
        self.listen_key: Optional[str] = None
        self.websocket = None
        self.is_running = False
        self.reconnect_delay = 5  # seconds
        self.max_reconnect_attempts = 10

        # Keepalive task
        self.keepalive_task = None
        self.keepalive_interval = 30 * 60  # 30 minutes (listenKey expires in 60 min)

        # Event handlers (to be set by user)
        self.on_account_update: Optional[Callable[[AccountUpdate], None]] = None
        self.on_order_update: Optional[Callable[[OrderUpdate], None]] = None
        self.on_margin_call: Optional[Callable[[MarginCallEvent], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None
        self.on_disconnect: Optional[Callable[[], None]] = None
        self.on_reconnect: Optional[Callable[[], None]] = None

        logger.info(f"UserDataStream initialized - {'TESTNET' if testnet else 'PRODUCTION'}")

    def _get_listen_key(self) -> str:
        """Obtiene un listenKey del API de Binance"""
        response = self.client._make_request(
            'POST',
            '/fapi/v1/listenKey',
            signed=False  # ListenKey endpoint uses API key in header, not signature
        )
        return response['listenKey']

    def _keepalive_listen_key(self):
        """Mantiene vivo el listenKey (debe llamarse cada 60 minutos max)"""
        try:
            self.client._make_request(
                'PUT',
                '/fapi/v1/listenKey',
                signed=False
            )
            logger.debug("ListenKey keepalive sent")
        except Exception as e:
            logger.error(f"Error in listenKey keepalive: {e}")

    def _close_listen_key(self):
        """Cierra el listenKey cuando ya no se necesita"""
        try:
            self.client._make_request(
                'DELETE',
                '/fapi/v1/listenKey',
                signed=False
            )
            logger.debug("ListenKey closed")
        except Exception as e:
            logger.error(f"Error closing listenKey: {e}")

    async def _keepalive_loop(self):
        """Loop de keepalive para el listenKey"""
        while self.is_running:
            await asyncio.sleep(self.keepalive_interval)
            if self.is_running:
                self._keepalive_listen_key()

    def _parse_account_update(self, data: Dict) -> AccountUpdate:
        """Parsea evento ACCOUNT_UPDATE"""
        return AccountUpdate(
            event_time=data['E'],
            transaction_time=data['T'],
            balances=[{
                'asset': b['a'],
                'wallet_balance': float(b['wb']),
                'cross_wallet_balance': float(b['cw']),
                'balance_change': float(b['bc'])
            } for b in data['a']['B']],
            positions=[{
                'symbol': p['s'],
                'position_amount': float(p['pa']),
                'entry_price': float(p['ep']),
                'unrealized_profit': float(p['up']),
                'margin_type': p['mt'],
                'isolated_wallet': float(p.get('iw', 0)),
                'position_side': p['ps']
            } for p in data['a']['P']],
            event_reason=data['a']['m']
        )

    def _parse_order_update(self, data: Dict) -> OrderUpdate:
        """Parsea evento ORDER_TRADE_UPDATE"""
        o = data['o']
        return OrderUpdate(
            event_time=data['E'],
            transaction_time=data['T'],
            symbol=o['s'],
            client_order_id=o['c'],
            side=o['S'],
            order_type=o['o'],
            time_in_force=o['f'],
            original_qty=float(o['q']),
            original_price=float(o['p']),
            avg_price=float(o['ap']),
            stop_price=float(o['sp']),
            execution_type=o['x'],
            order_status=o['X'],
            order_id=o['i'],
            last_filled_qty=float(o['l']),
            cumulative_filled_qty=float(o['z']),
            last_filled_price=float(o['L']),
            commission=float(o['n']),
            commission_asset=o.get('N', ''),
            trade_time=o['T'],
            trade_id=o['t'],
            position_side=o['ps'],
            realized_profit=float(o['rp'])
        )

    def _parse_margin_call(self, data: Dict) -> MarginCallEvent:
        """Parsea evento MARGIN_CALL"""
        return MarginCallEvent(
            event_time=data['E'],
            cross_wallet_balance=float(data['cw']),
            positions=[{
                'symbol': p['s'],
                'position_side': p['ps'],
                'position_amount': float(p['pa']),
                'margin_type': p['mt'],
                'isolated_wallet': float(p.get('iw', 0)),
                'mark_price': float(p['mp']),
                'unrealized_profit': float(p['up']),
                'maintenance_margin_required': float(p['mm'])
            } for p in data['p']]
        )

    async def _handle_message(self, message: str):
        """Procesa mensaje recibido del WebSocket"""
        try:
            data = json.loads(message)
            event_type = data.get('e')

            if event_type == EventType.ACCOUNT_UPDATE.value:
                logger.debug("ACCOUNT_UPDATE received")
                account_update = self._parse_account_update(data)

                if self.on_account_update:
                    try:
                        self.on_account_update(account_update)
                    except Exception as e:
                        logger.error(f"Error in on_account_update handler: {e}")

            elif event_type == EventType.ORDER_TRADE_UPDATE.value:
                logger.debug(f"ORDER_TRADE_UPDATE received: {data['o']['s']} {data['o']['X']}")
                order_update = self._parse_order_update(data)

                if self.on_order_update:
                    try:
                        self.on_order_update(order_update)
                    except Exception as e:
                        logger.error(f"Error in on_order_update handler: {e}")

            elif event_type == EventType.MARGIN_CALL.value:
                logger.warning("MARGIN_CALL received!")
                margin_call = self._parse_margin_call(data)

                if self.on_margin_call:
                    try:
                        self.on_margin_call(margin_call)
                    except Exception as e:
                        logger.error(f"Error in on_margin_call handler: {e}")

            elif event_type == EventType.LISTEN_KEY_EXPIRED.value:
                logger.warning("ListenKey expired, reconnecting...")
                await self._reconnect()

            elif event_type == EventType.ACCOUNT_CONFIG_UPDATE.value:
                logger.info(f"ACCOUNT_CONFIG_UPDATE: {data}")

            else:
                logger.debug(f"Unknown event type: {event_type}")

        except json.JSONDecodeError as e:
            logger.error(f"Error parsing WebSocket message: {e}")
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")

    async def _reconnect(self):
        """Reconecta el WebSocket"""
        logger.info("Attempting to reconnect...")

        # Close existing connection
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception:
                pass

        # Get new listenKey
        try:
            self.listen_key = self._get_listen_key()
        except Exception as e:
            logger.error(f"Failed to get new listenKey: {e}")
            return

        # Connect with new listenKey
        ws_url = f"{self.ws_url_base}{self.listen_key}"

        try:
            self.websocket = await websockets.connect(ws_url)
            logger.info("Reconnected to User Data Stream")

            if self.on_reconnect:
                self.on_reconnect()

        except Exception as e:
            logger.error(f"Reconnection failed: {e}")

    async def start(self):
        """Inicia el User Data Stream"""
        logger.info("Starting User Data Stream...")

        # Get listenKey
        try:
            self.listen_key = self._get_listen_key()
            logger.info(f"ListenKey obtained: {self.listen_key[:20]}...")
        except Exception as e:
            logger.error(f"Failed to get listenKey: {e}")
            raise

        # Build WebSocket URL
        ws_url = f"{self.ws_url_base}{self.listen_key}"

        self.is_running = True
        reconnect_attempts = 0

        # Start keepalive task
        self.keepalive_task = asyncio.create_task(self._keepalive_loop())

        while self.is_running and reconnect_attempts < self.max_reconnect_attempts:
            try:
                async with websockets.connect(ws_url) as websocket:
                    self.websocket = websocket
                    reconnect_attempts = 0  # Reset on successful connection
                    logger.info("Connected to User Data Stream")

                    async for message in websocket:
                        if not self.is_running:
                            break
                        await self._handle_message(message)

            except websockets.ConnectionClosed as e:
                logger.warning(f"WebSocket connection closed: {e}")
                if self.on_disconnect:
                    self.on_disconnect()

            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                if self.on_error:
                    self.on_error(e)

            if self.is_running:
                reconnect_attempts += 1
                delay = self.reconnect_delay * reconnect_attempts
                logger.info(f"Reconnecting in {delay} seconds... (attempt {reconnect_attempts})")
                await asyncio.sleep(delay)

                # Get new listenKey for reconnection
                try:
                    self.listen_key = self._get_listen_key()
                    ws_url = f"{self.ws_url_base}{self.listen_key}"
                except Exception as e:
                    logger.error(f"Failed to get new listenKey: {e}")

        logger.info("User Data Stream stopped")

    async def stop(self):
        """Detiene el User Data Stream"""
        logger.info("Stopping User Data Stream...")
        self.is_running = False

        # Cancel keepalive task
        if self.keepalive_task:
            self.keepalive_task.cancel()
            try:
                await self.keepalive_task
            except asyncio.CancelledError:
                pass

        # Close WebSocket
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception:
                pass

        # Close listenKey
        self._close_listen_key()

        logger.info("User Data Stream stopped")


class UserDataStreamManager:
    """
    Manager para integrar User Data Stream con el sistema de trading

    Proporciona callbacks preconfigurados para actualizar el portfolio
    y notificar eventos importantes.
    """

    def __init__(
        self,
        binance_client: 'BinanceFuturesClient',
        portfolio: 'LivePortfolio' = None,
        telegram_notifier = None,
        testnet: bool = False
    ):
        """
        Args:
            binance_client: Cliente de Binance Futures
            portfolio: LivePortfolio para actualizar
            telegram_notifier: Notificador de Telegram (opcional)
            testnet: True para testnet
        """
        self.stream = UserDataStream(binance_client, testnet)
        self.portfolio = portfolio
        self.notifier = telegram_notifier

        # Configure handlers
        self.stream.on_account_update = self._handle_account_update
        self.stream.on_order_update = self._handle_order_update
        self.stream.on_margin_call = self._handle_margin_call
        self.stream.on_error = self._handle_error
        self.stream.on_disconnect = self._handle_disconnect
        self.stream.on_reconnect = self._handle_reconnect

    def _handle_account_update(self, update: AccountUpdate):
        """Handler para actualizaciones de cuenta"""
        logger.info(f"Account update: {update.event_reason}")

        # Log balance changes
        for balance in update.balances:
            if balance['balance_change'] != 0:
                logger.info(
                    f"Balance change: {balance['asset']} "
                    f"{'+' if balance['balance_change'] > 0 else ''}{balance['balance_change']:.4f}"
                )

        # Log position changes
        for position in update.positions:
            if position['position_amount'] != 0:
                logger.info(
                    f"Position: {position['symbol']} {position['position_side']} "
                    f"qty={position['position_amount']} unrealized_pnl={position['unrealized_profit']:.2f}"
                )

        # Sync portfolio if available
        if self.portfolio:
            self.portfolio._sync_with_binance()

    def _handle_order_update(self, update: OrderUpdate):
        """Handler para actualizaciones de ordenes"""
        logger.info(
            f"Order update: {update.symbol} {update.side} {update.order_type} "
            f"status={update.order_status} filled={update.cumulative_filled_qty}"
        )

        # Notify on filled orders
        if update.order_status == 'FILLED':
            pnl_str = f" P&L: ${update.realized_profit:.2f}" if update.realized_profit != 0 else ""
            logger.info(
                f"Order FILLED: {update.symbol} {update.side} @ ${update.avg_price:.2f} "
                f"qty={update.cumulative_filled_qty}{pnl_str}"
            )

            # Send Telegram notification
            if self.notifier and update.realized_profit != 0:
                asyncio.create_task(self._send_fill_notification(update))

    async def _send_fill_notification(self, update: OrderUpdate):
        """Envia notificacion de orden llenada"""
        try:
            emoji = "+" if update.realized_profit > 0 else ""
            message = (
                f"{'LIVE' if not self.stream.testnet else 'TESTNET'} Order Filled\n\n"
                f"Symbol: {update.symbol}\n"
                f"Side: {update.side}\n"
                f"Price: ${update.avg_price:.2f}\n"
                f"Quantity: {update.cumulative_filled_qty}\n"
                f"Realized P&L: {emoji}${update.realized_profit:.2f}"
            )

            if hasattr(self.notifier, 'send_message'):
                await self.notifier.send_message(message)
        except Exception as e:
            logger.error(f"Error sending fill notification: {e}")

    def _handle_margin_call(self, event: MarginCallEvent):
        """Handler para margin calls"""
        logger.warning(f"MARGIN CALL! Wallet balance: ${event.cross_wallet_balance:.2f}")

        for pos in event.positions:
            logger.warning(
                f"At risk: {pos['symbol']} {pos['position_side']} "
                f"qty={pos['position_amount']} "
                f"maintenance_margin={pos['maintenance_margin_required']:.2f}"
            )

        # Send urgent Telegram notification
        if self.notifier:
            asyncio.create_task(self._send_margin_call_notification(event))

    async def _send_margin_call_notification(self, event: MarginCallEvent):
        """Envia notificacion urgente de margin call"""
        try:
            positions_str = "\n".join([
                f"  - {p['symbol']}: qty={p['position_amount']}"
                for p in event.positions
            ])

            message = (
                f"MARGIN CALL ALERT!\n\n"
                f"Wallet Balance: ${event.cross_wallet_balance:.2f}\n"
                f"Positions at risk:\n{positions_str}\n\n"
                f"Please add margin or reduce positions immediately!"
            )

            if hasattr(self.notifier, 'send_error_message'):
                await self.notifier.send_error_message(message)
            elif hasattr(self.notifier, 'send_message'):
                await self.notifier.send_message(message)
        except Exception as e:
            logger.error(f"Error sending margin call notification: {e}")

    def _handle_error(self, error: Exception):
        """Handler para errores"""
        logger.error(f"User Data Stream error: {error}")

    def _handle_disconnect(self):
        """Handler para desconexion"""
        logger.warning("User Data Stream disconnected")

    def _handle_reconnect(self):
        """Handler para reconexion"""
        logger.info("User Data Stream reconnected")

    async def start(self):
        """Inicia el stream"""
        await self.stream.start()

    async def stop(self):
        """Detiene el stream"""
        await self.stream.stop()
