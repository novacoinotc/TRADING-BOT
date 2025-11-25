"""
Live Portfolio Manager - Gestiona el balance y posiciones REALES de Binance Futures
Reemplaza el Portfolio de paper trading para modo produccion
"""

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.trading.binance_futures_client import BinanceFuturesClient

logger = logging.getLogger(__name__)


class LivePortfolio:
    """
    Gestiona el portfolio REAL de Binance Futures

    Diferencias con Portfolio (paper trading):
    - Balance y posiciones se obtienen de la API de Binance
    - No hay simulacion, todas las operaciones son reales
    - Mantiene historial local de trades para estadisticas
    """

    def __init__(
        self,
        binance_client: 'BinanceFuturesClient',
        initial_reference_balance: float = 50000.0
    ):
        """
        Args:
            binance_client: Cliente de Binance Futures
            initial_reference_balance: Balance de referencia para calcular ROI
                                       (deberia ser el balance al iniciar el bot)
        """
        self.client = binance_client
        self.initial_balance = initial_reference_balance

        # Cache de posiciones locales (para tracking interno)
        self.local_positions: Dict[str, Dict] = {}  # {symbol: position_data}
        self.closed_trades: List[Dict] = []  # Historial de trades cerrados

        # Performance tracking (calculado desde historial local)
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.total_loss = 0.0
        self.max_drawdown = 0.0
        self.peak_equity = initial_reference_balance

        # Data storage
        self.data_dir = Path('data/trades')
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.portfolio_file = self.data_dir / 'live_portfolio.json'

        # Load existing data if exists
        self._load_portfolio()

        # Sync with Binance on startup
        self._sync_with_binance()

        logger.info(f"LivePortfolio initialized with reference balance: ${initial_reference_balance:,.2f}")

    def _sync_with_binance(self):
        """Sincroniza posiciones locales con las reales de Binance"""
        try:
            # Get real positions from Binance
            real_positions = self.client.get_open_positions()

            # Update local tracking
            synced_symbols = set()
            for pos in real_positions:
                symbol = pos.symbol
                synced_symbols.add(symbol)

                # Convert to local format for compatibility
                pair = self._symbol_to_pair(symbol)
                side = 'BUY' if pos.position_amt > 0 else 'SELL'

                self.local_positions[pair] = {
                    'pair': pair,
                    'symbol': symbol,
                    'side': side,
                    'trade_type': 'FUTURES',
                    'leverage': pos.leverage,
                    'entry_price': pos.entry_price,
                    'entry_time': datetime.now().isoformat(),  # Approximate
                    'quantity': abs(pos.position_amt),
                    'position_value': abs(pos.notional),
                    'collateral': abs(pos.isolated_margin) if pos.margin_type == 'isolated' else abs(pos.notional) / pos.leverage,
                    'liquidation_price': pos.liquidation_price,
                    'liquidated': False,
                    'stop_loss': None,  # Not tracked by Binance API directly
                    'take_profit': {},
                    'current_price': pos.mark_price,
                    'unrealized_pnl': pos.unrealized_profit,
                    'status': 'OPEN',
                    'margin_type': pos.margin_type
                }

            # Remove positions that no longer exist in Binance
            pairs_to_remove = [p for p in self.local_positions if self._pair_to_symbol(p) not in synced_symbols]
            for pair in pairs_to_remove:
                logger.warning(f"Position {pair} no longer exists in Binance, removing from local tracking")
                del self.local_positions[pair]

            logger.info(f"Synced {len(self.local_positions)} positions with Binance")

        except Exception as e:
            logger.error(f"Error syncing with Binance: {e}")

    def _symbol_to_pair(self, symbol: str) -> str:
        """Convierte BTCUSDT -> BTC/USDT (usando mapeo de símbolos del cliente)"""
        # Usar el método del cliente que maneja símbolos especiales como 1000SHIB
        return self.client.convert_symbol_to_pair(symbol)

    def _pair_to_symbol(self, pair: str) -> str:
        """Convierte BTC/USDT -> BTCUSDT (usando mapeo de símbolos del cliente)"""
        # Usar el método del cliente que maneja símbolos especiales como 1000SHIB
        return self.client.convert_pair_format(pair)

    def get_available_balance(self) -> float:
        """Retorna balance USDT disponible para trading (REAL de Binance)"""
        try:
            usdt_balance = self.client.get_usdt_balance()
            return usdt_balance.available_balance
        except Exception as e:
            logger.error(f"Error getting USDT balance: {e}")
            return 0.0

    def get_total_balance(self) -> float:
        """Retorna balance total USDT (incluyendo en posiciones)"""
        try:
            usdt_balance = self.client.get_usdt_balance()
            return usdt_balance.balance
        except Exception as e:
            logger.error(f"Error getting total balance: {e}")
            return 0.0

    def get_equity(self) -> float:
        """Retorna equity total (balance + unrealized PnL)"""
        try:
            usdt_balance = self.client.get_usdt_balance()
            return usdt_balance.balance + usdt_balance.cross_unrealized_pnl
        except Exception as e:
            logger.error(f"Error getting equity: {e}")
            return 0.0

    def get_position(self, pair: str) -> Optional[Dict]:
        """Retorna posicion abierta para un par"""
        return self.local_positions.get(pair)

    def has_position(self, pair: str) -> bool:
        """Verifica si hay posicion abierta para un par"""
        # Check both local cache and Binance
        if pair in self.local_positions:
            return True

        # Double check with Binance
        try:
            symbol = self._pair_to_symbol(pair)
            positions = self.client.get_positions(symbol)
            for pos in positions:
                if pos.position_amt != 0:
                    return True
        except Exception as e:
            logger.error(f"Error checking position for {pair}: {e}")

        return False

    def register_opened_position(
        self,
        pair: str,
        side: str,
        entry_price: float,
        quantity: float,
        stop_loss: float,
        take_profit: Dict,
        leverage: int = 1,
        order_result: Optional[Dict] = None
    ) -> Dict:
        """
        Registra una posicion abierta en el tracking local

        NOTA: Esta funcion NO abre la posicion en Binance.
        La posicion ya debe estar abierta via BinanceFuturesClient.
        Esta funcion solo registra para tracking interno.

        Args:
            pair: Par de trading (ej. BTC/USDT)
            side: 'BUY' (LONG) o 'SELL' (SHORT)
            entry_price: Precio de entrada
            quantity: Cantidad
            stop_loss: Precio de stop loss
            take_profit: Dict con tp1, tp2, tp3
            leverage: Nivel de leverage
            order_result: Resultado de la orden de Binance (opcional)
        """
        # Validate inputs
        if entry_price is None or entry_price <= 0 or math.isnan(entry_price) or math.isinf(entry_price):
            logger.error(f"Invalid entry price for {pair}: {entry_price}")
            return None

        if quantity is None or quantity <= 0 or math.isnan(quantity) or math.isinf(quantity):
            logger.error(f"Invalid quantity for {pair}: {quantity}")
            return None

        position_value = entry_price * quantity
        collateral = position_value / leverage

        # Calculate liquidation price
        if side == 'BUY':
            liquidation_price = entry_price * (1 - (1 / leverage) * 0.9)
        else:
            liquidation_price = entry_price * (1 + (1 / leverage) * 0.9)

        position = {
            'pair': pair,
            'symbol': self._pair_to_symbol(pair),
            'side': side,
            'trade_type': 'FUTURES',
            'leverage': leverage,
            'entry_price': entry_price,
            'entry_time': datetime.now().isoformat(),
            'quantity': quantity,
            'position_value': position_value,
            'collateral': collateral,
            'liquidation_price': liquidation_price,
            'liquidated': False,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'current_price': entry_price,
            'unrealized_pnl': 0.0,
            'status': 'OPEN',
            'order_id': order_result.get('orderId') if order_result else None
        }

        self.local_positions[pair] = position
        self.total_trades += 1

        logger.info(f"Position registered: {pair} {side} @ ${entry_price:.2f} x {leverage}x")

        self._save_portfolio()
        return position

    def update_position(self, pair: str, current_price: float) -> Optional[Dict]:
        """
        Actualiza posicion con precio actual

        Args:
            pair: Par de trading
            current_price: Precio actual del mercado

        Returns:
            Posicion actualizada o None si no existe
        """
        if pair not in self.local_positions:
            return None

        position = self.local_positions[pair]
        position['current_price'] = current_price

        # Calculate unrealized P&L
        # Note: Leverage is already factored into quantity (more coins bought with collateral)
        # so we do NOT multiply by leverage here - that would be double-counting
        if position['side'] == 'BUY':
            pnl = (current_price - position['entry_price']) * position['quantity']
        else:
            pnl = (position['entry_price'] - current_price) * position['quantity']

        position['unrealized_pnl'] = pnl

        # Check for liquidation
        if not position.get('liquidated'):
            liquidation_price = position.get('liquidation_price', 0)
            if liquidation_price:
                is_liquidated = False
                if position['side'] == 'BUY' and current_price <= liquidation_price:
                    is_liquidated = True
                elif position['side'] == 'SELL' and current_price >= liquidation_price:
                    is_liquidated = True

                if is_liquidated:
                    logger.warning(f"LIQUIDATION detected for {pair}!")
                    position['liquidated'] = True

        return position

    def register_closed_position(
        self,
        pair: str,
        exit_price: float,
        reason: str = 'MANUAL',
        order_result: Optional[Dict] = None
    ) -> Optional[Dict]:
        """
        Registra el cierre de una posicion

        NOTA: Esta funcion NO cierra la posicion en Binance.
        La posicion ya debe estar cerrada via BinanceFuturesClient.
        Esta funcion solo registra para tracking interno.

        Args:
            pair: Par de trading
            exit_price: Precio de salida
            reason: Razon del cierre (TP, SL, SIGNAL, MANUAL)
            order_result: Resultado de la orden de Binance (opcional)

        Returns:
            Trade cerrado o None si no existe posicion
        """
        if pair not in self.local_positions:
            logger.warning(f"Cannot close {pair} - No position tracked")
            return None

        position = self.local_positions[pair]
        leverage = position.get('leverage', 1)
        liquidated = position.get('liquidated', False)

        # Validate exit price
        if exit_price is None or exit_price <= 0 or math.isnan(exit_price) or math.isinf(exit_price):
            logger.error(f"Invalid exit price for {pair}: {exit_price}")
            return None

        # Calculate realized P&L
        # Note: Leverage is already factored into quantity (more coins bought with collateral)
        # so we do NOT multiply by leverage here - that would be double-counting
        if position['side'] == 'BUY':
            pnl = (exit_price - position['entry_price']) * position['quantity']
            pnl_pct = ((exit_price / position['entry_price']) - 1) * 100
        else:
            pnl = (position['entry_price'] - exit_price) * position['quantity']
            pnl_pct = ((position['entry_price'] / exit_price) - 1) * 100

        # For ROI percentage, we calculate based on collateral (not full position value)
        # ROI = PnL / collateral * 100
        collateral = position.get('collateral', position['position_value'] / leverage)
        if collateral > 0:
            pnl_pct = (pnl / collateral) * 100

        # Handle liquidation
        if reason == 'LIQUIDATION' or liquidated:
            collateral = position.get('collateral', position['position_value'] / leverage)
            pnl = -collateral
            pnl_pct = -100.0
            exit_value = 0
        else:
            collateral = position.get('collateral', position['position_value'] / leverage)
            exit_value = collateral + pnl

        # Update statistics
        if not (math.isnan(pnl) or math.isinf(pnl)):
            if pnl > 0:
                self.winning_trades += 1
                self.total_profit += pnl
            else:
                self.losing_trades += 1
                self.total_loss += abs(pnl)

        # Calculate duration
        duration_minutes = 0
        if 'entry_time' in position:
            entry_time = position['entry_time']
            if isinstance(entry_time, str):
                entry_time = datetime.fromisoformat(entry_time)
            duration_minutes = (datetime.now() - entry_time).total_seconds() / 60

        # Create closed trade record
        closed_trade = {
            **position,
            'exit_price': exit_price,
            'exit_time': datetime.now().isoformat(),
            'exit_value': exit_value,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'reason': reason,
            'status': 'CLOSED',
            'duration': duration_minutes,
            'exit_order_id': order_result.get('orderId') if order_result else None
        }

        self.closed_trades.append(closed_trade)

        # Remove from local positions
        del self.local_positions[pair]

        # Update equity tracking
        self._update_equity_tracking()

        emoji = "+" if pnl > 0 else ""
        logger.info(f"Position closed: {pair} | P&L: {emoji}${pnl:.2f} ({pnl_pct:+.2f}%) | Reason: {reason}")

        self._save_portfolio()
        return closed_trade

    def _update_equity_tracking(self):
        """Actualiza tracking de equity y drawdown"""
        try:
            current_equity = self.get_equity()

            # Update peak equity
            if current_equity > self.peak_equity:
                self.peak_equity = current_equity

            # Update max drawdown
            if self.peak_equity > 0:
                current_drawdown = ((self.peak_equity - current_equity) / self.peak_equity) * 100
                if current_drawdown > self.max_drawdown:
                    self.max_drawdown = current_drawdown

        except Exception as e:
            logger.error(f"Error updating equity tracking: {e}")

    def get_statistics(self) -> Dict:
        """
        Retorna estadisticas completas del portfolio

        Returns:
            Dict con todas las estadisticas
        """
        try:
            current_balance = self.get_total_balance()
            equity = self.get_equity()
        except Exception:
            current_balance = 0.0
            equity = 0.0

        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        avg_win = self.total_profit / self.winning_trades if self.winning_trades > 0 else 0
        avg_loss = self.total_loss / self.losing_trades if self.losing_trades > 0 else 0
        profit_factor = self.total_profit / self.total_loss if self.total_loss > 0 else float('inf')
        net_pnl = self.total_profit - self.total_loss
        roi = ((equity - self.initial_balance) / self.initial_balance) * 100 if self.initial_balance > 0 else 0

        return {
            'initial_balance': self.initial_balance,
            'current_balance': current_balance,
            'equity': equity,
            'net_pnl': net_pnl,
            'roi': roi,
            'total_trades': self.total_trades,
            'open_positions': len(self.local_positions),
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'total_profit': self.total_profit,
            'total_loss': self.total_loss,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'trading_mode': 'LIVE'
        }

    def _calculate_sharpe_ratio(self) -> float:
        """Calcula Sharpe Ratio (simplificado)"""
        if not self.closed_trades:
            return 0.0

        returns = [trade['pnl_pct'] for trade in self.closed_trades if 'pnl_pct' in trade]

        if len(returns) < 2:
            return 0.0

        try:
            import numpy as np
            mean_return = np.mean(returns)
            std_return = np.std(returns)

            if std_return == 0:
                return 0.0

            sharpe = mean_return / std_return
            return round(sharpe, 2)
        except Exception:
            return 0.0

    def _save_portfolio(self):
        """Guarda estado del portfolio en disco"""
        data = {
            'initial_balance': self.initial_balance,
            'local_positions': self.local_positions,
            'closed_trades': self.closed_trades[-500:],  # Keep last 500 trades
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'total_profit': self.total_profit,
            'total_loss': self.total_loss,
            'max_drawdown': self.max_drawdown,
            'peak_equity': self.peak_equity,
            'statistics': self.get_statistics(),
            'last_updated': datetime.now().isoformat(),
            'trading_mode': 'LIVE'
        }

        try:
            with open(self.portfolio_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving portfolio: {e}")

    def _load_portfolio(self):
        """Carga estado del portfolio desde disco"""
        if not self.portfolio_file.exists():
            logger.info(f"New LIVE portfolio - Reference balance: ${self.initial_balance:,.2f}")
            return

        try:
            with open(self.portfolio_file, 'r') as f:
                data = json.load(f)

            # Verify it's a live portfolio file
            if data.get('trading_mode') != 'LIVE':
                logger.warning("Found paper trading portfolio file, starting fresh for LIVE trading")
                return

            self.initial_balance = data.get('initial_balance', self.initial_balance)
            self.local_positions = data.get('local_positions', {})
            self.closed_trades = data.get('closed_trades', [])
            self.total_trades = data.get('total_trades', 0)
            self.winning_trades = data.get('winning_trades', 0)
            self.losing_trades = data.get('losing_trades', 0)
            self.total_profit = data.get('total_profit', 0.0)
            self.total_loss = data.get('total_loss', 0.0)
            self.max_drawdown = data.get('max_drawdown', 0.0)
            self.peak_equity = data.get('peak_equity', self.initial_balance)

            logger.info(f"LIVE Portfolio loaded: {self.total_trades} trades, {len(self.local_positions)} open positions")

        except Exception as e:
            logger.error(f"Error loading portfolio: {e}")
            logger.info("Starting with fresh portfolio state")

    @property
    def positions(self) -> Dict:
        """Property for compatibility with PaperTrader interface"""
        return self.local_positions

    def get_full_state_for_export(self) -> Dict:
        """Retorna estado completo del portfolio para exportar"""
        return {
            'initial_balance': self.initial_balance,
            'local_positions': self.local_positions,
            'closed_trades': self.closed_trades,
            'statistics': self.get_statistics(),
            'counters': {
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'total_profit': self.total_profit,
                'total_loss': self.total_loss,
                'max_drawdown': self.max_drawdown,
                'peak_equity': self.peak_equity
            },
            'timestamp': datetime.now().isoformat(),
            'trading_mode': 'LIVE'
        }
