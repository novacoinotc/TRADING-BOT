"""
Portfolio Manager - Gestiona el balance y posiciones del paper trading
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class Portfolio:
    """
    Gestiona el portfolio virtual del paper trading
    Balance inicial: $50,000 USDT
    """

    def __init__(self, initial_balance: float = 50000.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance  # USDT disponible
        self.equity = initial_balance  # Balance + valor posiciones
        self.positions = {}  # Posiciones abiertas: {pair: Position}
        self.closed_trades = []  # Historial de trades cerrados

        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.total_loss = 0.0
        self.max_drawdown = 0.0
        self.peak_equity = initial_balance

        # Data storage
        self.data_dir = Path('data/trades')
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.portfolio_file = self.data_dir / 'portfolio.json'

        # Load existing portfolio if exists
        self._load_portfolio()

    def get_available_balance(self) -> float:
        """Retorna balance USDT disponible para trading"""
        return self.balance

    def get_equity(self) -> float:
        """Retorna equity total (balance + valor posiciones)"""
        return self.equity

    def get_position(self, pair: str) -> Optional[Dict]:
        """Retorna posición abierta para un par"""
        return self.positions.get(pair)

    def has_position(self, pair: str) -> bool:
        """Verifica si hay posición abierta para un par"""
        return pair in self.positions

    def open_position(self, pair: str, side: str, entry_price: float,
                     quantity: float, stop_loss: float, take_profit: Dict) -> Dict:
        """
        Abre nueva posición

        Args:
            pair: Par de trading (ej. BTC/USDT)
            side: 'BUY' o 'SELL'
            entry_price: Precio de entrada
            quantity: Cantidad en cripto
            stop_loss: Precio de stop loss
            take_profit: Dict con tp1, tp2, tp3

        Returns:
            Posición creada
        """
        position_value = entry_price * quantity

        # Reducir balance disponible
        self.balance -= position_value

        position = {
            'pair': pair,
            'side': side,
            'entry_price': entry_price,
            'entry_time': datetime.now().isoformat(),
            'quantity': quantity,
            'position_value': position_value,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'current_price': entry_price,
            'unrealized_pnl': 0.0,
            'status': 'OPEN'
        }

        self.positions[pair] = position
        self.total_trades += 1

        logger.info(f"📈 Posición abierta: {pair} {side} @ ${entry_price:.2f} | Qty: {quantity:.6f}")

        self._save_portfolio()
        return position

    def update_position(self, pair: str, current_price: float) -> Optional[Dict]:
        """
        Actualiza posición con precio actual

        Args:
            pair: Par de trading
            current_price: Precio actual del mercado

        Returns:
            Posición actualizada o None si no existe
        """
        if pair not in self.positions:
            return None

        position = self.positions[pair]
        position['current_price'] = current_price

        # Calcular P&L no realizado
        if position['side'] == 'BUY':
            pnl = (current_price - position['entry_price']) * position['quantity']
        else:  # SELL
            pnl = (position['entry_price'] - current_price) * position['quantity']

        position['unrealized_pnl'] = pnl

        # Actualizar equity
        self._update_equity()

        return position

    def close_position(self, pair: str, exit_price: float, reason: str = 'MANUAL') -> Optional[Dict]:
        """
        Cierra posición y calcula P&L

        Args:
            pair: Par de trading
            exit_price: Precio de salida
            reason: Razón del cierre (TP, SL, SIGNAL, MANUAL)

        Returns:
            Trade cerrado o None si no existe posición
        """
        if pair not in self.positions:
            logger.warning(f"No se puede cerrar {pair} - No hay posición abierta")
            return None

        position = self.positions[pair]

        # Calcular P&L realizado
        if position['side'] == 'BUY':
            pnl = (exit_price - position['entry_price']) * position['quantity']
            pnl_pct = ((exit_price / position['entry_price']) - 1) * 100
        else:  # SELL
            pnl = (position['entry_price'] - exit_price) * position['quantity']
            pnl_pct = ((position['entry_price'] / exit_price) - 1) * 100

        # Devolver capital + ganancia/pérdida
        exit_value = exit_price * position['quantity']
        self.balance += exit_value

        # Actualizar estadísticas
        if pnl > 0:
            self.winning_trades += 1
            self.total_profit += pnl
        else:
            self.losing_trades += 1
            self.total_loss += abs(pnl)

        # Crear registro del trade cerrado
        closed_trade = {
            **position,
            'exit_price': exit_price,
            'exit_time': datetime.now().isoformat(),
            'exit_value': exit_value,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'reason': reason,
            'status': 'CLOSED'
        }

        self.closed_trades.append(closed_trade)

        # Remover de posiciones abiertas
        del self.positions[pair]

        # Actualizar equity y drawdown
        self._update_equity()
        self._update_drawdown()

        emoji = "✅" if pnl > 0 else "❌"
        logger.info(f"{emoji} Posición cerrada: {pair} | P&L: ${pnl:.2f} ({pnl_pct:+.2f}%) | Razón: {reason}")

        self._save_portfolio()
        return closed_trade

    def _update_equity(self):
        """Actualiza equity total (balance + valor posiciones abiertas)"""
        positions_value = sum(
            pos['current_price'] * pos['quantity']
            for pos in self.positions.values()
        )
        self.equity = self.balance + positions_value

        # Actualizar peak equity
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

    def _update_drawdown(self):
        """Calcula y actualiza drawdown máximo"""
        if self.peak_equity > 0:
            current_drawdown = ((self.peak_equity - self.equity) / self.peak_equity) * 100
            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown

    def get_statistics(self) -> Dict:
        """
        Retorna estadísticas completas del portfolio

        Returns:
            Dict con todas las estadísticas
        """
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0

        avg_win = self.total_profit / self.winning_trades if self.winning_trades > 0 else 0
        avg_loss = self.total_loss / self.losing_trades if self.losing_trades > 0 else 0

        profit_factor = self.total_profit / self.total_loss if self.total_loss > 0 else float('inf')

        net_pnl = self.total_profit - self.total_loss
        roi = ((self.equity - self.initial_balance) / self.initial_balance) * 100

        return {
            'initial_balance': self.initial_balance,
            'current_balance': self.balance,
            'equity': self.equity,
            'net_pnl': net_pnl,
            'roi': roi,
            'total_trades': self.total_trades,
            'open_positions': len(self.positions),
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'total_profit': self.total_profit,
            'total_loss': self.total_loss,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self._calculate_sharpe_ratio()
        }

    def _calculate_sharpe_ratio(self) -> float:
        """Calcula Sharpe Ratio (simplificado)"""
        if not self.closed_trades:
            return 0.0

        returns = [trade['pnl_pct'] for trade in self.closed_trades]

        if len(returns) < 2:
            return 0.0

        import numpy as np
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return 0.0

        # Sharpe simplificado (sin risk-free rate)
        sharpe = mean_return / std_return
        return round(sharpe, 2)

    def _save_portfolio(self):
        """Guarda estado del portfolio en disco"""
        data = {
            'balance': self.balance,
            'equity': self.equity,
            'positions': self.positions,
            'closed_trades': self.closed_trades[-100:],  # Últimos 100 trades
            'statistics': self.get_statistics(),
            'last_updated': datetime.now().isoformat()
        }

        with open(self.portfolio_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_portfolio(self):
        """Carga estado del portfolio desde disco"""
        if not self.portfolio_file.exists():
            logger.info(f"💰 Nuevo portfolio creado: ${self.initial_balance:,.2f} USDT")
            return

        try:
            with open(self.portfolio_file, 'r') as f:
                data = json.load(f)

            self.balance = data.get('balance', self.initial_balance)
            self.equity = data.get('equity', self.initial_balance)
            self.positions = data.get('positions', {})
            self.closed_trades = data.get('closed_trades', [])

            # Recalcular estadísticas desde closed_trades
            for trade in self.closed_trades:
                if trade['pnl'] > 0:
                    self.winning_trades += 1
                    self.total_profit += trade['pnl']
                else:
                    self.losing_trades += 1
                    self.total_loss += abs(trade['pnl'])

            self.total_trades = len(self.closed_trades) + len(self.positions)

            logger.info(f"💰 Portfolio cargado: ${self.equity:,.2f} USDT | {len(self.positions)} posiciones abiertas")

        except Exception as e:
            logger.error(f"Error cargando portfolio: {e}")
            logger.info(f"💰 Nuevo portfolio creado: ${self.initial_balance:,.2f} USDT")
