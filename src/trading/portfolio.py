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
                     quantity: float, stop_loss: float, take_profit: Dict,
                     trade_type: str = 'SPOT', leverage: int = 1) -> Dict:
        """
        Abre nueva posición (SPOT o FUTURES)

        Args:
            pair: Par de trading (ej. BTC/USDT)
            side: 'BUY' o 'SELL'
            entry_price: Precio de entrada
            quantity: Cantidad en cripto
            stop_loss: Precio de stop loss
            take_profit: Dict con tp1, tp2, tp3
            trade_type: 'SPOT' o 'FUTURES'
            leverage: 1-20x (solo para FUTURES)

        Returns:
            Posición creada
        """
        position_value = entry_price * quantity

        # Validar leverage para FUTURES
        if trade_type == 'FUTURES':
            if not (1 <= leverage <= 20):
                raise ValueError(
                    f"❌ Leverage {leverage}x inválido para {pair}\n"
                    f"   Permitido: 1-20x\n"
                    f"   Verifica que el RL Agent esté calculando leverage correctamente"
                )

        # Para SPOT: usa el valor completo
        # Para FUTURES: solo usa colateral (valor / leverage)
        if trade_type == 'FUTURES':
            collateral = position_value / leverage
            self.balance -= collateral
        else:
            self.balance -= position_value

        # Calcular liquidation price para futures
        liquidation_price = None
        if trade_type == 'FUTURES':
            # Fórmula simplificada de liquidación
            # Para LONG: liquidation = entry * (1 - 1/leverage * 0.9)
            # Para SHORT: liquidation = entry * (1 + 1/leverage * 0.9)
            if side == 'BUY':
                liquidation_price = entry_price * (1 - (1 / leverage) * 0.9)
            else:  # SELL
                liquidation_price = entry_price * (1 + (1 / leverage) * 0.9)

        position = {
            'pair': pair,
            'side': side,
            'trade_type': trade_type,
            'leverage': leverage,
            'entry_price': entry_price,
            'entry_time': datetime.now().isoformat(),
            'quantity': quantity,
            'position_value': position_value,
            'collateral': position_value / leverage if trade_type == 'FUTURES' else position_value,
            'liquidation_price': liquidation_price,
            'liquidated': False,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'current_price': entry_price,
            'unrealized_pnl': 0.0,
            'status': 'OPEN'
        }

        self.positions[pair] = position
        self.total_trades += 1

        log_msg = f"📈 Posición abierta: {pair} {side} {trade_type} @ ${entry_price:.2f}"
        if trade_type == 'FUTURES':
            log_msg += f" | Leverage: {leverage}x | Liquidación: ${liquidation_price:.2f}"
        logger.info(log_msg)

        self._save_portfolio()
        return position

    def update_position(self, pair: str, current_price: float) -> Optional[Dict]:
        """
        Actualiza posición con precio actual y verifica liquidación para futures

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

        # Para FUTURES: multiplicar PnL por leverage
        if position.get('trade_type') == 'FUTURES':
            leverage = position.get('leverage', 1)
            pnl = pnl * leverage

        position['unrealized_pnl'] = pnl

        # Verificar liquidación para FUTURES
        if position.get('trade_type') == 'FUTURES' and not position.get('liquidated'):
            liquidation_price = position.get('liquidation_price')
            if liquidation_price:
                is_liquidated = False
                if position['side'] == 'BUY' and current_price <= liquidation_price:
                    is_liquidated = True
                elif position['side'] == 'SELL' and current_price >= liquidation_price:
                    is_liquidated = True

                if is_liquidated:
                    logger.warning(f"💥 LIQUIDACIÓN en {pair}! Precio: ${current_price:.2f} | Liquidación: ${liquidation_price:.2f}")
                    position['liquidated'] = True
                    # Cerrar posición inmediatamente por liquidación
                    self.close_position(pair, current_price, reason='LIQUIDATION')
                    return None  # Posición ya no existe

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
        trade_type = position.get('trade_type', 'SPOT')
        leverage = position.get('leverage', 1)
        liquidated = position.get('liquidated', False)

        # Calcular P&L realizado
        if position['side'] == 'BUY':
            pnl = (exit_price - position['entry_price']) * position['quantity']
            pnl_pct = ((exit_price / position['entry_price']) - 1) * 100
        else:  # SELL
            pnl = (position['entry_price'] - exit_price) * position['quantity']
            pnl_pct = ((position['entry_price'] / exit_price) - 1) * 100

        # Para FUTURES: aplicar leverage al PnL
        exit_value = 0
        if trade_type == 'FUTURES':
            pnl = pnl * leverage
            pnl_pct = pnl_pct * leverage

            # Para LIQUIDACIÓN: PnL es -100% del colateral
            if reason == 'LIQUIDATION':
                collateral = position.get('collateral', position['position_value'] / leverage)
                pnl = -collateral
                pnl_pct = -100.0
                exit_value = 0  # Se pierde todo
                # Devolver 0 (se pierde todo el colateral)
                self.balance += 0
            else:
                # Devolver colateral + PnL
                collateral = position.get('collateral', position['position_value'] / leverage)
                exit_value = collateral + pnl
                self.balance += exit_value
        else:
            # SPOT: devolver capital + ganancia/pérdida
            exit_value = exit_price * position['quantity']
            self.balance += exit_value

        # Actualizar estadísticas
        if pnl > 0:
            self.winning_trades += 1
            self.total_profit += pnl
        else:
            self.losing_trades += 1
            self.total_loss += abs(pnl)

        # Calcular duración del trade
        duration_minutes = 0
        if 'entry_time' in position:
            entry_time = position['entry_time']
            # Si entry_time es string, parsear a datetime
            if isinstance(entry_time, str):
                entry_time = datetime.fromisoformat(entry_time)
            # Calcular diferencia (ahora entry_time es datetime)
            duration_minutes = (datetime.now() - entry_time).total_seconds() / 60

        # Crear registro del trade cerrado
        closed_trade = {
            **position,
            'exit_price': exit_price,
            'exit_time': datetime.now().isoformat(),
            'exit_value': exit_value,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'reason': reason,
            'status': 'CLOSED',
            'duration': duration_minutes
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
            'closed_trades': self.closed_trades,  # ✅ TODO EL HISTORIAL (sin límites)
            'total_trades': self.total_trades,  # ✅ GUARDAR total histórico
            'winning_trades': self.winning_trades,  # ✅ GUARDAR contadores
            'losing_trades': self.losing_trades,
            'total_profit': self.total_profit,
            'total_loss': self.total_loss,
            'max_drawdown': self.max_drawdown,
            'peak_equity': self.peak_equity,
            'statistics': self.get_statistics(),
            'last_updated': datetime.now().isoformat()
        }

        logger.debug(f"💾 Guardando portfolio: {len(self.closed_trades)} trades cerrados, {self.total_trades} total histórico")

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

            # Cargar total_trades desde el archivo guardado, NO recalcular
            saved_total = data.get('total_trades', len(self.closed_trades))
            self.total_trades = saved_total

            # Restaurar contadores guardados
            self.winning_trades = data.get('winning_trades', 0)
            self.losing_trades = data.get('losing_trades', 0)
            self.total_profit = data.get('total_profit', 0.0)
            self.total_loss = data.get('total_loss', 0.0)
            self.max_drawdown = data.get('max_drawdown', 0.0)
            self.peak_equity = data.get('peak_equity', self.initial_balance)

            logger.info(f"📊 Total trades histórico: {self.total_trades}")

            # SOLO recalcular si no hay contadores guardados (primera vez)
            if not data.get('total_trades'):
                # Recalcular estadísticas desde closed_trades
                self.winning_trades = 0
                self.losing_trades = 0
                self.total_profit = 0.0
                self.total_loss = 0.0

                for trade in self.closed_trades:
                    if trade['pnl'] > 0:
                        self.winning_trades += 1
                        self.total_profit += trade['pnl']
                    else:
                        self.losing_trades += 1
                        self.total_loss += abs(trade['pnl'])

            logger.info(f"💰 Portfolio cargado: ${self.equity:,.2f} USDT | {len(self.positions)} posiciones abiertas")

        except Exception as e:
            logger.error(f"Error cargando portfolio: {e}")
            logger.info(f"💰 Nuevo portfolio creado: ${self.initial_balance:,.2f} USDT")

    def get_full_state_for_export(self) -> Dict:
        """
        Retorna estado completo del portfolio para exportar con RL Agent
        INCLUYE TODO EL HISTORIAL SIN LÍMITES
        """
        return {
            'balance': self.balance,
            'equity': self.equity,
            'initial_balance': self.initial_balance,
            'positions': self.positions,
            'closed_trades': self.closed_trades,  # ✅ TODO EL HISTORIAL
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
            'timestamp': datetime.now().isoformat()
        }

    def restore_from_state(self, state: Dict) -> bool:
        """
        Restaura el portfolio completo desde un estado exportado
        Usado para importar inteligencia con paper trading incluido
        """
        try:
            self.balance = state.get('balance', self.initial_balance)
            self.equity = state.get('equity', self.initial_balance)
            self.positions = state.get('positions', {})
            self.closed_trades = state.get('closed_trades', [])

            # Restaurar contadores guardados (NO recalcular)
            counters = state.get('counters', {})
            self.total_trades = counters.get('total_trades', len(self.closed_trades))
            self.winning_trades = counters.get('winning_trades', 0)
            self.losing_trades = counters.get('losing_trades', 0)
            self.total_profit = counters.get('total_profit', 0.0)
            self.total_loss = counters.get('total_loss', 0.0)
            self.max_drawdown = counters.get('max_drawdown', 0.0)
            self.peak_equity = counters.get('peak_equity', self.initial_balance)

            logger.info(
                f"✅ Portfolio restaurado desde export:\n"
                f"   Balance: ${self.balance:,.2f}\n"
                f"   Total trades: {self.total_trades}\n"
                f"   Closed trades cargados: {len(self.closed_trades)}\n"
                f"   Win rate: {(self.winning_trades/self.total_trades*100) if self.total_trades > 0 else 0:.1f}%"
            )

            # Guardar estado restaurado
            self._save_portfolio()

            return True
        except Exception as e:
            logger.error(f"❌ Error restaurando portfolio: {e}", exc_info=True)
            return False
