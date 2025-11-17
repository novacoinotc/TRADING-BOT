"""
Test para verificar que el bug de cierre automático está corregido

BUG ORIGINAL:
- El bot cerraba trades con entry_price = exit_price
- P&L = $0.00
- Principalmente en AXS/USDT y SUI/USDT

CAUSA:
- TP/SL configurados incorrectamente (igual al entry_price)
- Sin validación de movimiento de precio antes de cerrar
- Sin validación de precio actual antes de comparar con TP/SL

SOLUCIÓN:
- Validar current_price antes de verificar condiciones de salida
- Validar que TP/SL son diferentes del entry_price
- Validar movimiento mínimo antes de cerrar (0.01%)
- Logs detallados para debugging
"""

import math
from src.trading.portfolio import Portfolio
from src.trading.position_manager import PositionManager, get_validated_current_price


class TestAutoCloseBugFix:
    """Tests para verificar que el bug de cierre automático está corregido"""

    def test_get_validated_current_price_valid(self):
        """Verificar que precios válidos pasan la validación"""
        price = 1.2700
        validated = get_validated_current_price(price, "AXS/USDT")
        assert validated == price

    def test_get_validated_current_price_none(self):
        """Verificar que precio None es rechazado"""
        validated = get_validated_current_price(None, "AXS/USDT")
        assert validated is None

    def test_get_validated_current_price_zero(self):
        """Verificar que precio 0 es rechazado"""
        validated = get_validated_current_price(0, "AXS/USDT")
        assert validated is None

    def test_get_validated_current_price_negative(self):
        """Verificar que precio negativo es rechazado"""
        validated = get_validated_current_price(-1.5, "AXS/USDT")
        assert validated is None

    def test_get_validated_current_price_nan(self):
        """Verificar que precio NaN es rechazado"""
        validated = get_validated_current_price(math.nan, "AXS/USDT")
        assert validated is None

    def test_get_validated_current_price_inf(self):
        """Verificar que precio Inf es rechazado"""
        validated = get_validated_current_price(math.inf, "AXS/USDT")
        assert validated is None

    def test_no_close_on_same_price(self):
        """
        TEST PRINCIPAL: Verificar que NO cierra si precio no cambió

        Simula el bug reportado:
        - Posición SELL en AXS/USDT a $1.2700
        - current_price = $1.2700 (sin cambio)
        - NO debe cerrar
        """
        portfolio = Portfolio(initial_balance=50000)
        position_manager = PositionManager(portfolio, max_position_size_pct=5.0)

        # Abrir posición SELL en AXS/USDT a $1.2700
        entry_price = 1.2700
        position = portfolio.open_position(
            pair='AXS/USDT',
            side='SELL',
            entry_price=entry_price,
            quantity=100,
            stop_loss=1.2900,  # SL correcto para SELL
            take_profit={'tp1': 1.2500, 'tp2': 1.2300},  # TP correcto para SELL
            trade_type='FUTURES',
            leverage=4
        )

        assert position is not None
        assert portfolio.has_position('AXS/USDT')

        # Intentar cerrar con MISMO precio (simular bug)
        current_price = 1.2700

        # Verificar que NO cierra
        closed_trade = position_manager.update_positions('AXS/USDT', current_price)

        # La posición NO debe haberse cerrado
        assert closed_trade is None
        assert portfolio.has_position('AXS/USDT')

    def test_no_close_with_tp_equal_to_entry(self):
        """
        Verificar que NO cierra si TP está mal configurado (igual al entry_price)

        Esto simula la causa del bug:
        - Si TP = entry_price por error en cálculo
        - NO debe cerrar
        """
        portfolio = Portfolio(initial_balance=50000)
        position_manager = PositionManager(portfolio, max_position_size_pct=5.0)

        # Abrir posición SELL
        entry_price = 1.2700
        position = portfolio.open_position(
            pair='SUI/USDT',
            side='SELL',
            entry_price=entry_price,
            quantity=100,
            stop_loss=1.2900,
            take_profit={'tp1': 1.2700},  # ❌ TP MAL CONFIGURADO (igual a entry)
            trade_type='FUTURES',
            leverage=4
        )

        assert position is not None

        # Precio actual = entry (sin movimiento)
        current_price = 1.2700

        # NO debe cerrar por TP mal configurado
        closed_trade = position_manager.update_positions('SUI/USDT', current_price)

        assert closed_trade is None
        assert portfolio.has_position('SUI/USDT')

    def test_no_close_with_sl_equal_to_entry(self):
        """
        Verificar que NO cierra si SL está mal configurado (igual al entry_price)
        """
        portfolio = Portfolio(initial_balance=50000)
        position_manager = PositionManager(portfolio, max_position_size_pct=5.0)

        # Abrir posición BUY
        entry_price = 2.5000
        position = portfolio.open_position(
            pair='CRV/USDT',
            side='BUY',
            entry_price=entry_price,
            quantity=100,
            stop_loss=2.5000,  # ❌ SL MAL CONFIGURADO (igual a entry)
            take_profit={'tp1': 2.6000},
            trade_type='FUTURES',
            leverage=3
        )

        assert position is not None

        # Precio actual = entry
        current_price = 2.5000

        # NO debe cerrar por SL mal configurado
        closed_trade = position_manager.update_positions('CRV/USDT', current_price)

        assert closed_trade is None
        assert portfolio.has_position('CRV/USDT')

    def test_close_with_valid_tp(self):
        """
        Verificar que SÍ cierra correctamente cuando TP es válido y se alcanza
        """
        portfolio = Portfolio(initial_balance=50000)
        position_manager = PositionManager(portfolio, max_position_size_pct=5.0)

        # Abrir posición SELL
        entry_price = 1.2700
        position = portfolio.open_position(
            pair='AXS/USDT',
            side='SELL',
            entry_price=entry_price,
            quantity=100,
            stop_loss=1.2900,
            take_profit={'tp1': 1.2500},  # TP correcto: menor que entry para SELL
            trade_type='FUTURES',
            leverage=4
        )

        assert position is not None

        # Precio baja a TP1 (movimiento real > 1%)
        current_price = 1.2500

        # SÍ debe cerrar porque:
        # 1. current_price es válido
        # 2. TP es diferente del entry
        # 3. Hubo movimiento real: (1.2700 - 1.2500) / 1.2700 = 1.57%
        # 4. Para SELL: current_price <= tp1 → 1.2500 <= 1.2500 → True
        closed_trade = position_manager.update_positions('AXS/USDT', current_price)

        assert closed_trade is not None
        assert closed_trade['reason'] == 'TP1'
        assert closed_trade['exit_price'] == 1.2500
        assert closed_trade['pnl'] > 0  # Ganancia en SELL
        assert not portfolio.has_position('AXS/USDT')

    def test_close_with_valid_sl(self):
        """
        Verificar que SÍ cierra correctamente cuando SL es válido y se alcanza
        """
        portfolio = Portfolio(initial_balance=50000)
        position_manager = PositionManager(portfolio, max_position_size_pct=5.0)

        # Abrir posición BUY
        entry_price = 2.5000
        position = portfolio.open_position(
            pair='CRV/USDT',
            side='BUY',
            entry_price=entry_price,
            quantity=100,
            stop_loss=2.4500,  # SL correcto: menor que entry para BUY
            take_profit={'tp1': 2.6000},
            trade_type='FUTURES',
            leverage=3
        )

        assert position is not None

        # Precio baja a SL (movimiento real: 2%)
        current_price = 2.4500

        # SÍ debe cerrar porque:
        # 1. current_price es válido
        # 2. SL es diferente del entry
        # 3. Hubo movimiento real: (2.5000 - 2.4500) / 2.5000 = 2%
        # 4. Para BUY: current_price <= stop_loss → 2.4500 <= 2.4500 → True
        closed_trade = position_manager.update_positions('CRV/USDT', current_price)

        assert closed_trade is not None
        assert closed_trade['reason'] == 'STOP_LOSS'
        assert closed_trade['exit_price'] == 2.4500
        assert closed_trade['pnl'] < 0  # Pérdida en BUY con SL
        assert not portfolio.has_position('CRV/USDT')

    def test_portfolio_rejects_zero_movement_close(self):
        """
        Verificar que Portfolio.close_position() rechaza cierres sin movimiento
        incluso si se llama directamente (última línea de defensa)
        """
        portfolio = Portfolio(initial_balance=50000)

        # Abrir posición
        entry_price = 1.2700
        position = portfolio.open_position(
            pair='AXS/USDT',
            side='SELL',
            entry_price=entry_price,
            quantity=100,
            stop_loss=1.2900,
            take_profit={'tp1': 1.2500},
            trade_type='FUTURES',
            leverage=4
        )

        assert position is not None

        # Intentar cerrar directamente con mismo precio
        exit_price = 1.2700

        closed_trade = portfolio.close_position('AXS/USDT', exit_price, reason='TEST')

        # NO debe cerrar (última línea de defensa)
        assert closed_trade is None
        assert portfolio.has_position('AXS/USDT')


if __name__ == '__main__':
    print("Run tests with: python -m pytest tests/test_auto_close_bug_fix.py -v")
