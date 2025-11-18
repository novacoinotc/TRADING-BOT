"""
Test Mode - Sistema de pruebas automáticas para Trading Bot v2.0
Ejecuta trades automáticamente cada 3 minutos para verificar funcionamiento completo
"""

import asyncio
import random
import logging
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class TestMode:
    """
    Modo de prueba que ejecuta trades automáticamente para verificar:
    - Conexión a Binance funciona
    - Se pueden abrir/cerrar posiciones
    - Dashboard se actualiza correctamente
    - Notificaciones Telegram funcionan
    - RL Agent aprende de los trades
    """

    def __init__(self, futures_trader=None, position_monitor=None, notifier=None, autonomy_controller=None):
        """
        Args:
            futures_trader: Instancia de FuturesTrader
            position_monitor: Instancia de PositionMonitor
            notifier: Instancia de TelegramNotifier
            autonomy_controller: Instancia de AutonomyController (para notificar P&L correcto al RL Agent)
        """
        self.futures_trader = futures_trader
        self.position_monitor = position_monitor
        self.notifier = notifier
        self.autonomy_controller = autonomy_controller

        # Estado del modo de prueba
        self.running = False
        self.task = None

        # Configuración
        self.symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        self.trade_amount = 100  # $100 por trade (mínimo de Binance Futures)
        self.trade_interval = 180  # 3 minutos = 180 segundos
        self.hold_time = 30  # Mantener posición 30 segundos

        # Estadísticas
        self.total_trades = 0
        self.winners = 0
        self.losers = 0
        self.total_pnl = 0.0
        self.start_time = None
        self.last_trade_time = None

        logger.info("🧪 Test Mode inicializado")

    async def start(self):
        """Inicia el modo de prueba automático"""
        if self.running:
            logger.warning("⚠️ Test Mode ya está corriendo")
            return False

        if not self.futures_trader:
            logger.error("❌ No hay futures_trader disponible")
            return False

        self.running = True
        self.start_time = datetime.now()

        # CRÍTICO: Activar flag en autonomy_controller para ignorar Position Monitor
        if self.autonomy_controller:
            self.autonomy_controller.test_mode_active = True
            logger.info("🔒 Autonomy Controller: Test Mode activo (ignorará Position Monitor)")

        logger.info("🟢 Test Mode INICIADO")
        logger.info(f"⚙️ Config: ${self.trade_amount} cada {self.trade_interval}s")

        # Enviar notificación
        if self.notifier:
            try:
                await self.notifier.send_message(
                    "🧪 **MODO DE PRUEBA INICIADO**\n\n"
                    f"⚙️ Configuración:\n"
                    f"   • Frecuencia: 1 trade cada {self.trade_interval // 60} minutos\n"
                    f"   • Tamaño: ${self.trade_amount} por trade\n"
                    f"   • Pares: {', '.join(self.symbols)}\n"
                    f"   • Leverage: 2-3x (aleatorio)\n\n"
                    f"📊 El bot ejecutará trades automáticamente.\n"
                    f"   Usa /test_status para ver progreso."
                )
            except Exception as e:
                logger.error(f"Error enviando notificación: {e}")

        # Iniciar loop en background
        self.task = asyncio.create_task(self._test_loop())
        return True

    def stop(self):
        """Detiene el modo de prueba"""
        if not self.running:
            logger.warning("⚠️ Test Mode no está corriendo")
            return False

        self.running = False

        # CRÍTICO: Desactivar flag en autonomy_controller
        if self.autonomy_controller:
            self.autonomy_controller.test_mode_active = False
            logger.info("🔓 Autonomy Controller: Test Mode desactivado (Position Monitor activo nuevamente)")

        if self.task:
            self.task.cancel()

        logger.info("🔴 Test Mode DETENIDO")
        logger.info(f"📊 Stats finales: {self.total_trades} trades, P&L: ${self.total_pnl:+.2f}")

        return True

    def get_stats(self) -> Dict:
        """Retorna estadísticas actuales del modo de prueba"""
        win_rate = (self.winners / self.total_trades * 100) if self.total_trades > 0 else 0

        # Calcular tiempo de próximo trade
        next_trade_in = None
        if self.running and self.last_trade_time:
            elapsed = (datetime.now() - self.last_trade_time).total_seconds()
            next_trade_in = max(0, self.trade_interval - elapsed)

        return {
            'running': self.running,
            'total_trades': self.total_trades,
            'winners': self.winners,
            'losers': self.losers,
            'total_pnl': self.total_pnl,
            'win_rate': win_rate,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'last_trade_time': self.last_trade_time.isoformat() if self.last_trade_time else None,
            'next_trade_in': next_trade_in
        }

    async def _test_loop(self):
        """Loop principal que ejecuta trades cada N minutos"""
        try:
            while self.running:
                try:
                    # Ejecutar un trade de prueba
                    await self._execute_test_trade()

                    # Esperar hasta el próximo trade
                    logger.info(f"⏳ Esperando {self.trade_interval}s hasta próximo trade...")
                    await asyncio.sleep(self.trade_interval)

                except asyncio.CancelledError:
                    logger.info("⏸️ Test loop cancelado")
                    break

                except Exception as e:
                    logger.error(f"❌ Error en test loop: {e}", exc_info=True)
                    # Continuar a pesar del error
                    await asyncio.sleep(10)

        except Exception as e:
            logger.error(f"❌ Error fatal en test loop: {e}", exc_info=True)
            self.running = False

    async def _execute_test_trade(self):
        """Ejecuta un trade de prueba completo (abrir → esperar → cerrar)"""
        try:
            # 1. Elegir parámetros aleatorios
            symbol = random.choice(self.symbols)
            side = random.choice(['BUY', 'SELL'])
            leverage = random.choice([2, 3])

            logger.info(f"\n{'='*60}")
            logger.info(f"🧪 TEST TRADE #{self.total_trades + 1}")
            logger.info(f"Symbol: {symbol} | Side: {side} | Leverage: {leverage}x")
            logger.info(f"{'='*60}")

            # Notificar inicio
            if self.notifier:
                try:
                    await self.notifier.send_message(
                        f"🧪 **Test Trade #{self.total_trades + 1}**\n"
                        f"📊 {symbol} | {'🟢 LONG' if side == 'BUY' else '🔴 SHORT'} | {leverage}x\n"
                        f"💰 ${self.trade_amount}"
                    )
                except:
                    pass

            # 2. Abrir posición
            logger.info(f"📈 Abriendo posición {side} en {symbol}...")

            try:
                result = self.futures_trader.open_position(
                    symbol=symbol,
                    side=side,
                    usdt_amount=self.trade_amount,
                    leverage=leverage,
                    stop_loss_pct=2.0,  # 2% SL
                    take_profit_pct=3.0  # 3% TP
                )

                # Verificar éxito: futures_trader devuelve dict con 'symbol' si exitoso
                if not result or 'symbol' not in result:
                    error_msg = result.get('error', 'Unknown error') if isinstance(result, dict) else 'No result'
                    logger.error(f"❌ Error abriendo posición: {error_msg}")
                    return

                entry_price = result.get('entry_price', 0)
                quantity = result.get('quantity', 0)
                order_id = result.get('market_order_id', result.get('order_id', 'N/A'))

                logger.info(f"✅ Posición abierta: {quantity} @ ${entry_price:,.2f} (ID: {order_id})")

            except Exception as e:
                logger.error(f"❌ Excepción abriendo posición: {e}", exc_info=True)
                return

            # 3. Esperar N segundos
            logger.info(f"⏳ Manteniendo posición por {self.hold_time}s...")
            await asyncio.sleep(self.hold_time)

            # 4. Cerrar posición
            logger.info(f"📉 Cerrando posición en {symbol}...")

            try:
                close_result = self.futures_trader.close_position(
                    symbol=symbol,
                    reason="Test mode auto-close"
                )

                # Verificar éxito: futures_trader devuelve dict con 'symbol' si exitoso
                if not close_result or 'symbol' not in close_result:
                    error_msg = close_result.get('error', 'Unknown error') if isinstance(close_result, dict) else 'No result'
                    logger.error(f"❌ Error cerrando posición: {error_msg}")
                    return

                exit_price = close_result.get('exit_price', 0)
                realized_pnl = close_result.get('pnl', 0)  # P&L en USDT (absoluto)

                logger.info(f"✅ Posición cerrada @ ${exit_price:,.2f}")
                logger.info(f"💰 P&L: ${realized_pnl:+.2f}")

                # CRÍTICO: Notificar autonomy_controller DIRECTAMENTE con el P&L CORRECTO
                # Esto evita que el RL Agent aprenda el P&L incorrecto del position_monitor
                if self.autonomy_controller:
                    try:
                        # Construir closed_info con el P&L REAL del test
                        test_closed_info = {
                            'symbol': symbol,
                            'side': 'LONG' if side == 'BUY' else 'SHORT',
                            'realized_pnl': realized_pnl,  # P&L REAL del test
                            'realized_pnl_pct': close_result.get('pnl_pct', 0),  # ROE%
                            'leverage': leverage,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'quantity': quantity,
                            'reason': 'Test mode auto-close',
                            'trade_id': f"test_{self.total_trades + 1}"
                        }

                        # Calcular reward (P&L en USDT)
                        reward = realized_pnl

                        # Notificar directamente al RL Agent
                        self.autonomy_controller.update_from_trade_result(test_closed_info, reward)

                        logger.info(f"✅ RL Agent notificado con P&L CORRECTO: ${realized_pnl:+.2f}")
                    except Exception as e:
                        logger.error(f"❌ Error notificando RL Agent: {e}", exc_info=True)

                # CRÍTICO: Guardar en historial del position_monitor para el dashboard
                if self.position_monitor:
                    try:
                        trade_record = {
                            'id': len(self.position_monitor.closed_trades) + 1,
                            'symbol': symbol,
                            'side': 'LONG' if side == 'BUY' else 'SHORT',
                            'leverage': leverage,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'quantity': quantity,
                            'realized_pnl': realized_pnl,
                            'realized_pnl_pct': close_result.get('pnl_pct', 0),
                            'reason': 'Test mode auto-close',
                            'open_time': datetime.now().isoformat(),  # Aproximado
                            'close_time': datetime.now().isoformat(),
                            'trade_id': f"test_{self.total_trades + 1}"
                        }
                        self.position_monitor.closed_trades.append(trade_record)
                        logger.info(f"💾 Trade guardado en historial del dashboard (ID: {trade_record['id']})")
                    except Exception as e:
                        logger.error(f"❌ Error guardando en historial: {e}", exc_info=True)

                # 5. Actualizar estadísticas
                self.total_trades += 1
                self.last_trade_time = datetime.now()

                if realized_pnl > 0:
                    self.winners += 1
                    emoji = "✅"
                else:
                    self.losers += 1
                    emoji = "❌"

                self.total_pnl += realized_pnl

                # Estadísticas actuales
                win_rate = (self.winners / self.total_trades * 100) if self.total_trades > 0 else 0

                logger.info(f"\n{emoji} RESULTADO FINAL:")
                logger.info(f"   P&L: ${realized_pnl:+.2f}")
                logger.info(f"   Total trades: {self.total_trades}")
                logger.info(f"   Win rate: {win_rate:.1f}%")
                logger.info(f"   P&L acumulado: ${self.total_pnl:+.2f}")
                logger.info(f"{'='*60}\n")

                # Notificar resultado
                if self.notifier:
                    try:
                        await self.notifier.send_message(
                            f"{emoji} **Test Trade Completado**\n\n"
                            f"📊 Resultado: ${realized_pnl:+.2f}\n"
                            f"📈 Stats:\n"
                            f"   • Total: {self.total_trades} trades\n"
                            f"   • Win rate: {win_rate:.1f}%\n"
                            f"   • P&L total: ${self.total_pnl:+.2f}"
                        )
                    except:
                        pass

            except Exception as e:
                logger.error(f"❌ Excepción cerrando posición: {e}", exc_info=True)
                return

        except Exception as e:
            logger.error(f"❌ Error ejecutando test trade: {e}", exc_info=True)
