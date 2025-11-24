"""
Daily Reporter Module
Generates and sends daily performance reports
"""
import asyncio
from datetime import datetime, time
import pytz
import logging
from typing import Dict
from config import config
from src.signal_tracker import SignalTracker
from src.telegram_bot import TelegramNotifier

logger = logging.getLogger(__name__)


class DailyReporter:
    """
    Generates daily performance reports and sends them via Telegram
    """

    def __init__(self, tracker: SignalTracker, notifier: TelegramNotifier):
        self.tracker = tracker
        self.notifier = notifier
        self.timezone = pytz.timezone(config.TIMEZONE)
        self.report_hour = config.DAILY_REPORT_HOUR
        self.report_minute = config.DAILY_REPORT_MINUTE
        self.last_report_date = None

    async def should_send_report(self) -> bool:
        """
        Check if it's time to send the daily report

        Returns:
            True if report should be sent
        """
        now = datetime.now(self.timezone)
        current_date = now.date()

        # Check if we already sent report today
        if self.last_report_date == current_date:
            return False

        # Check if it's the right time
        current_time = now.time()
        report_time = time(hour=self.report_hour, minute=self.report_minute)

        # Send report if it's past the scheduled time and we haven't sent it today
        if current_time >= report_time:
            return True

        return False

    def _format_daily_report(self, stats: Dict, today_signals: list) -> str:
        """
        Format the daily report message

        Args:
            stats: Accuracy statistics
            today_signals: List of today's signals

        Returns:
            Formatted HTML message
        """
        now = datetime.now(self.timezone)
        date_str = now.strftime("%d/%m/%Y")

        message = "📊 <b>REPORTE DIARIO DE TRADING</b> 📊\n"
        message += f"📅 Fecha: {date_str}\n"
        message += "━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        # Overall Statistics
        message += "<b>📈 RESUMEN DEL DÍA</b>\n"
        message += f"• Señales enviadas: <b>{stats['total_signals']}</b>\n"

        if stats['total_signals'] > 0:
            message += f"• ✅ Exitosas: {stats['successful']}\n"
            message += f"• ❌ Fallidas: {stats['failed']}\n"
            message += f"• ⏳ Pendientes: {stats['total_signals'] - stats['successful'] - stats['failed']}\n"
            message += f"• 🎯 Precisión: <b>{stats['accuracy']:.1f}%</b>\n"
            message += f"• 💰 Ganancia promedio: <b>{stats['avg_profit']:.2f}%</b>\n\n"

            # Best and Worst Signals
            if stats['best_signal']:
                best = stats['best_signal']
                message += "🏆 <b>MEJOR SEÑAL:</b>\n"
                message += f"  {best['pair']} - {best['action']}\n"
                message += f"  Ganancia: <b>+{best['profit_percent']:.2f}%</b>\n"
                message += f"  Precio: ${best['signal_price']:.2f} → ${best['outcome_price']:.2f}\n\n"

            if stats['worst_signal'] and stats['worst_signal']['profit_percent'] < 0:
                worst = stats['worst_signal']
                message += "⚠️ <b>PEOR SEÑAL:</b>\n"
                message += f"  {worst['pair']} - {worst['action']}\n"
                message += f"  Pérdida: <b>{worst['profit_percent']:.2f}%</b>\n"
                message += f"  Precio: ${worst['signal_price']:.2f} → ${worst['outcome_price']:.2f}\n\n"

            # Stats by Pair
            if stats['by_pair']:
                message += "📊 <b>POR PAR:</b>\n"
                # Sort by total signals
                sorted_pairs = sorted(stats['by_pair'].items(),
                                     key=lambda x: x[1]['total'],
                                     reverse=True)
                for pair, pair_stats in sorted_pairs[:5]:  # Top 5
                    if pair_stats['total'] > 0:
                        message += f"  • {pair}: {pair_stats['success']}/{pair_stats['total']} "
                        message += f"({pair_stats['accuracy']:.0f}%)\n"
                message += "\n"

            # Stats by Action
            if stats['by_action']:
                message += "🎲 <b>POR ACCIÓN:</b>\n"
                for action, action_stats in stats['by_action'].items():
                    if action_stats['total'] > 0:
                        emoji = "🟢" if action == "BUY" else "🔴"
                        message += f"  {emoji} {action}: {action_stats['success']}/{action_stats['total']} "
                        message += f"({action_stats['accuracy']:.0f}%)\n"
                message += "\n"

        else:
            message += "\nℹ️ No se enviaron señales hoy.\n\n"

        # List of today's signals
        if today_signals:
            message += "📋 <b>SEÑALES DEL DÍA:</b>\n"
            for i, signal in enumerate(today_signals[:10], 1):  # Max 10
                signal_time = datetime.fromisoformat(signal['signal_time'])
                time_str = signal_time.strftime("%H:%M")

                status_emoji = {
                    'success': '✅',
                    'failed': '❌',
                    'pending': '⏳'
                }.get(signal['status'], '❓')

                action_emoji = '🟢' if signal['action'] == 'BUY' else '🔴'

                profit_str = ""
                if signal['profit_percent'] is not None:
                    sign = "+" if signal['profit_percent'] > 0 else ""
                    profit_str = f" ({sign}{signal['profit_percent']:.1f}%)"

                message += f"{i}. {status_emoji} {action_emoji} <b>{signal['pair']}</b> @ {time_str}{profit_str}\n"

            if len(today_signals) > 10:
                message += f"\n... y {len(today_signals) - 10} señales más\n"

        message += "\n━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        message += "🤖 <i>Bot de Señales de Trading</i>\n"
        message += f"⏰ Reporte generado: {now.strftime('%H:%M')} CDMX\n\n"

        # Performance evaluation
        if stats['total_signals'] > 0:
            if stats['accuracy'] >= 70:
                message += "🎉 <b>¡Excelente día! Alta precisión.</b>"
            elif stats['accuracy'] >= 50:
                message += "👍 <b>Día aceptable. Precisión moderada.</b>"
            else:
                message += "⚠️ <b>Día complicado. Revisar estrategia.</b>"

        return message

    async def send_daily_report(self):
        """
        Generate and send the daily report
        """
        try:
            logger.info("Generating daily report...")

            # Get today's signals
            today_signals = self.tracker.get_today_signals()

            # Get accuracy stats for today
            stats = self.tracker.get_accuracy_stats(days=1)

            # Format message
            message = self._format_daily_report(stats, today_signals)

            # Send report
            await self.notifier.send_status_message(message)

            # Update last report date
            self.last_report_date = datetime.now(self.timezone).date()

            logger.info(f"Daily report sent: {stats['total_signals']} signals, {stats['accuracy']:.1f}% accuracy")

        except Exception as e:
            logger.error(f"Error sending daily report: {e}")

    async def check_and_send_report(self):
        """
        Check if it's time and send report if needed
        """
        if await self.should_send_report():
            await self.send_daily_report()

    async def monitor_schedule(self):
        """
        Continuously monitor for report time (to be called from main loop)
        """
        while True:
            try:
                await self.check_and_send_report()
                # Check every 5 minutes
                await asyncio.sleep(300)
            except Exception as e:
                logger.error(f"Error in report monitor: {e}")
                await asyncio.sleep(60)
