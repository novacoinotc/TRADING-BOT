"""
Telegram Bot Module
Handles notifications and user interactions via Telegram
"""
import asyncio
from telegram import Bot
from telegram.error import TelegramError
from config import config
import logging

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """
    Handles Telegram notifications for trading signals
    """

    def __init__(self, tracker=None):
        if not config.TELEGRAM_BOT_TOKEN:
            raise ValueError("TELEGRAM_BOT_TOKEN not configured")
        if not config.TELEGRAM_CHAT_ID:
            raise ValueError("TELEGRAM_CHAT_ID not configured")

        self.bot = Bot(token=config.TELEGRAM_BOT_TOKEN)
        self.chat_id = config.TELEGRAM_CHAT_ID
        self.last_signals = {}  # Avoid duplicate notifications
        self.tracker = tracker  # Signal tracker for accuracy measurement

    async def send_signal(self, pair: str, analysis: dict):
        """
        Send trading signal notification

        Args:
            pair: Trading pair (e.g., 'BTC/USDT')
            analysis: Analysis results with indicators and signals
        """
        if not analysis or 'signals' not in analysis:
            return

        signals = analysis['signals']
        indicators = analysis['indicators']

        # Check if we should send this signal (avoid spam)
        if not self._should_send_signal(pair, signals):
            return

        # Format message
        message = self._format_signal_message(pair, indicators, signals)

        # Send message
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='HTML'
            )
            logger.info(f"Signal sent for {pair}: {signals['action']}")

            # Track signal for accuracy measurement
            if self.tracker and config.TRACK_SIGNALS:
                self.tracker.add_signal(
                    pair=pair,
                    action=signals['action'],
                    price=indicators['current_price'],
                    indicators=indicators,
                    reasons=signals['reasons']
                )

            # Update last signal
            self.last_signals[pair] = signals['action']

        except TelegramError as e:
            logger.error(f"Failed to send Telegram message: {e}")

    def _should_send_signal(self, pair: str, signals: dict) -> bool:
        """
        Determine if signal should be sent (avoid spam)

        Args:
            pair: Trading pair
            signals: Current signals

        Returns:
            True if signal should be sent
        """
        # Only send BUY or SELL signals
        if signals['action'] == 'HOLD':
            return False

        # Check if this is a new signal
        last_signal = self.last_signals.get(pair)
        if last_signal == signals['action']:
            return False  # Same signal already sent

        return True

    def _format_signal_message(self, pair: str, indicators: dict, signals: dict) -> str:
        """
        Format trading signal as HTML message

        Args:
            pair: Trading pair
            indicators: Technical indicators
            signals: Trading signals

        Returns:
            Formatted HTML message
        """
        # Emoji based on action
        if signals['action'] == 'BUY':
            emoji = '🟢'
            action_text = '<b>COMPRAR</b>'
        elif signals['action'] == 'SELL':
            emoji = '🔴'
            action_text = '<b>VENDER</b>'
        else:
            emoji = '⚪'
            action_text = 'MANTENER'

        # Build message
        message = f"{emoji} <b>SEÑAL DE TRADING</b> {emoji}\n\n"
        message += f"<b>Par:</b> {pair}\n"
        message += f"<b>Acción:</b> {action_text}\n"
        message += f"<b>Fuerza:</b> {'⭐' * signals['strength']}\n\n"

        message += f"💰 <b>Precio:</b> ${indicators['current_price']:,.2f}\n\n"

        message += "📊 <b>Indicadores:</b>\n"
        message += f"• RSI: {indicators['rsi']:.2f}\n"
        message += f"• MACD: {indicators['macd']:.4f}\n"
        message += f"• MACD Señal: {indicators['macd_signal']:.4f}\n"
        message += f"• EMA(9): ${indicators['ema_short']:,.2f}\n"
        message += f"• EMA(21): ${indicators['ema_medium']:,.2f}\n"
        message += f"• EMA(50): ${indicators['ema_long']:,.2f}\n\n"

        message += "📈 <b>Razones:</b>\n"
        for reason in signals['reasons']:
            message += f"• {reason}\n"

        message += f"\n⏰ <i>Análisis automático en timeframe {config.TIMEFRAME}</i>"

        return message

    async def send_status_message(self, message: str):
        """
        Send general status message

        Args:
            message: Status message to send
        """
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='HTML'
            )
        except TelegramError as e:
            logger.error(f"Failed to send status message: {e}")

    async def send_error_message(self, error: str):
        """
        Send error notification

        Args:
            error: Error message
        """
        message = f"⚠️ <b>ERROR</b>\n\n{error}"
        await self.send_status_message(message)
