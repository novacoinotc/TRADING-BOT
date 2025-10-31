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
        Format advanced trading signal as HTML message

        Args:
            pair: Trading pair
            indicators: Technical indicators
            signals: Trading signals

        Returns:
            Formatted HTML message
        """
        # Check if this is a flash signal
        is_flash = signals.get('signal_type') == 'FLASH'
        timeframe = signals.get('timeframe', '1h')

        # Emoji based on action
        if signals['action'] == 'BUY':
            emoji = '🟢'
            action_text = '<b>COMPRAR</b>'
            entry_label = 'Entrada sugerida'
        elif signals['action'] == 'SELL':
            emoji = '🔴'
            action_text = '<b>VENDER</b>'
            entry_label = 'Entrada sugerida'
        else:
            emoji = '⚪'
            action_text = 'MANTENER'
            entry_label = 'Precio'

        # Header with score
        if is_flash:
            message = f"⚡ <b>SEÑAL FLASH - RIESGOSA</b> ⚡\n"
            message += f"⚠️ <i>Operación de alto riesgo ({timeframe})</i>\n\n"
        else:
            message = f"{emoji} <b>SEÑAL DE TRADING FUERTE</b> {emoji}\n\n"
        message += f"<b>Par:</b> {pair}\n"
        message += f"<b>Acción:</b> {action_text}\n"

        # Quality score
        score = signals.get('score', 0)
        max_score = signals.get('max_score', 10)
        message += f"💎 <b>Calidad:</b> {score:.1f}/{max_score} "

        if score >= 9:
            message += "(EXCEPCIONAL)\n"
        elif score >= 8:
            message += "(MUY ALTA)\n"
        elif score >= 7:
            message += "(ALTA)\n"

        message += f"<b>Fuerza:</b> {'⭐' * signals.get('strength', 0)}\n\n"

        # Price and entry info
        current_price = indicators['current_price']
        message += f"💰 <b>Precio actual:</b> ${current_price:,.2f}\n"

        # Entry range (±0.5%)
        if signals['action'] != 'HOLD':
            entry_low = current_price * 0.995
            entry_high = current_price * 1.005
            message += f"📍 <b>{entry_label}:</b> ${entry_low:,.2f} - ${entry_high:,.2f}\n\n"

        # Stop Loss and Take Profit
        if 'stop_loss' in signals and signals['stop_loss']:
            sl = signals['stop_loss']
            tp = signals['take_profit']
            message += f"🎯 <b>Take Profit:</b>\n"
            message += f"   TP1: ${tp['tp1']:,.2f} ({((tp['tp1']/current_price-1)*100):+.1f}%)\n"
            message += f"   TP2: ${tp['tp2']:,.2f} ({((tp['tp2']/current_price-1)*100):+.1f}%)\n"
            message += f"   TP3: ${tp['tp3']:,.2f} ({((tp['tp3']/current_price-1)*100):+.1f}%)\n\n"

            sl_pct = ((sl/current_price-1)*100) if signals['action'] == 'BUY' else ((current_price/sl-1)*100)
            message += f"🛡️ <b>Stop Loss:</b> ${sl:,.2f} ({sl_pct:+.1f}%)\n"
            message += f"📊 <b>Ratio R/R:</b> 1:{signals.get('risk_reward', 0)}\n\n"

        # Multi-timeframe analysis
        mtf_trends = indicators.get('mtf_trends', {})
        if mtf_trends:
            message += "📈 <b>Análisis Multi-Timeframe:</b>\n"
            for tf, trend in mtf_trends.items():
                if tf != 'alignment':
                    emoji_trend = '✅' if trend == signals['action'].lower() else '❌' if trend != 'neutral' else '⚪'
                    message += f"• {tf}: {trend.capitalize()} {emoji_trend}\n"
            message += "\n"

        # Key indicators
        message += f"📊 <b>Indicadores ({score:.1f} pts):</b>\n"
        message += f"• RSI: {indicators['rsi']:.1f}"

        if indicators['rsi'] < 30:
            message += " (sobreventa fuerte)"
        elif indicators['rsi'] > 70:
            message += " (sobrecompra fuerte)"
        message += "\n"

        message += f"• MACD: "
        if indicators['macd'] > indicators['macd_signal']:
            message += "Alcista ✅\n"
        else:
            message += "Bajista ❌\n"

        # Volume analysis
        volume_ratio = indicators.get('volume_ratio', 1)
        message += f"• Volumen: "
        if volume_ratio > 1.5:
            message += f"+{(volume_ratio-1)*100:.0f}% 🔥\n"
        elif volume_ratio > 1.2:
            message += f"+{(volume_ratio-1)*100:.0f}% ✅\n"
        else:
            message += f"{(volume_ratio-1)*100:+.0f}%\n"

        # Divergences
        divergences = indicators.get('divergences', {})
        if divergences.get('rsi_divergence'):
            message += f"• Divergencia: {divergences['rsi_divergence'].capitalize()} 💎\n"

        # Support/Resistance
        sr = indicators.get('support_resistance', {})
        if sr:
            message += f"• Soporte: ${sr.get('nearest_support', 0):,.2f}\n"
            message += f"• Resistencia: ${sr.get('nearest_resistance', 0):,.2f}\n"

        message += "\n"

        # Reasons
        message += "📈 <b>Razones:</b>\n"
        for reason in signals.get('reasons', [])[:6]:  # Limit to 6 reasons
            message += f"• {reason}\n"

        # Risk and confidence
        message += f"\n⚠️ <b>Riesgo:</b> {signals.get('risk_level', 'MEDIUM')}\n"
        message += f"💡 <b>Confianza:</b> {signals.get('confidence', 0)}%\n"

        # Footer based on signal type
        if is_flash:
            message += f"\n⏰ <i>Análisis flash {timeframe} - Operación opcional</i>"
        else:
            message += f"\n⏰ <i>Análisis multi-timeframe 1h/4h/1d</i>"

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

    async def send_trading_stats(self, stats: dict):
        """
        Send paper trading statistics

        Args:
            stats: Trading statistics dictionary
        """
        try:
            trading = stats.get('trading', {})
            ml_model = stats.get('ml_model', {})
            params = stats.get('optimized_params', {})

            message = "📊 <b>PAPER TRADING STATS</b>\n\n"

            # Balance info
            balance = trading.get('current_balance', 0)
            equity = trading.get('equity', 0)
            initial = trading.get('initial_balance', 50000)
            pnl = trading.get('net_pnl', 0)
            roi = trading.get('roi', 0)

            message += f"💰 <b>Balance:</b> ${balance:,.2f} USDT\n"
            message += f"💎 <b>Equity:</b> ${equity:,.2f} USDT\n"
            message += f"📈 <b>P&L:</b> ${pnl:,.2f} ({roi:+.2f}%)\n\n"

            # Trading stats
            total_trades = trading.get('total_trades', 0)
            win_rate = trading.get('win_rate', 0)
            profit_factor = trading.get('profit_factor', 0)
            sharpe = trading.get('sharpe_ratio', 0)
            dd = trading.get('max_drawdown', 0)

            message += f"📊 <b>Trading:</b>\n"
            message += f"• Total Trades: {total_trades}\n"
            message += f"• Win Rate: {win_rate:.1f}%\n"
            message += f"• Profit Factor: {profit_factor:.2f}\n"
            message += f"• Sharpe Ratio: {sharpe:.2f}\n"
            message += f"• Max Drawdown: {dd:.2f}%\n\n"

            # Open positions
            open_pos = trading.get('open_positions', 0)
            message += f"🔄 <b>Posiciones Abiertas:</b> {open_pos}\n\n"

            # ML Model info
            if ml_model.get('available'):
                metrics = ml_model.get('metrics', {})
                acc = metrics.get('test_accuracy', 0)
                prec = metrics.get('test_precision', 0)

                message += f"🧠 <b>ML Model:</b>\n"
                message += f"• Accuracy: {acc:.2%}\n"
                message += f"• Precision: {prec:.2%}\n"
                message += f"• Samples: {metrics.get('samples_total', 0)}\n\n"

            # Optimized params
            message += f"⚙️ <b>Parámetros:</b>\n"
            message += f"• Flash Threshold: {params.get('flash_threshold', 0):.1f}\n"
            message += f"• Min Confidence: {params.get('flash_min_confidence', 0)}%\n"
            message += f"• Position Size: {params.get('position_size_pct', 0):.1f}%\n"

            await self.send_status_message(message)

        except Exception as e:
            logger.error(f"Failed to send trading stats: {e}")
