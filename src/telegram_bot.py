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
        sentiment_data = analysis.get('sentiment_data')  # Get sentiment data if available
        orderbook_data = analysis.get('orderbook_data')  # Get order book data if available
        regime_data = analysis.get('regime_data')  # Get regime data if available

        # Check if we should send this signal (avoid spam)
        if not self._should_send_signal(pair, signals):
            return

        # Format message (pass full analysis to show trade decision)
        message = self._format_signal_message(pair, indicators, signals, sentiment_data, orderbook_data, regime_data, analysis)

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

    def _format_signal_message(self, pair: str, indicators: dict, signals: dict,
                              sentiment_data: dict = None, orderbook_data: dict = None,
                              regime_data: dict = None, analysis: dict = None) -> str:
        """
        Format advanced trading signal as HTML message

        Args:
            pair: Trading pair
            indicators: Technical indicators
            signals: Trading signals
            sentiment_data: Sentiment analysis data (optional)
            orderbook_data: Order book analysis data (optional)
            regime_data: Market regime data (optional)
            analysis: Full analysis dict including trade_result (optional)

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

        # PAR Y DECISIÓN (más claro sobre si se tomó el trade)
        message += f"<b>Par:</b> {pair}\n"

        # Verificar si el trade fue tomado
        trade_result = analysis.get('trade_result') if analysis else None
        if trade_result:
            if trade_result.get('status') == 'OPEN':
                message += f"✅ <b>TRADE ABIERTO:</b> {action_text}\n"
            else:
                message += f"⚠️ <b>SEÑAL DETECTADA</b> (evaluando condiciones)\n"
        else:
            message += f"<b>Señal:</b> {action_text}\n"

        # BREVE RESUMEN DEL "POR QUÉ" (top 3 razones)
        reasons_list = signals.get('reasons', [])
        if len(reasons_list) > 0:
            top_reasons = reasons_list[:2]  # Top 2 razones
            brief_summary = ", ".join(top_reasons)
            if len(brief_summary) > 100:
                brief_summary = brief_summary[:97] + "..."
            message += f"\n💡 <b>Por qué:</b> {brief_summary}\n"

        # Quality score
        score = signals.get('score', 0)
        max_score = signals.get('max_score', 10)
        message += f"\n💎 <b>Calidad:</b> {score:.1f}/{max_score} "

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

        # Market Regime (if available)
        if regime_data:
            message += "🎯 <b>Market Regime:</b>\n"

            regime = regime_data.get('regime', 'SIDEWAYS')
            regime_strength = regime_data.get('regime_strength', 'MODERATE')
            confidence = regime_data.get('confidence', 50)

            if regime == 'BULL':
                regime_emoji = "🐂"
                regime_label = "Alcista"
            elif regime == 'BEAR':
                regime_emoji = "🐻"
                regime_label = "Bajista"
            else:
                regime_emoji = "↔️"
                regime_label = "Lateral"

            message += f"• Régimen: {regime_label} {regime_emoji} ({regime_strength})\n"
            message += f"• Confianza: {confidence:.0f}%\n"
            message += "\n"

        # Order Book (if available)
        if orderbook_data:
            message += "📚 <b>Order Book:</b>\n"

            imbalance = orderbook_data.get('imbalance', 0)
            market_pressure = orderbook_data.get('market_pressure', 'NEUTRAL')

            if market_pressure == 'STRONG_BUY':
                pressure_emoji = "🟢🟢"
            elif market_pressure == 'BUY':
                pressure_emoji = "🟢"
            elif market_pressure == 'STRONG_SELL':
                pressure_emoji = "🔴🔴"
            elif market_pressure == 'SELL':
                pressure_emoji = "🔴"
            else:
                pressure_emoji = "⚪"

            message += f"• Presión: {market_pressure} {pressure_emoji}\n"
            message += f"• Imbalance: {imbalance:+.2f}\n"

            # Bid walls
            bid_walls = orderbook_data.get('bid_walls', [])
            if bid_walls and len(bid_walls) > 0:
                nearest_wall = bid_walls[0]
                message += f"• Pared Compra: ${nearest_wall['price']:,.2f} ({nearest_wall['distance_pct']:.1f}% away) 🛡️\n"

            # Ask walls
            ask_walls = orderbook_data.get('ask_walls', [])
            if ask_walls and len(ask_walls) > 0:
                nearest_wall = ask_walls[0]
                message += f"• Pared Venta: ${nearest_wall['price']:,.2f} ({nearest_wall['distance_pct']:.1f}% away) 🚧\n"

            message += "\n"

        # Sentiment Analysis (if available)
        if sentiment_data:
            message += "📰 <b>Sentiment Analysis:</b>\n"

            # Fear & Greed Index
            fg_index = sentiment_data.get('fear_greed_index', 0.5) * 100
            if fg_index < 25:
                fg_emoji = "😱"
                fg_label = "Extreme Fear"
            elif fg_index < 45:
                fg_emoji = "😰"
                fg_label = "Fear"
            elif fg_index < 55:
                fg_emoji = "😐"
                fg_label = "Neutral"
            elif fg_index < 75:
                fg_emoji = "😊"
                fg_label = "Greed"
            else:
                fg_emoji = "🤑"
                fg_label = "Extreme Greed"

            message += f"• Fear & Greed: {fg_index:.0f}/100 {fg_emoji} ({fg_label})\n"

            # News sentiment
            news_sentiment = sentiment_data.get('news_sentiment_overall', 0.5)
            if news_sentiment >= 0.6:
                news_emoji = "📈"
                news_label = "Positivo"
            elif news_sentiment >= 0.4:
                news_emoji = "➡️"
                news_label = "Neutral"
            else:
                news_emoji = "📉"
                news_label = "Negativo"

            message += f"• Noticias: {news_sentiment:.2f} {news_emoji} ({news_label})\n"

            # High impact news
            high_impact = int(sentiment_data.get('high_impact_news_count', 0))
            if high_impact > 0:
                message += f"• Noticias de Alto Impacto: {high_impact} 🔥\n"

            # Sentiment trend
            if sentiment_data.get('sentiment_trend_improving', 0) > 0.5:
                message += f"• Tendencia: Mejorando ✅\n"

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

    async def send_trade_opened(self, trade_data: dict):
        """
        Notifica cuando se abre un trade

        Args:
            trade_data: Información del trade abierto
        """
        try:
            pair = trade_data.get('pair', 'UNKNOWN')
            side = trade_data.get('side', 'BUY')
            entry_price = trade_data.get('entry_price', 0)
            quantity = trade_data.get('quantity', 0)
            position_value = trade_data.get('position_value', 0)
            trade_type = trade_data.get('trade_type', 'FUTURES')  # Default FUTURES
            leverage = trade_data.get('leverage', 1)

            emoji = "🟢" if side == 'BUY' else "🔴"
            type_emoji = "⚡" if trade_type == 'FUTURES' else "💎"

            message = (
                f"{emoji} <b>TRADE ABIERTO</b> {emoji}\n\n"
                f"📌 <b>Par:</b> {pair}\n"
                f"📊 <b>Dirección:</b> {side}\n"
                f"{type_emoji} <b>Tipo:</b> {trade_type}\n"
            )

            # Add leverage info for FUTURES
            if trade_type == 'FUTURES':
                liquidation_price = trade_data.get('liquidation_price')
                message += f"📈 <b>Apalancamiento:</b> {leverage}x\n"
                if liquidation_price:
                    message += f"⚠️ <b>Precio Liquidación:</b> ${liquidation_price:,.4f}\n"

            message += (
                f"💰 <b>Precio Entrada:</b> ${entry_price:,.4f}\n"
                f"📦 <b>Cantidad:</b> {quantity:.6f}\n"
                f"💵 <b>Valor Posición:</b> ${position_value:,.2f}\n"
            )

            # Add TPs if available
            take_profit = trade_data.get('take_profit', {})
            if take_profit:
                tp1 = take_profit.get('tp1', 0)
                tp2 = take_profit.get('tp2', 0)
                tp3 = take_profit.get('tp3', 0)

                message += f"\n🎯 <b>Take Profits:</b>\n"
                if tp1:
                    tp1_pct = ((tp1 / entry_price - 1) * 100) if side == 'BUY' else ((entry_price / tp1 - 1) * 100)
                    message += f"   TP1: ${tp1:,.4f} ({tp1_pct:+.2f}%)\n"
                if tp2:
                    tp2_pct = ((tp2 / entry_price - 1) * 100) if side == 'BUY' else ((entry_price / tp2 - 1) * 100)
                    message += f"   TP2: ${tp2:,.4f} ({tp2_pct:+.2f}%)\n"
                if tp3:
                    tp3_pct = ((tp3 / entry_price - 1) * 100) if side == 'BUY' else ((entry_price / tp3 - 1) * 100)
                    message += f"   TP3: ${tp3:,.4f} ({tp3_pct:+.2f}%)\n"

            # Add SL if available
            stop_loss = trade_data.get('stop_loss')
            if stop_loss:
                sl_pct = ((stop_loss / entry_price - 1) * 100) if side == 'BUY' else ((entry_price / stop_loss - 1) * 100)
                message += f"\n🛡️ <b>Stop Loss:</b> ${stop_loss:,.4f} ({sl_pct:+.2f}%)"

            await self.send_status_message(message)

        except Exception as e:
            logger.error(f"Error enviando notificación de trade abierto: {e}")

    async def send_trade_closed(self, trade_data: dict):
        """
        Notifica cuando se cierra un trade

        Args:
            trade_data: Información del trade cerrado
        """
        try:
            pair = trade_data.get('pair', 'UNKNOWN')
            side = trade_data.get('side', 'BUY')
            entry_price = trade_data.get('entry_price', 0)
            exit_price = trade_data.get('exit_price', 0)
            pnl = trade_data.get('pnl', 0)
            pnl_pct = trade_data.get('pnl_pct', 0)
            reason = trade_data.get('reason', 'UNKNOWN')
            trade_type = trade_data.get('trade_type', 'FUTURES')  # Default FUTURES
            leverage = trade_data.get('leverage', 1)

            # Emoji según resultado
            if pnl > 0:
                emoji = "✅"
                result_text = "GANANCIA"
            else:
                emoji = "❌"
                result_text = "PÉRDIDA"

            type_emoji = "⚡" if trade_type == 'FUTURES' else "💎"

            message = (
                f"{emoji} <b>TRADE CERRADO - {result_text}</b> {emoji}\n\n"
                f"📌 <b>Par:</b> {pair}\n"
                f"📊 <b>Dirección:</b> {side}\n"
                f"{type_emoji} <b>Tipo:</b> {trade_type}"
            )

            if trade_type == 'FUTURES' and leverage > 1:
                message += f" ({leverage}x)\n"
            else:
                message += "\n"

            message += (
                f"💰 <b>Entrada:</b> ${entry_price:,.4f}\n"
                f"💰 <b>Salida:</b> ${exit_price:,.4f}\n"
                f"📈 <b>P&L:</b> ${pnl:,.2f} (<b>{pnl_pct:+.2f}%</b>)\n"
                f"🏁 <b>Razón:</b> {reason}\n"
            )

            # Extra info si es ganancia > 1%
            if pnl_pct > 1.0:
                message += f"\n🎉 <b>Excelente trade!</b>"
            elif pnl_pct > 0.5:
                message += f"\n👍 <b>Buen trade</b>"

            await self.send_status_message(message)

        except Exception as e:
            logger.error(f"Error enviando notificación de trade cerrado: {e}")

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
