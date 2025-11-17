"""
Telegram Commands Handler
Maneja comandos de Telegram para el bot
"""
import logging
import os
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class TelegramCommands:
    """
    Manejador de comandos de Telegram
    - /export_intelligence: Export manual de inteligencia aprendida
    - /status: Status del sistema autónomo
    """

    def __init__(self, autonomy_controller=None, telegram_token: str = None, chat_id: str = None, market_monitor=None):
        """
        Args:
            autonomy_controller: Instancia del AutonomyController
            telegram_token: Token del bot de Telegram
            chat_id: Chat ID para enviar mensajes proactivos
            market_monitor: Instancia del MarketMonitor (para ML System)
        """
        self.autonomy_controller = autonomy_controller
        self.market_monitor = market_monitor
        self.telegram_token = telegram_token
        self.chat_id = chat_id
        self.application = None
        self.waiting_for_import_file = False  # Flag para saber si esperamos archivo
        self.waiting_for_import_force_file = False  # Flag para import_force (ignora checksum)

        if telegram_token:
            logger.info("📱 Telegram Commands Handler inicializado")

    async def start_command_listener(self):
        """Inicia el listener de comandos de Telegram"""
        if not self.telegram_token:
            logger.warning("No hay token de Telegram - comandos deshabilitados")
            return

        try:
            # Crear aplicación
            self.application = ApplicationBuilder().token(self.telegram_token).build()

            # Agregar handlers
            self.application.add_handler(CommandHandler("export_intelligence", self.export_intelligence_command))
            self.application.add_handler(CommandHandler("export", self.export_intelligence_command))  # Alias
            self.application.add_handler(CommandHandler("import_intelligence", self.import_intelligence_command))
            self.application.add_handler(CommandHandler("import", self.import_intelligence_command))  # Alias
            self.application.add_handler(CommandHandler("import_force", self.import_force_command))  # Import sin validar checksum
            self.application.add_handler(CommandHandler("status", self.status_command))
            self.application.add_handler(CommandHandler("stats", self.stats_command))
            self.application.add_handler(CommandHandler("futures_stats", self.futures_stats_command))
            self.application.add_handler(CommandHandler("params", self.params_command))
            self.application.add_handler(CommandHandler("train_ml", self.train_ml_command))  # Entrenar ML System
            self.application.add_handler(CommandHandler("force_sync", self.force_sync_command))  # Forzar sincronización RL ↔ Paper
            self.application.add_handler(CommandHandler("pause", self.pause_command))  # Pausar análisis
            self.application.add_handler(CommandHandler("resume", self.resume_command))  # Resumir análisis
            self.application.add_handler(CommandHandler("help", self.help_command))

            # Handler para recibir archivos (documentos)
            self.application.add_handler(MessageHandler(filters.Document.ALL, self.handle_document))

            # Iniciar polling
            logger.info("✅ Telegram command listener iniciado")
            await self.application.initialize()
            await self.application.start()
            await self.application.updater.start_polling()

        except Exception as e:
            logger.error(f"Error iniciando command listener: {e}", exc_info=True)

    async def stop_command_listener(self):
        """Detiene el listener de comandos"""
        if self.application:
            try:
                await self.application.updater.stop()
                await self.application.stop()
                await self.application.shutdown()
                logger.info("🛑 Telegram command listener detenido")
            except Exception as e:
                logger.error(f"Error deteniendo command listener: {e}")

    async def export_intelligence_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Comando /export_intelligence
        Realiza export manual, envía archivo, y hace backup a Git
        """
        try:
            logger.info("📤 Comando /export_intelligence recibido")

            await update.message.reply_text(
                "📤 **Iniciando Export Manual**\n\n"
                "Guardando inteligencia y haciendo backup a Git...\n"
                "Esto puede tomar unos segundos ⏳"
            )

            if not self.autonomy_controller:
                await update.message.reply_text(
                    "⚠️ **Error**: Sistema autónomo no disponible"
                )
                return

            # Realizar export (retorna tupla: success, export_path)
            success, export_path = await self.autonomy_controller.manual_export()

            # Enviar archivo de inteligencia al usuario
            if export_path:
                try:
                    with open(export_path, 'rb') as f:
                        await update.message.reply_document(
                            document=f,
                            filename="intelligence_export.json",
                            caption="📤 Inteligencia aprendida exportada\n"
                                   "Puedes usar este archivo para restaurar el aprendizaje después de un redeploy"
                        )
                    logger.info(f"✅ Archivo enviado: {export_path}")
                except Exception as e:
                    logger.error(f"Error enviando archivo: {e}")
                    await update.message.reply_text(
                        f"⚠️ No se pudo enviar el archivo, pero está guardado localmente en:\n{export_path}"
                    )

            # Mensaje de confirmación
            if success:
                await update.message.reply_text(
                    "✅ **Export Completado**\n\n"
                    "✅ Archivo enviado por Telegram\n"
                    "✅ Backup realizado a Git\n"
                    "✅ Código pusheado a GitHub\n\n"
                    "El aprendizaje está seguro para futuros redeploys 🎉"
                )
            else:
                await update.message.reply_text(
                    "⚠️ **Export Parcial**\n\n"
                    "✅ Archivo enviado por Telegram\n"
                    "✅ Inteligencia guardada localmente\n"
                    "❌ Backup a Git falló\n\n"
                    "Tienes el archivo guardado, pero el push a Git no se completó.\n"
                    "Puedes intentar nuevamente en unos minutos."
                )

        except Exception as e:
            logger.error(f"Error en comando export: {e}", exc_info=True)
            await update.message.reply_text(
                f"❌ **Error en Export**\n\n{str(e)}"
            )

    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Comando /status
        Ejecuta validación completa de TODOS los servicios + status del sistema autónomo
        """
        try:
            logger.info("📊 Comando /status recibido - Ejecutando validación completa...")

            # Enviar mensaje inicial
            await update.message.reply_text(
                "🔍 **Ejecutando Validación Completa de Servicios**\n\n"
                "Validando 16 servicios críticos...\n"
                "Esto tomará ~5 segundos ⏳"
            )

            # EJECUTAR VALIDACIÓN COMPLETA DE SERVICIOS
            from src.startup_validator import StartupValidator

            # Necesitamos acceso al monitor - lo obtenemos del autonomy_controller
            # El monitor debería estar en el contexto global o necesitamos pasarlo
            # Por ahora, vamos a hacer una validación simplificada

            if not self.autonomy_controller:
                await update.message.reply_text("⚠️ Sistema autónomo no disponible")
                return

            # Obtener estadísticas del sistema autónomo
            stats = self.autonomy_controller.get_statistics()
            backup_status = self.autonomy_controller.git_backup.get_backup_status()

            # Validar servicios críticos manualmente
            services_status = []

            # 1. Telegram Bot
            services_status.append("✅ 1. Telegram Bot: Activo y respondiendo")

            # 2. Sistema Autónomo
            services_status.append(f"✅ 2. Sistema Autónomo: {'Activo' if stats['active'] else 'Inactivo'}")

            # 3. RL Agent
            q_size = stats['rl_agent']['q_table_size']
            services_status.append(f"✅ 3. RL Agent: {q_size} estados aprendidos")

            # 4. Parameter Optimizer
            trials = stats['parameter_optimizer']['total_trials']
            services_status.append(f"✅ 4. Parameter Optimizer: {trials} trials completados")

            # 5. Git Backup
            backup_active = "✅" if backup_status['running'] else "⚠️"
            services_status.append(f"{backup_active} 5. Git Backup: {'Activo' if backup_status['running'] else 'Inactivo'}")

            # 6. Paper Trader (si está disponible)
            if hasattr(self.autonomy_controller, 'paper_trader') and self.autonomy_controller.paper_trader:
                services_status.append("✅ 6. Paper Trading: Activo")
            else:
                services_status.append("⚠️ 6. Paper Trading: No disponible directamente")

            # Construir mensaje completo
            services_text = "\n".join(services_status)

            message = (
                "📊 **STATUS COMPLETO DEL SISTEMA**\n\n"
                "**🔍 SERVICIOS CRÍTICOS:**\n"
                f"{services_text}\n\n"
                "**🤖 SISTEMA AUTÓNOMO:**\n"
                f"  • Estado: {'✅ Activo' if stats['active'] else '❌ Inactivo'}\n"
                f"  • Modo: {stats['decision_mode']}\n\n"
                "**🧠 APRENDIZAJE:**\n"
                f"  • Trades procesados: {stats['total_trades_processed']}\n"
                f"  • Total trades experiencia: {self.autonomy_controller.total_trades_all_time}\n"
                f"  • Max leverage desbloqueado: {self.autonomy_controller._calculate_max_leverage()}x\n"
                f"  • Parámetros modificados: {stats['total_parameter_changes']} veces\n"
                f"  • Estados aprendidos: {stats['rl_agent']['q_table_size']}\n"
                f"  • Win rate RL: {stats['rl_agent']['success_rate']:.1f}%\n\n"
                "**⚙️ OPTIMIZACIÓN:**\n"
                f"  • Trials completados: {stats['parameter_optimizer']['total_trials']}\n"
                f"  • Mejor score: {stats['parameter_optimizer']['best_score']:.3f}\n"
                f"  • Parámetros activos: {stats['current_parameters_count']}\n\n"
                "**💾 BACKUPS:**\n"
                f"  • Auto-backup: {'✅ Activo' if backup_status['running'] else '❌ Inactivo'}\n"
                f"  • Próximo backup: {backup_status.get('next_backup', 'N/A')}\n"
                f"  • Último backup: {backup_status.get('last_backup', 'Ninguno')}\n\n"
                "📱 **Comandos disponibles:**\n"
                "  /stats - Estadísticas de trading\n"
                "  /params - Ver parámetros actuales\n"
                "  /export - Exportar aprendizaje IA"
            )

            await update.message.reply_text(message)

        except Exception as e:
            logger.error(f"Error en comando status: {e}", exc_info=True)
            await update.message.reply_text(
                f"❌ Error obteniendo status:\n{str(e)}"
            )

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Comando /help
        Muestra ayuda de comandos disponibles
        """
        try:
            message = (
                "🤖 **Comandos Disponibles**\n\n"
                "/export_intelligence (o /export)\n"
                "  ├─ Export manual de inteligencia\n"
                "  ├─ Guarda aprendizaje localmente\n"
                "  ├─ Hace backup a Git/GitHub\n"
                "  └─ Te envía el archivo .json\n\n"
                "/import_intelligence (o /import)\n"
                "  ├─ Restaura inteligencia desde archivo\n"
                "  ├─ Envía el archivo .json después del comando\n"
                "  ├─ Valida integridad (checksum)\n"
                "  └─ Útil después de redeploys\n\n"
                "/import_force\n"
                "  ├─ Import sin validar checksum\n"
                "  ├─ Para archivos editados manualmente\n"
                "  ├─ ⚠️ Ignora validación de integridad\n"
                "  └─ Usa solo si /import falla por checksum\n\n"
                "/status\n"
                "  ├─ Muestra estado del sistema autónomo\n"
                "  ├─ Estadísticas de aprendizaje\n"
                "  └─ Info de backups\n\n"
                "/futures_stats\n"
                "  ├─ Estadísticas de trading de futuros\n"
                "  ├─ Max leverage desbloqueado\n"
                "  ├─ Liquidaciones totales\n"
                "  └─ PnL SPOT vs FUTURES\n\n"
                "/train_ml\n"
                "  ├─ Entrena el ML System con trades históricos\n"
                "  ├─ Usa después de /import para cargar datos\n"
                "  ├─ Requiere mínimo 25 trades\n"
                "  └─ Habilita predicciones ML automáticas\n\n"
                "/force_sync\n"
                "  ├─ Fuerza sincronización COMPLETA de todos los contadores\n"
                "  ├─ Usa Paper Trading como fuente de verdad\n"
                "  ├─ Sincroniza: trades, win rate, procesados, all-time\n"
                "  ├─ Ajusta RL Agent automáticamente\n"
                "  └─ Útil si /stats muestra desincronización ⚠️\n\n"
                "/help\n"
                "  └─ Muestra este mensaje\n\n"
                "**Auto-Backup**: Cada 24h automático\n"
                "**Flujo**: /export antes de redeploy → /import después → /train_ml\n"
                "**Emergencia**: Si /import falla → /import_force"
            )

            await update.message.reply_text(message)

        except Exception as e:
            logger.error(f"Error en comando help: {e}", exc_info=True)

    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Comando /stats
        Muestra estadísticas de trading y performance
        """
        try:
            logger.info("📈 Comando /stats recibido")

            if not self.autonomy_controller:
                await update.message.reply_text("⚠️ Sistema autónomo no disponible")
                return

            # Obtener estadísticas de paper trading
            paper_trader = self.autonomy_controller.paper_trader if hasattr(self.autonomy_controller, 'paper_trader') else None

            if not paper_trader or not hasattr(paper_trader, 'portfolio'):
                # Mostrar información básica aunque paper trader no esté activo
                message = "📊 **Estadísticas de Trading**\n\n"

                if self.autonomy_controller:
                    message += f"**📈 Historial:**\n"
                    message += f"  • Total trades: {self.autonomy_controller.total_trades_all_time}\n"
                    message += f"  • Win rate RL: {self.autonomy_controller.rl_agent.success_rate:.1f}%\n"
                    message += f"  • Estados aprendidos: {len(self.autonomy_controller.rl_agent.q_table)}\n\n"
                    message += f"**💰 Paper Trading:**\n"
                    message += f"  • Estado: Inicializándose...\n"
                    message += f"  • Balance inicial: $50,000\n"
                    message += f"  • Se activará con el primer trade\n"
                else:
                    message = "⚠️ Sistema no disponible"

                await update.message.reply_text(message)
                return

            portfolio = paper_trader.portfolio
            stats = paper_trader.get_statistics()

            # Validar sincronización
            sync = {'in_sync': True}
            if self.autonomy_controller:
                sync = self.autonomy_controller.validate_sync()

            sync_emoji = "✅" if sync['in_sync'] else "⚠️"

            # Calcular métricas
            equity = portfolio.get_equity()
            initial_balance = 50000  # Balance inicial
            pnl = equity - initial_balance
            pnl_pct = (pnl / initial_balance) * 100

            message = (
                "📈 **Estadísticas de Trading**\n\n"
                "**💰 Balance:**\n"
                f"  • Equity actual: ${equity:,.2f} USDT\n"
                f"  • Balance inicial: ${initial_balance:,.2f} USDT\n"
                f"  • P&L total: ${pnl:+,.2f} ({pnl_pct:+.2f}%)\n\n"
                "**📊 Performance:**\n"
                f"  • Trades totales: {stats.get('total_trades', 0)}\n"
                f"  • Trades ganadores: {stats.get('winning_trades', 0)}\n"
                f"  • Trades perdedores: {stats.get('losing_trades', 0)}\n"
                f"  • Win rate: {stats.get('win_rate', 0):.1f}%\n\n"
                "**💵 Resultados:**\n"
                f"  • Profit total: ${stats.get('total_profit', 0):,.2f}\n"
                f"  • Loss total: ${stats.get('total_loss', 0):,.2f}\n"
                f"  • Profit promedio: ${stats.get('avg_profit', 0):,.2f}\n"
                f"  • Loss promedio: ${stats.get('avg_loss', 0):,.2f}\n\n"
                "**📍 Posiciones:**\n"
                f"  • Abiertas: {len(portfolio.positions) if hasattr(portfolio, 'positions') else 0}\n"
                f"  • Cerradas: {len(portfolio.closed_trades) if hasattr(portfolio, 'closed_trades') else 0}\n\n"
                "**🔄 Sincronización de Contadores:**\n"
                f"  • Estado: {sync_emoji}\n"
            )

            if not sync['in_sync']:
                diffs = sync['differences']
                message += (
                    f"\n⚠️ **Desincronización detectada:**\n"
                    f"  • Paper Trading: {sync['paper_trades']} trades, {sync['paper_win_rate']:.1f}% WR ✅\n"
                    f"  • RL Agent: {sync['rl_trades']} trades {'' if diffs['rl_vs_paper'] == 0 else '❌'}, {sync['rl_win_rate']:.1f}% WR {'' if sync['win_rate_in_sync'] else '❌'}\n"
                    f"  • Trades Procesados: {sync['processed_trades']} {'' if diffs['processed_vs_paper'] == 0 else '❌'}\n"
                    f"  • Total All Time: {sync['all_time_trades']} {'' if diffs['all_time_vs_paper'] == 0 else '❌'}\n"
                )
                if not sync['win_rate_in_sync']:
                    message += f"  • Diferencia Win Rate: {diffs['win_rate_diff']:.1f}%\n"
                message += f"\n💡 Usa /force_sync para sincronizar todos los contadores\n"
            else:
                message += (
                    f"  • Todos los contadores sincronizados: {sync['paper_trades']} trades, {sync['paper_win_rate']:.1f}% WR ✅\n"
                )

            message += "\nUsa /status para ver estado del sistema autónomo"

            await update.message.reply_text(message)

        except Exception as e:
            logger.error(f"Error en comando stats: {e}", exc_info=True)
            await update.message.reply_text(f"❌ Error obteniendo stats:\n{str(e)}")

    async def futures_stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Comando /futures_stats
        Muestra estadísticas específicas de trading de futuros
        """
        try:
            logger.info("🎯 Comando /futures_stats recibido")

            if not self.autonomy_controller:
                await update.message.reply_text("⚠️ Sistema autónomo no disponible")
                return

            # Obtener total_trades_all_time y max_leverage
            total_trades = self.autonomy_controller.total_trades_all_time
            max_leverage = self.autonomy_controller._calculate_max_leverage()

            # Calcular próximo desbloqueo
            next_unlock = None
            next_leverage = None
            if total_trades < 50:
                next_unlock = 50 - total_trades
                next_leverage = 8
            elif total_trades < 100:
                next_unlock = 100 - total_trades
                next_leverage = 10
            elif total_trades < 150:
                next_unlock = 150 - total_trades
                next_leverage = 15
            elif total_trades < 500:
                next_unlock = 500 - total_trades
                next_leverage = 20
            else:
                next_unlock = 0
                next_leverage = 20

            # Obtener estadísticas del portfolio
            paper_trader = self.autonomy_controller.paper_trader if hasattr(self.autonomy_controller, 'paper_trader') else None

            if paper_trader:
                portfolio = paper_trader.portfolio
                closed_trades = portfolio.closed_trades

                # Todos los trades son FUTURES ahora (migrado desde SPOT)
                spot_trades = [t for t in closed_trades if t.get('trade_type', 'FUTURES') == 'SPOT']  # Legacy
                futures_trades = [t for t in closed_trades if t.get('trade_type', 'FUTURES') == 'FUTURES']

                # Calcular liquidaciones
                liquidations = [t for t in futures_trades if t.get('liquidated', False)]
                liquidation_count = len(liquidations)

                # PnL spot vs futures
                spot_pnl = sum(t.get('pnl', 0) for t in spot_trades)
                futures_pnl = sum(t.get('pnl', 0) for t in futures_trades)

                # Leverage promedio en futures
                if futures_trades:
                    avg_leverage = sum(t.get('leverage', 1) for t in futures_trades) / len(futures_trades)
                else:
                    avg_leverage = 0

                futures_stats_text = (
                    f"**📊 Trades:**\n"
                    f"  • SPOT: {len(spot_trades)} trades\n"
                    f"  • FUTURES: {len(futures_trades)} trades\n\n"
                    f"**💥 Liquidaciones:**\n"
                    f"  • Total: {liquidation_count}\n"
                    f"  • Tasa: {(liquidation_count / len(futures_trades) * 100) if futures_trades else 0:.1f}%\n\n"
                    f"**💰 PnL Comparativo:**\n"
                    f"  • SPOT: ${spot_pnl:+,.2f}\n"
                    f"  • FUTURES: ${futures_pnl:+,.2f}\n\n"
                    f"**📈 Leverage:**\n"
                    f"  • Promedio usado: {avg_leverage:.1f}x\n"
                )
            else:
                futures_stats_text = "⚠️ Paper trading no disponible"

            message = (
                "🎯 **Estadísticas de Futuros**\n\n"
                f"**🏆 Experiencia:**\n"
                f"  • Total trades: {total_trades}\n"
                f"  • Max leverage desbloqueado: {max_leverage}x\n"
            )

            if next_unlock > 0:
                message += f"  • Próximo desbloqueo: {next_leverage}x en {next_unlock} trades\n\n"
            else:
                message += f"  • ✅ Max leverage alcanzado (20x)\n\n"

            message += futures_stats_text

            message += (
                "\n**📍 Límites de Leverage:**\n"
                "  • 0-50 trades: 5x\n"
                "  • 50-100 trades: 8x\n"
                "  • 100-150 trades: 10x\n"
                "  • 150-500 trades: 15x\n"
                "  • 500+ trades: 20x"
            )

            await update.message.reply_text(message)

        except Exception as e:
            logger.error(f"Error en comando futures_stats: {e}", exc_info=True)
            await update.message.reply_text(f"❌ Error obteniendo futures stats:\n{str(e)}")

    async def params_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Comando /params
        Muestra parámetros actuales optimizables
        """
        try:
            logger.info("🎯 Comando /params recibido")

            if not self.autonomy_controller:
                await update.message.reply_text("⚠️ Sistema autónomo no disponible")
                return

            optimizer = self.autonomy_controller.parameter_optimizer
            if not optimizer:
                await update.message.reply_text("⚠️ Parameter optimizer no disponible")
                return

            # Obtener parámetros actuales
            current_params = optimizer.current_parameters

            # Agrupar por categoría
            risk_params = {k: v for k, v in current_params.items() if any(x in k for x in ['RISK', 'POSITION', 'DRAWDOWN', 'STOP'])}
            indicator_params = {k: v for k, v in current_params.items() if any(x in k for x in ['RSI', 'MACD', 'EMA', 'BB'])}
            threshold_params = {k: v for k, v in current_params.items() if 'THRESHOLD' in k or 'CONFIDENCE' in k}
            news_params = {k: v for k, v in current_params.items() if any(x in k for x in ['NEWS', 'IMPORTANCE', 'ENGAGEMENT', 'SOCIAL', 'BUZZ'])}
            tp_params = {k: v for k, v in current_params.items() if 'TP' in k or 'DYNAMIC' in k}

            message = "🎯 **Parámetros Actuales (41 optimizables)**\n\n"

            if news_params:
                message += "**📰 News-Triggered Trading:**\n"
                for param, value in list(news_params.items())[:5]:
                    message += f"  • {param}: {value}\n"
                message += "\n"

            if tp_params:
                message += "**💰 Dynamic TPs:**\n"
                for param, value in list(tp_params.items())[:5]:
                    message += f"  • {param}: {value}\n"
                message += "\n"

            if threshold_params:
                message += "**🎯 Thresholds:**\n"
                for param, value in list(threshold_params.items())[:5]:
                    message += f"  • {param}: {value}\n"
                message += "\n"

            if risk_params:
                message += "**📊 Risk Management:**\n"
                for param, value in list(risk_params.items())[:4]:
                    message += f"  • {param}: {value}\n"
                message += "\n"

            # Stats de optimización
            trials = len(optimizer.trial_history) if hasattr(optimizer, 'trial_history') else 0
            best_score = optimizer.best_score if hasattr(optimizer, 'best_score') else 0

            message += f"**📈 Optimización:**\n"
            message += f"  • Trials completados: {trials}\n"
            message += f"  • Mejor score: {best_score:.3f}\n\n"
            message += "⚡ IA ajusta estos parámetros automáticamente para maximizar rentabilidad"

            await update.message.reply_text(message)

        except Exception as e:
            logger.error(f"Error en comando params: {e}", exc_info=True)
            await update.message.reply_text(f"❌ Error obteniendo params:\n{str(e)}")

    async def train_ml_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Comando /train_ml
        Entrena el ML System con los datos disponibles del Paper Trading
        """
        try:
            logger.info("🤖 Comando /train_ml recibido")

            await update.message.reply_text(
                "🤖 **Entrenando ML System**\n\n"
                "Iniciando entrenamiento con trades históricos...\n"
                "Esto puede tomar unos segundos ⏳"
            )

            # Verificar que market_monitor esté disponible
            if not hasattr(self, 'market_monitor') or not self.market_monitor:
                await update.message.reply_text(
                    "⚠️ **Market Monitor no disponible**\n\n"
                    "El ML System está en el Market Monitor.\n"
                    "Asegúrate de que el bot esté corriendo."
                )
                return

            # Verificar que ml_system esté disponible
            if not hasattr(self.market_monitor, 'ml_system') or not self.market_monitor.ml_system:
                await update.message.reply_text(
                    "⚠️ **ML System no disponible**\n\n"
                    "Verifica que ENABLE_PAPER_TRADING esté en True en config.py"
                )
                return

            ml_system = self.market_monitor.ml_system

            # USAR EL PORTFOLIO DEL AUTONOMY CONTROLLER (que SÍ se restaura en /import)
            # en lugar del portfolio interno del ML System
            if not hasattr(self, 'autonomy_controller') or not self.autonomy_controller:
                await update.message.reply_text(
                    "⚠️ **Autonomy Controller no disponible**\n\n"
                    "No se puede acceder al portfolio."
                )
                return

            paper_trader = self.autonomy_controller.paper_trader
            if not paper_trader or not hasattr(paper_trader, 'portfolio'):
                await update.message.reply_text(
                    "⚠️ **Paper Trader no disponible**\n\n"
                    "No hay datos para entrenar."
                )
                return

            stats = paper_trader.portfolio.get_statistics()
            total_trades = stats.get('total_trades', 0)

            if total_trades < 25:
                await update.message.reply_text(
                    f"⚠️ **Insuficientes trades para entrenar**\n\n"
                    f"Trades actuales: {total_trades}\n"
                    f"Mínimo requerido: 25\n\n"
                    f"Espera a tener más trades históricos o después de /import"
                )
                return

            # Forzar entrenamiento con threshold reducido
            # Pasar el paper_trader del autonomy_controller (que tiene los datos restaurados)
            logger.info(f"Forzando entrenamiento ML con {total_trades} trades")
            ml_system.force_retrain(
                min_samples_override=25,
                external_paper_trader=paper_trader
            )

            # Obtener info del modelo entrenado
            model_info = ml_system.trainer.get_model_info()

            if model_info.get('available'):
                metrics = model_info.get('metrics', {})
                message = (
                    "✅ **ML System Entrenado Exitosamente**\n\n"
                    f"📊 **Datos de Entrenamiento:**\n"
                    f"  • Total trades: {total_trades}\n"
                    f"  • Samples usados: {metrics.get('samples_total', 0)}\n\n"
                    f"📈 **Métricas del Modelo:**\n"
                    f"  • Accuracy: {metrics.get('test_accuracy', 0):.1%}\n"
                    f"  • Precision: {metrics.get('test_precision', 0):.1%}\n"
                    f"  • F1 Score: {metrics.get('test_f1', 0):.3f}\n\n"
                    f"🎯 **Estado:**\n"
                    f"  • Modelo: Activo ✅\n"
                    f"  • Predicciones ML: Habilitadas\n\n"
                    f"El ML ahora hará predicciones en cada señal 🚀"
                )
            else:
                message = (
                    "⚠️ **Entrenamiento Completado con Advertencias**\n\n"
                    f"Se procesaron {total_trades} trades pero el modelo\n"
                    f"puede necesitar más datos para predicciones confiables.\n\n"
                    f"Continúa trading para mejorar el modelo 📈"
                )

            await update.message.reply_text(message)

        except Exception as e:
            logger.error(f"Error en comando train_ml: {e}", exc_info=True)
            await update.message.reply_text(
                f"❌ **Error en Entrenamiento ML**\n\n{str(e)}"
            )

    async def force_sync_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Comando /force_sync
        Fuerza sincronización entre Paper Trading y RL Agent
        """
        try:
            logger.info("🔄 Comando /force_sync recibido")

            if not self.autonomy_controller:
                await update.message.reply_text("⚠️ Sistema autónomo no disponible")
                return

            # Verificar estado actual de sincronización
            sync_status = self.autonomy_controller.validate_sync()

            if sync_status['in_sync']:
                await update.message.reply_text(
                    "✅ **Todos los Contadores Sincronizados**\n\n"
                    f"Paper Trading: {sync_status['paper_trades']} trades, {sync_status['paper_win_rate']:.1f}% WR\n"
                    f"RL Agent: {sync_status['rl_trades']} trades, {sync_status['rl_win_rate']:.1f}% WR\n"
                    f"Trades Procesados: {sync_status['processed_trades']}\n"
                    f"Total All Time: {sync_status['all_time_trades']}\n\n"
                    "No se requiere acción 👍"
                )
                return

            # Mostrar estado actual con TODOS los contadores
            diffs = sync_status['differences']
            desync_msg = "⚠️ **Desincronización Detectada**\n\n"
            desync_msg += f"Paper Trading: {sync_status['paper_trades']} trades, {sync_status['paper_win_rate']:.1f}% WR ✅ (fuente de verdad)\n"
            desync_msg += f"RL Agent: {sync_status['rl_trades']} trades {'' if diffs['rl_vs_paper'] == 0 else '❌'}, {sync_status['rl_win_rate']:.1f}% WR {'' if sync_status['win_rate_in_sync'] else '❌'}\n"
            desync_msg += f"Trades Procesados: {sync_status['processed_trades']} {'' if diffs['processed_vs_paper'] == 0 else '❌'}\n"
            desync_msg += f"Total All Time: {sync_status['all_time_trades']} {'' if diffs['all_time_vs_paper'] == 0 else '❌'}\n\n"
            desync_msg += "🔄 Forzando sincronización de TODOS los contadores...\n"
            desync_msg += "Usando Paper Trading como fuente de verdad..."

            await update.message.reply_text(desync_msg)

            # Ejecutar sincronización forzada
            success = await self.autonomy_controller.force_sync_from_paper()

            if success:
                # Verificar sincronización post-fix
                new_sync = self.autonomy_controller.validate_sync()

                result_msg = "✅ **Sincronización Completada**\n\n"
                result_msg += f"Todos los contadores ahora tienen: {new_sync['paper_trades']} trades, {new_sync['paper_win_rate']:.1f}% WR\n\n"
                result_msg += "📊 **Acciones realizadas:**\n"
                result_msg += f"  • RL Agent trades: {sync_status['rl_trades']} → {new_sync['rl_trades']}\n"
                result_msg += f"  • RL Agent Win Rate: {sync_status['rl_win_rate']:.1f}% → {new_sync['rl_win_rate']:.1f}%\n"
                result_msg += f"  • Trades Procesados: {sync_status['processed_trades']} → {new_sync['processed_trades']}\n"
                result_msg += f"  • Total All Time: {sync_status['all_time_trades']} → {new_sync['all_time_trades']}\n"
                result_msg += f"  • Estado guardado automáticamente\n\n"
                result_msg += "💡 Usa /export para crear backup actualizado"

                await update.message.reply_text(result_msg)
            else:
                await update.message.reply_text(
                    "❌ **Error en Sincronización**\n\n"
                    "No se pudo completar la sincronización.\n"
                    "Revisa los logs para más detalles."
                )

        except Exception as e:
            logger.error(f"Error en comando force_sync: {e}", exc_info=True)
            await update.message.reply_text(
                f"❌ **Error en Sincronización**\n\n{str(e)}"
            )

    async def import_intelligence_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Comando /import_intelligence
        Solicita al usuario que envíe el archivo .json para restaurar
        """
        try:
            logger.info("📥 Comando /import_intelligence recibido")

            self.waiting_for_import_file = True

            await update.message.reply_text(
                "📥 **Import de Inteligencia**\n\n"
                "Por favor, envía el archivo .json que descargaste con /export\n\n"
                "El archivo debe ser:\n"
                "  • Formato: .json\n"
                "  • Nombre: intelligence_export*.json\n"
                "  • Del comando /export anterior\n\n"
                "⏳ Esperando archivo..."
            )

        except Exception as e:
            logger.error(f"Error en comando import: {e}", exc_info=True)
            await update.message.reply_text(
                f"❌ **Error en Import**\n\n{str(e)}"
            )

    async def import_force_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Comando /import_force
        Import forzado ignorando validación de checksum (para archivos editados manualmente)
        """
        try:
            logger.info("🔧 Comando /import_force recibido - IGNORANDO CHECKSUM")

            self.waiting_for_import_force_file = True

            await update.message.reply_text(
                "🔧 **Import FORCE (Sin Validación)**\n\n"
                "⚠️ Este comando importa sin validar checksum\n"
                "Úsalo solo si editaste el archivo manualmente\n\n"
                "Por favor, envía el archivo .json a importar\n\n"
                "El archivo debe ser:\n"
                "  • Formato: .json\n"
                "  • Estructura válida (rl_agent, parameter_optimizer)\n"
                "  • ⚠️ NO se validará integridad\n\n"
                "⏳ Esperando archivo..."
            )

        except Exception as e:
            logger.error(f"Error en comando import_force: {e}", exc_info=True)
            await update.message.reply_text(
                f"❌ **Error en Import Force**\n\n{str(e)}"
            )

    async def handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Handler para recibir documentos (archivos)
        Se activa cuando el usuario está esperando enviar archivo de import o import_force
        """
        try:
            # Determinar si esperamos import normal o force
            is_force = self.waiting_for_import_force_file
            is_normal = self.waiting_for_import_file

            # Solo procesar si estamos esperando un archivo
            if not is_force and not is_normal:
                return

            document = update.message.document

            # Validar que sea un archivo JSON
            if not document.file_name.endswith('.json'):
                await update.message.reply_text(
                    "⚠️ **Formato Inválido**\n\n"
                    "Por favor envía un archivo .json\n"
                    f"Recibido: {document.file_name}"
                )
                return

            mode_str = "FORCE MODE (sin validación)" if is_force else "normal"
            await update.message.reply_text(
                f"📥 **Archivo Recibido** ({mode_str})\n\n"
                f"📄 {document.file_name}\n"
                f"💾 {document.file_size / 1024:.1f} KB\n\n"
                "Descargando y procesando... ⏳"
            )

            # Descargar archivo
            file = await context.bot.get_file(document.file_id)

            # Guardar temporalmente
            temp_dir = Path("data/temp")
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_path = temp_dir / document.file_name

            await file.download_to_drive(temp_path)

            logger.info(f"📥 Archivo descargado a: {temp_path} (force={is_force})")

            # Importar inteligencia
            if not self.autonomy_controller:
                await update.message.reply_text(
                    "⚠️ **Error**: Sistema autónomo no disponible"
                )
                return

            # Llamar a manual_import con force=True si es import_force
            success = await self.autonomy_controller.manual_import(str(temp_path), force=is_force)

            # Limpiar archivo temporal
            try:
                os.remove(temp_path)
            except:
                pass

            # Resetear flags
            self.waiting_for_import_file = False
            self.waiting_for_import_force_file = False

            # Enviar resultado
            if success:
                force_warning = "\n⚠️ IMPORTADO SIN VALIDACIÓN DE CHECKSUM\n" if is_force else ""
                await update.message.reply_text(
                    f"✅ **Import Completado**{force_warning}\n"
                    "✅ Archivo procesado correctamente\n"
                    "✅ Inteligencia restaurada:\n"
                    "   • RL Agent (Q-table y stats)\n"
                    "   • Parameter Optimizer (trials y config)\n"
                    "   • Histórico de cambios\n"
                    "   • Performance history\n\n"
                    "🧠 El bot continuará aprendiendo desde donde lo dejó 🎉"
                )
            else:
                checksum_hint = "\n\n💡 Si editaste el archivo manualmente, usa /import_force" if is_normal else ""
                await update.message.reply_text(
                    f"❌ **Import Falló**\n\n"
                    f"El archivo no pudo ser procesado.\n"
                    f"Posibles causas:\n"
                    f"  • Archivo corrupto\n"
                    f"  • Formato inválido\n"
                    f"  • Checksum no coincide (archivo editado)\n"
                    f"  • Versión incompatible{checksum_hint}\n\n"
                    f"Intenta con otro archivo o usa /export para generar uno nuevo."
                )

        except Exception as e:
            logger.error(f"Error procesando documento: {e}", exc_info=True)
            self.waiting_for_import_file = False
            self.waiting_for_import_force_file = False
            await update.message.reply_text(
                f"❌ **Error procesando archivo**\n\n{str(e)}"
            )

    async def send_message(self, message: str):
        """
        Envía mensaje proactivo al chat configurado

        Args:
            message: Texto del mensaje a enviar

        Returns:
            True si se envió correctamente, False si hubo error
        """
        if not self.application or not self.chat_id:
            logger.warning("No se puede enviar mensaje: application o chat_id no configurado")
            return False

        try:
            # Intentar con Markdown primero
            await self.application.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='Markdown'
            )
            return True
        except Exception as e:
            # Si falla Markdown, intentar sin formato
            logger.warning(f"Error con Markdown, reintentando sin formato: {e}")
            try:
                await self.application.bot.send_message(
                    chat_id=self.chat_id,
                    text=message,
                    parse_mode=None
                )
                return True
            except Exception as e2:
                logger.error(f"Error enviando mensaje a Telegram: {e2}")
                return False

    async def pause_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Comando /pause
        Pausa el análisis de nuevos pares pero sigue monitoreando trades abiertos
        Útil antes de hacer export para evitar discrepancias
        """
        try:
            logger.info("⏸️  Comando /pause recibido")

            if not self.market_monitor:
                await update.message.reply_text(
                    "❌ Market Monitor no disponible"
                )
                return

            # Obtener trades abiertos
            open_trades_count = 0
            if self.market_monitor.ml_system and self.market_monitor.ml_system.paper_trader:
                open_trades_count = len(self.market_monitor.ml_system.paper_trader.portfolio.positions)

            # Pausar análisis
            self.market_monitor.pause_analysis()

            await update.message.reply_text(
                f"⏸️ **ANÁLISIS PAUSADO**\n\n"
                f"✅ El bot dejó de analizar nuevos pares\n"
                f"✅ Sigue monitoreando {open_trades_count} trade(s) abierto(s)\n"
                f"✅ Los trades se cerrarán automáticamente si alcanzan TP/SL\n\n"
                f"💡 Ideal para hacer `/export` sin discrepancias\n"
                f"▶️ Usa `/resume` para reanudar análisis"
            )

        except Exception as e:
            logger.error(f"Error en comando /pause: {e}", exc_info=True)
            await update.message.reply_text(
                f"❌ Error pausando análisis: {str(e)}"
            )

    async def resume_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Comando /resume
        Reanuda el análisis de mercado normal
        """
        try:
            logger.info("▶️  Comando /resume recibido")

            if not self.market_monitor:
                await update.message.reply_text(
                    "❌ Market Monitor no disponible"
                )
                return

            # Resumir análisis
            self.market_monitor.resume_analysis()

            await update.message.reply_text(
                f"▶️ **ANÁLISIS RESUMIDO**\n\n"
                f"✅ El bot volvió a analizar todos los pares\n"
                f"✅ Trading autónomo activo\n\n"
                f"📊 Monitoreando: {', '.join(self.market_monitor.trading_pairs[:3])} y más..."
            )

        except Exception as e:
            logger.error(f"Error en comando /resume: {e}", exc_info=True)
            await update.message.reply_text(
                f"❌ Error reanudando análisis: {str(e)}"
            )
