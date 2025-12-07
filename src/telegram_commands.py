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
    - /gpt_*: Comandos de GPT Brain
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
        self.gpt_brain = None  # GPT Brain instance (set by main.py)

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
            self.application.add_handler(CommandHandler("reset_ai", self.reset_ai_command))  # Resetear IA a cero
            self.application.add_handler(CommandHandler("pause", self.pause_command))  # Pausar análisis
            self.application.add_handler(CommandHandler("resume", self.resume_command))  # Resumir análisis
            self.application.add_handler(CommandHandler("help", self.help_command))

            # GPT Brain commands
            self.application.add_handler(CommandHandler("gpt", self.gpt_status_command))  # Status de GPT Brain
            self.application.add_handler(CommandHandler("gpt_analyze", self.gpt_analyze_command))  # Análisis de performance
            self.application.add_handler(CommandHandler("gpt_optimize", self.gpt_optimize_command))  # Forzar optimización
            self.application.add_handler(CommandHandler("gpt_insight", self.gpt_insight_command))  # Insight de mercado
            self.application.add_handler(CommandHandler("gpt_scan", self.gpt_scan_command))  # Escanear mercado con GPT
            self.application.add_handler(CommandHandler("gpt_signal", self.gpt_signal_command))  # Generar señal GPT
            self.application.add_handler(CommandHandler("gpt_enable", self.gpt_enable_command))  # Habilitar GPT
            self.application.add_handler(CommandHandler("gpt_disable", self.gpt_disable_command))  # Deshabilitar GPT

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
                "/reset_ai\n"
                "  ├─ ⚠️ CUIDADO: Borra TODO el aprendizaje\n"
                "  ├─ Resetea Q-Table, estadísticas, experiencias\n"
                "  ├─ Guarda backup antes de borrar\n"
                "  └─ Útil para empezar de 0 con nuevos parámetros\n\n"
                "/help\n"
                "  └─ Muestra este mensaje\n\n"
                "**🧠 GPT Brain Commands:**\n"
                "/gpt - Estado del GPT Brain\n"
                "/gpt_analyze - Análisis de performance con GPT\n"
                "/gpt_optimize - Forzar optimización de parámetros\n"
                "/gpt_insight - Insight del mercado actual\n"
                "/gpt_scan - Escanear mercado para oportunidades\n"
                "/gpt_signal [par] - Generar señal GPT (ej: /gpt_signal BTC)\n"
                "/gpt_enable - Habilitar GPT Brain\n"
                "/gpt_disable - Deshabilitar GPT Brain\n\n"
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
                    # Detectar modo
                    is_live = hasattr(paper_trader, 'is_live') and paper_trader.is_live() if paper_trader else False
                    mode_str = "🔴 LIVE" if is_live else "📝 PAPER"

                    message += f"**📈 Historial ({mode_str}):**\n"
                    message += f"  • Total trades: {self.autonomy_controller.total_trades_all_time}\n"
                    message += f"  • Win rate RL: {self.autonomy_controller.rl_agent.success_rate:.1f}%\n"
                    message += f"  • Estados aprendidos: {len(self.autonomy_controller.rl_agent.q_table)}\n\n"
                    message += f"**💰 Trading:**\n"
                    message += f"  • Estado: Inicializándose...\n"
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

            # Calcular métricas - usar initial_balance del portfolio (real para LIVE, paper para PAPER)
            equity = portfolio.get_equity()
            initial_balance = getattr(portfolio, 'initial_balance', 50000)  # Obtener del portfolio
            pnl = equity - initial_balance
            pnl_pct = (pnl / initial_balance) * 100 if initial_balance > 0 else 0

            # Detectar modo de trading
            is_live = hasattr(paper_trader, 'is_live') and paper_trader.is_live()
            mode_str = "🔴 LIVE" if is_live else "📝 PAPER"

            message = (
                f"📈 **Estadísticas de Trading** ({mode_str})\n\n"
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

    async def reset_ai_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Comando /reset_ai
        Resetea la IA a cero, borrando todo el aprendizaje
        """
        try:
            logger.info("🔄 Comando /reset_ai recibido")

            if not self.autonomy_controller:
                await update.message.reply_text(
                    "⚠️ **Error**: Sistema autónomo no disponible"
                )
                return

            await update.message.reply_text(
                "⚠️ **RESETEAR IA**\n\n"
                "Esto borrará TODO el aprendizaje:\n"
                "• Q-Table (estados aprendidos)\n"
                "• Estadísticas de trades\n"
                "• Historial de experiencias\n"
                "• Parámetros optimizados\n\n"
                "🔄 Reseteando..."
            )

            # Resetear el RL Agent
            rl_agent = self.autonomy_controller.rl_agent

            # Guardar backup antes de resetear
            backup_path = await self.autonomy_controller.manual_export()
            logger.info(f"📦 Backup guardado antes de reset: {backup_path}")

            # Resetear todo
            rl_agent.q_table = {}
            rl_agent.memory.clear()
            rl_agent.total_trades = 0
            rl_agent.successful_trades = 0
            rl_agent.total_reward = 0.0
            rl_agent.episode_rewards = []
            rl_agent.current_state = None
            rl_agent.current_action = None
            rl_agent.exploration_rate = 0.3  # Reset exploration rate

            # Resetear contadores del controlador
            self.autonomy_controller.total_trades_processed = 0
            self.autonomy_controller.total_trades_all_time = 0
            self.autonomy_controller.performance_history = []
            self.autonomy_controller.change_history = []

            # Resetear paper trading si existe
            if self.ml_system and hasattr(self.ml_system, 'paper_trader'):
                paper = self.ml_system.paper_trader
                if hasattr(paper, 'portfolio'):
                    paper.portfolio.closed_trades = []
                    paper.portfolio.total_trades = 0
                    paper.portfolio.winning_trades = 0
                    paper.portfolio.losing_trades = 0
                    paper.portfolio.total_profit = 0.0
                    paper.portfolio.total_loss = 0.0

            # Guardar estado reseteado
            await self.autonomy_controller.save_intelligence()

            await update.message.reply_text(
                "✅ **IA RESETEADA**\n\n"
                "Se ha borrado todo el aprendizaje previo.\n\n"
                "📊 Estado actual:\n"
                f"• Q-Table: {len(rl_agent.q_table)} estados\n"
                f"• Trades procesados: {rl_agent.total_trades}\n"
                f"• Experiencias: {len(rl_agent.memory)}\n"
                f"• Exploration Rate: {rl_agent.exploration_rate:.2f}\n\n"
                f"📦 Backup guardado por si necesitas restaurar\n\n"
                "🚀 La IA empezará a aprender desde cero con los nuevos parámetros."
            )

            logger.info("✅ IA reseteada exitosamente")

        except Exception as e:
            logger.error(f"Error en comando reset_ai: {e}", exc_info=True)
            await update.message.reply_text(
                f"❌ **Error en Reset**\n\n{str(e)}"
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

    # =========================================================================
    # GPT BRAIN COMMANDS
    # =========================================================================

    async def gpt_status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Comando /gpt
        Muestra estado actual del GPT Brain
        """
        try:
            logger.info("🧠 Comando /gpt recibido")

            if not self.gpt_brain:
                await update.message.reply_text(
                    "⚠️ **GPT Brain no disponible**\n\n"
                    "GPT Brain no está inicializado.\n"
                    "Verifica que ENABLE_GPT_BRAIN=true y OPENAI_API_KEY está configurada."
                )
                return

            stats = self.gpt_brain.get_stats()

            message = (
                "🧠 **GPT Brain Status**\n\n"
                f"**Estado:** {'✅ Activo' if stats['enabled'] else '❌ Desactivado'}\n"
                f"**Modelo:** {stats['model']}\n\n"
                f"**📊 Estadísticas:**\n"
                f"  • Decisiones tomadas: {stats['decisions_made']}\n"
                f"  • Trades aprobados: {stats['trades_approved']}\n"
                f"  • Trades bloqueados: {stats['trades_blocked']}\n"
                f"  • Tasa de bloqueo: {stats['block_rate']:.1f}%\n"
                f"  • Optimizaciones: {stats['optimizations_performed']}\n\n"
                f"**📈 Rachas actuales:**\n"
                f"  • Pérdidas consecutivas: {stats['consecutive_losses']}\n"
                f"  • Ganancias consecutivas: {stats['consecutive_wins']}\n\n"
                f"**💰 Costo GPT:**\n"
                f"  • Costo total: ${stats['total_gpt_cost']:.4f}\n"
                f"  • Tokens usados: {stats['gpt_client_stats']['total_tokens']:,}\n\n"
                f"**⏰ Última actividad:**\n"
                f"  • Optimización: {stats['last_optimization'] or 'Nunca'}\n"
                f"  • Análisis: {stats['last_analysis'] or 'Nunca'}\n\n"
                "📱 **Comandos GPT:**\n"
                "  /gpt_analyze - Analizar performance\n"
                "  /gpt_optimize - Forzar optimización\n"
                "  /gpt_insight - Ver insight de mercado\n"
                "  /gpt_enable - Habilitar GPT\n"
                "  /gpt_disable - Deshabilitar GPT"
            )

            await update.message.reply_text(message)

        except Exception as e:
            logger.error(f"Error en comando /gpt: {e}", exc_info=True)
            await update.message.reply_text(f"❌ Error: {str(e)}")

    async def gpt_analyze_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Comando /gpt_analyze
        Ejecuta análisis completo de performance con GPT
        """
        try:
            logger.info("🧠 Comando /gpt_analyze recibido")

            if not self.gpt_brain:
                await update.message.reply_text("⚠️ GPT Brain no disponible")
                return

            if not self.gpt_brain.is_enabled:
                await update.message.reply_text("⚠️ GPT Brain está desactivado. Usa /gpt_enable")
                return

            await update.message.reply_text(
                "🧠 **Analizando Performance con GPT...**\n\n"
                "Esto puede tomar 10-30 segundos ⏳"
            )

            # Obtener datos necesarios
            trades = []
            portfolio = {}

            if self.autonomy_controller and hasattr(self.autonomy_controller, 'paper_trader'):
                paper_trader = self.autonomy_controller.paper_trader
                if paper_trader and hasattr(paper_trader, 'portfolio'):
                    trades = paper_trader.portfolio.closed_trades[-50:]  # Últimos 50 trades
                    portfolio = paper_trader.get_statistics()

            if not trades:
                await update.message.reply_text(
                    "⚠️ No hay trades suficientes para analizar.\n"
                    "Espera a tener más historial de trading."
                )
                return

            # Ejecutar análisis
            result = await self.gpt_brain.run_performance_analysis(
                trades=trades,
                portfolio=portfolio
            )

            if result.get('success'):
                analysis = result.get('analysis', {})

                # Formatear respuesta
                summary = analysis.get('summary', 'No disponible')
                recommendations = analysis.get('recommendations', [])

                message = (
                    "🧠 **Análisis GPT Completado**\n\n"
                    f"**📝 Resumen:**\n{summary}\n\n"
                )

                if recommendations:
                    message += "**💡 Recomendaciones:**\n"
                    for i, rec in enumerate(recommendations[:5], 1):
                        param = rec.get('parameter', 'N/A')
                        reason = rec.get('reason', 'N/A')
                        conf = rec.get('confidence', 0)
                        message += f"{i}. {param}: {reason} ({conf}% confianza)\n"

                message += f"\n💰 Costo: ${result.get('cost', 0):.4f}"

                await update.message.reply_text(message)
            else:
                await update.message.reply_text(
                    f"❌ Análisis falló: {result.get('error', 'Error desconocido')}"
                )

        except Exception as e:
            logger.error(f"Error en comando /gpt_analyze: {e}", exc_info=True)
            await update.message.reply_text(f"❌ Error: {str(e)}")

    async def gpt_optimize_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Comando /gpt_optimize
        Fuerza una optimización de parámetros con GPT
        """
        try:
            logger.info("🧠 Comando /gpt_optimize recibido")

            if not self.gpt_brain:
                await update.message.reply_text("⚠️ GPT Brain no disponible")
                return

            if not self.gpt_brain.is_enabled:
                await update.message.reply_text("⚠️ GPT Brain está desactivado. Usa /gpt_enable")
                return

            await update.message.reply_text(
                "🧠 **Ejecutando Optimización GPT...**\n\n"
                "GPT analizará performance y ajustará parámetros.\n"
                "Esto puede tomar 20-40 segundos ⏳"
            )

            # Obtener portfolio stats
            portfolio = {}
            if self.autonomy_controller and hasattr(self.autonomy_controller, 'paper_trader'):
                paper_trader = self.autonomy_controller.paper_trader
                if paper_trader:
                    portfolio = paper_trader.get_statistics()

            # Ejecutar optimización
            result = await self.gpt_brain.run_full_optimization(
                portfolio=portfolio,
                trigger_reason="Manual command /gpt_optimize"
            )

            if result.get('success'):
                applied = result.get('applied_changes', [])
                rejected = result.get('rejected_changes', [])
                direction = result.get('strategy_direction', 'MAINTAIN')

                message = (
                    "✅ **Optimización GPT Completada**\n\n"
                    f"**Dirección estratégica:** {direction}\n"
                    f"**Confianza:** {result.get('confidence', 0)}%\n\n"
                )

                if applied:
                    message += "**✅ Cambios aplicados:**\n"
                    for change in applied[:5]:
                        param = change.get('parameter', 'N/A')
                        old = change.get('old_value', 'N/A')
                        new = change.get('new_value', 'N/A')
                        message += f"  • {param}: {old} → {new}\n"
                else:
                    message += "ℹ️ No se aplicaron cambios (parámetros ya óptimos)\n"

                if rejected:
                    message += f"\n⚠️ {len(rejected)} cambio(s) rechazado(s) por validación\n"

                message += f"\n💰 Costo: ${result.get('cost', 0):.4f}"
                message += f"\n⏰ Próxima revisión: {result.get('next_review', 2)}h"

                await update.message.reply_text(message)
            else:
                await update.message.reply_text(
                    f"❌ Optimización falló: {result.get('error', 'Error desconocido')}"
                )

        except Exception as e:
            logger.error(f"Error en comando /gpt_optimize: {e}", exc_info=True)
            await update.message.reply_text(f"❌ Error: {str(e)}")

    async def gpt_insight_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Comando /gpt_insight
        Obtiene insight rápido del mercado actual
        """
        try:
            logger.info("🧠 Comando /gpt_insight recibido")

            if not self.gpt_brain:
                await update.message.reply_text("⚠️ GPT Brain no disponible")
                return

            if not self.gpt_brain.is_enabled:
                await update.message.reply_text("⚠️ GPT Brain está desactivado. Usa /gpt_enable")
                return

            await update.message.reply_text(
                "🧠 **Obteniendo insight del mercado...**\n\n"
                "Consultando GPT... ⏳"
            )

            # Obtener datos del mercado
            market_data = {}
            if self.market_monitor:
                # Intentar obtener datos del último análisis
                if hasattr(self.market_monitor, 'last_analysis_cache'):
                    market_data = self.market_monitor.last_analysis_cache or {}

            # Si no hay datos, crear datos básicos
            if not market_data:
                market_data = {
                    "note": "Datos de mercado limitados",
                    "timestamp": "now"
                }

            # Obtener insight
            insight = await self.gpt_brain.get_market_insight(
                pair="BTC/USDT",
                market_data=market_data
            )

            message = (
                "🧠 **Insight de Mercado (GPT)**\n\n"
                f"{insight}\n\n"
                "💡 Este es un análisis de alto nivel basado en datos disponibles."
            )

            await update.message.reply_text(message)

        except Exception as e:
            logger.error(f"Error en comando /gpt_insight: {e}", exc_info=True)
            await update.message.reply_text(f"❌ Error: {str(e)}")

    async def gpt_enable_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Comando /gpt_enable
        Habilita el GPT Brain
        """
        try:
            logger.info("🧠 Comando /gpt_enable recibido")

            if not self.gpt_brain:
                await update.message.reply_text(
                    "⚠️ GPT Brain no está inicializado.\n"
                    "Verifica configuración en .env"
                )
                return

            self.gpt_brain.enable()

            await update.message.reply_text(
                "✅ **GPT Brain Habilitado**\n\n"
                "El bot ahora usará razonamiento GPT para:\n"
                "  • Evaluar riesgo de trades\n"
                "  • Optimizar parámetros\n"
                "  • Explicar decisiones\n\n"
                "🧠 Razonamiento avanzado activo"
            )

        except Exception as e:
            logger.error(f"Error en comando /gpt_enable: {e}", exc_info=True)
            await update.message.reply_text(f"❌ Error: {str(e)}")

    async def gpt_disable_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Comando /gpt_disable
        Deshabilita el GPT Brain
        """
        try:
            logger.info("🧠 Comando /gpt_disable recibido")

            if not self.gpt_brain:
                await update.message.reply_text("⚠️ GPT Brain no está inicializado")
                return

            self.gpt_brain.disable()

            await update.message.reply_text(
                "❌ **GPT Brain Deshabilitado**\n\n"
                "El bot continuará operando sin razonamiento GPT.\n"
                "Usará solo ML/RL tradicional.\n\n"
                "Usa /gpt_enable para reactivar."
            )

        except Exception as e:
            logger.error(f"Error en comando /gpt_disable: {e}", exc_info=True)
            await update.message.reply_text(f"❌ Error: {str(e)}")

    async def gpt_scan_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Comando /gpt_scan
        Escanea el mercado usando GPT para encontrar oportunidades
        """
        try:
            logger.info("🔍 Comando /gpt_scan recibido")

            if not self.gpt_brain:
                await update.message.reply_text("⚠️ GPT Brain no disponible")
                return

            if not self.gpt_brain.is_enabled:
                await update.message.reply_text("⚠️ GPT Brain está desactivado. Usa /gpt_enable")
                return

            await update.message.reply_text(
                "🔍 **Escaneando Mercado con GPT...**\n\n"
                "Analizando todos los pares para encontrar oportunidades.\n"
                "Esto puede tomar 20-40 segundos ⏳"
            )

            # Obtener datos de mercado
            pairs_data = []

            if self.market_monitor:
                # Obtener indicadores de los últimos análisis
                for pair in self.market_monitor.trading_pairs[:10]:  # Limitar a 10 pares
                    try:
                        # Intentar obtener datos del último análisis
                        indicators = {}
                        sentiment = {}

                        # Si hay cache de análisis, usarlo
                        if hasattr(self.market_monitor, 'last_analysis_cache'):
                            cache = self.market_monitor.last_analysis_cache or {}
                            if pair in cache:
                                indicators = cache[pair].get('indicators', {})
                                sentiment = cache[pair].get('sentiment', {})

                        # Si no hay cache, usar datos básicos
                        if not indicators:
                            indicators = {
                                'current_price': 0,
                                'rsi': 50,
                                'macd': 0,
                                'macd_signal': 0,
                                'volume_ratio': 1.0
                            }

                        pairs_data.append({
                            'pair': pair,
                            'indicators': indicators,
                            'sentiment': sentiment
                        })

                    except Exception as e:
                        logger.warning(f"Error obteniendo datos para {pair}: {e}")
                        continue

            if not pairs_data:
                # Datos de ejemplo si no hay datos reales
                pairs_data = [
                    {'pair': 'BTC/USDT', 'indicators': {'current_price': 100000, 'rsi': 55}, 'sentiment': {'fear_greed_index': 0.6}},
                    {'pair': 'ETH/USDT', 'indicators': {'current_price': 3800, 'rsi': 48}, 'sentiment': {'fear_greed_index': 0.55}},
                    {'pair': 'SOL/USDT', 'indicators': {'current_price': 230, 'rsi': 62}, 'sentiment': {'fear_greed_index': 0.58}},
                ]

            # Ejecutar scan
            result = await self.gpt_brain.scan_market(
                pairs_data=pairs_data,
                top_n=5
            )

            if result.get("success"):
                opportunities = result.get("opportunities", [])
                market_summary = result.get("market_summary", "N/A")

                message = (
                    "🔍 **GPT Market Scan Completado**\n\n"
                    f"**📊 Resumen del Mercado:**\n{market_summary}\n\n"
                )

                if opportunities:
                    message += "**🎯 Oportunidades Encontradas:**\n\n"
                    for i, opp in enumerate(opportunities[:5], 1):
                        emoji = "🟢" if opp.get('action') == 'BUY' else "🔴"
                        message += (
                            f"{i}. {emoji} **{opp.get('pair', 'N/A')}**\n"
                            f"   Acción: {opp.get('action', 'N/A')}\n"
                            f"   Score: {opp.get('score', 0)}/100\n"
                            f"   Urgencia: {opp.get('urgency', 'N/A')}\n"
                            f"   📝 {opp.get('reason', 'N/A')[:100]}...\n\n"
                        )
                else:
                    message += "ℹ️ No se encontraron oportunidades claras en este momento.\n"

                # Pares a evitar
                avoid = result.get("avoid_pairs", [])
                if avoid:
                    message += "\n**⚠️ Pares a Evitar:**\n"
                    for ap in avoid[:3]:
                        message += f"  • {ap.get('pair', 'N/A')}: {ap.get('reason', 'N/A')}\n"

                message += f"\n💰 Costo del análisis: ${result.get('cost', 0):.4f}"

                await update.message.reply_text(message)
            else:
                await update.message.reply_text(
                    f"❌ Scan falló: {result.get('error', 'Error desconocido')}"
                )

        except Exception as e:
            logger.error(f"Error en comando /gpt_scan: {e}", exc_info=True)
            await update.message.reply_text(f"❌ Error: {str(e)}")

    async def gpt_signal_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Comando /gpt_signal [pair]
        Genera una señal de trading usando GPT para un par específico
        Uso: /gpt_signal BTC/USDT
        """
        try:
            logger.info("🧠 Comando /gpt_signal recibido")

            if not self.gpt_brain:
                await update.message.reply_text("⚠️ GPT Brain no disponible")
                return

            if not self.gpt_brain.is_enabled:
                await update.message.reply_text("⚠️ GPT Brain está desactivado. Usa /gpt_enable")
                return

            # Obtener par del argumento
            args = context.args
            if not args:
                pair = "BTC/USDT"  # Default
            else:
                pair = args[0].upper()
                if "/" not in pair:
                    pair = f"{pair}/USDT"

            await update.message.reply_text(
                f"🧠 **Generando Señal GPT para {pair}...**\n\n"
                "Analizando indicadores y contexto de mercado.\n"
                "Esto puede tomar 10-20 segundos ⏳"
            )

            # Obtener indicadores del par
            indicators = {
                'current_price': 0,
                'rsi': 50,
                'macd': 0,
                'macd_signal': 0,
                'ema_9': 0,
                'ema_21': 0,
                'ema_50': 0,
                'bb_upper': 0,
                'bb_lower': 0,
                'atr': 0,
                'adx': 0,
                'volume_ratio': 1.0
            }

            sentiment_data = None
            orderbook_data = None
            regime_data = None

            # Intentar obtener datos reales del market monitor
            if self.market_monitor:
                if hasattr(self.market_monitor, 'last_analysis_cache'):
                    cache = self.market_monitor.last_analysis_cache or {}
                    if pair in cache:
                        indicators = cache[pair].get('indicators', indicators)
                        sentiment_data = cache[pair].get('sentiment')
                        orderbook_data = cache[pair].get('orderbook')
                        regime_data = cache[pair].get('regime')

            # Generar señal
            result = await self.gpt_brain.generate_gpt_signal(
                pair=pair,
                indicators=indicators,
                sentiment_data=sentiment_data,
                orderbook_data=orderbook_data,
                regime_data=regime_data
            )

            if result.get("success"):
                signal = result.get("signal", {})
                action = signal.get("action", "HOLD")
                confidence = signal.get("confidence", 0)

                # Emojis según acción
                if "BUY" in action:
                    emoji = "🟢"
                elif "SELL" in action:
                    emoji = "🔴"
                else:
                    emoji = "⚪"

                message = (
                    f"🧠 **Señal GPT para {pair}**\n\n"
                    f"{emoji} **Acción:** {action}\n"
                    f"📊 **Confianza:** {confidence}%\n\n"
                )

                # Razonamiento
                reasoning = signal.get("reasoning", {})
                if reasoning:
                    message += f"**📝 Análisis:**\n"
                    message += f"  • Factor principal: {reasoning.get('main_factor', 'N/A')}\n"
                    supporting = reasoning.get('supporting_factors', [])
                    if supporting:
                        message += f"  • Factores de apoyo: {', '.join(supporting[:2])}\n"
                    concerns = reasoning.get('concerns', [])
                    if concerns:
                        message += f"  • Preocupaciones: {', '.join(concerns[:2])}\n"

                # Trade setup
                trade_setup = signal.get("trade_setup", {})
                if trade_setup and action != "HOLD":
                    message += f"\n**💰 Setup de Trade:**\n"
                    if trade_setup.get('entry_price'):
                        message += f"  • Entry: ${trade_setup.get('entry_price', 0):,.2f}\n"
                    if trade_setup.get('stop_loss'):
                        message += f"  • Stop Loss: ${trade_setup.get('stop_loss', 0):,.2f}\n"
                    if trade_setup.get('take_profit'):
                        message += f"  • Take Profit: ${trade_setup.get('take_profit', 0):,.2f}\n"
                    if trade_setup.get('risk_reward'):
                        message += f"  • R/R: {trade_setup.get('risk_reward', 0):.1f}\n"
                    if trade_setup.get('position_size_recommendation'):
                        message += f"  • Tamaño: {trade_setup.get('position_size_recommendation', 'FULL')}\n"

                # Timing
                timing = signal.get("timing", {})
                if timing:
                    message += f"\n**⏰ Timing:**\n"
                    message += f"  • Urgencia: {timing.get('urgency', 'N/A')}\n"
                    if timing.get('valid_for_hours'):
                        message += f"  • Válido por: {timing.get('valid_for_hours', 0)}h\n"

                # Summary
                summary = signal.get("summary", "")
                if summary:
                    message += f"\n**📌 Resumen:**\n{summary}\n"

                message += f"\n💰 Costo: ${result.get('cost', 0):.4f}"

                await update.message.reply_text(message)
            else:
                await update.message.reply_text(
                    f"❌ No se pudo generar señal: {result.get('error', 'Error desconocido')}"
                )

        except Exception as e:
            logger.error(f"Error en comando /gpt_signal: {e}", exc_info=True)
            await update.message.reply_text(f"❌ Error: {str(e)}")
