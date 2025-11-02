"""
Telegram Commands Handler
Maneja comandos de Telegram para el bot
"""
import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from typing import Optional

logger = logging.getLogger(__name__)


class TelegramCommands:
    """
    Manejador de comandos de Telegram
    - /export_intelligence: Export manual de inteligencia aprendida
    - /status: Status del sistema autónomo
    """

    def __init__(self, autonomy_controller=None, telegram_token: str = None):
        """
        Args:
            autonomy_controller: Instancia del AutonomyController
            telegram_token: Token del bot de Telegram
        """
        self.autonomy_controller = autonomy_controller
        self.telegram_token = telegram_token
        self.application = None

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
            self.application.add_handler(CommandHandler("status", self.status_command))
            self.application.add_handler(CommandHandler("help", self.help_command))

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
        Muestra status del sistema autónomo
        """
        try:
            logger.info("📊 Comando /status recibido")

            if not self.autonomy_controller:
                await update.message.reply_text(
                    "⚠️ Sistema autónomo no disponible"
                )
                return

            # Obtener estadísticas
            stats = self.autonomy_controller.get_statistics()
            backup_status = self.autonomy_controller.git_backup.get_backup_status()

            message = (
                "📊 **Status del Sistema Autónomo**\n\n"
                f"🤖 Estado: {'✅ Activo' if stats['active'] else '❌ Inactivo'}\n"
                f"🎯 Modo: {stats['decision_mode']}\n\n"
                "**Aprendizaje:**\n"
                f"  • Trades procesados: {stats['total_trades_processed']}\n"
                f"  • Parámetros modificados: {stats['total_parameter_changes']} veces\n"
                f"  • Estados aprendidos: {stats['rl_agent']['q_table_size']}\n"
                f"  • Win rate RL: {stats['rl_agent']['success_rate']:.1f}%\n\n"
                "**Optimización:**\n"
                f"  • Trials completados: {stats['parameter_optimizer']['total_trials']}\n"
                f"  • Mejor score: {stats['parameter_optimizer']['best_score']:.3f}\n"
                f"  • Parámetros activos: {stats['current_parameters_count']}\n\n"
                "**Backups:**\n"
                f"  • Auto-backup: {'✅ Activo' if backup_status['running'] else '❌ Inactivo'}\n"
                f"  • Próximo backup: {backup_status.get('next_backup', 'N/A')}\n"
                f"  • Último backup: {backup_status.get('last_backup', 'Ninguno')}\n\n"
                "Usa /export_intelligence para backup manual"
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
                "  └─ Hace backup a Git/GitHub\n\n"
                "/status\n"
                "  ├─ Muestra estado del sistema autónomo\n"
                "  ├─ Estadísticas de aprendizaje\n"
                "  └─ Info de backups\n\n"
                "/help\n"
                "  └─ Muestra este mensaje\n\n"
                "**Auto-Backup**: Cada 24h automático"
            )

            await update.message.reply_text(message)

        except Exception as e:
            logger.error(f"Error en comando help: {e}", exc_info=True)
