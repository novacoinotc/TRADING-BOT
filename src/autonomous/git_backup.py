"""
Git Backup System - Auto-commit de inteligencia aprendida cada 24h
Asegura que nunca se pierda el aprendizaje del bot
"""
import logging
import asyncio
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class GitBackup:
    """
    Sistema de backup automático a Git
    - Auto-commit cada 24h
    - Notifica a Telegram cuando guarda
    - Maneja errores de git gracefully
    """

    def __init__(
        self,
        telegram_notifier=None,
        backup_interval_hours: float = 24.0,
        backup_dir: str = "data/autonomous"
    ):
        """
        Args:
            telegram_notifier: Instancia de TelegramNotifier
            backup_interval_hours: Intervalo entre backups (default: 24h)
            backup_dir: Directorio a respaldar
        """
        self.telegram_notifier = telegram_notifier
        self.backup_interval = backup_interval_hours
        self.backup_dir = Path(backup_dir)
        self.last_backup_time = None
        self.running = False
        self.backup_task = None

        logger.info(f"💾 Git Backup inicializado (intervalo: {backup_interval_hours}h)")

    async def start_auto_backup(self):
        """Inicia el loop de auto-backup"""
        self.running = True
        self.backup_task = asyncio.create_task(self._backup_loop())
        logger.info("🔄 Auto-backup iniciado")

    async def stop_auto_backup(self):
        """Detiene el loop de auto-backup"""
        self.running = False
        if self.backup_task:
            self.backup_task.cancel()
            try:
                await self.backup_task
            except asyncio.CancelledError:
                pass
        logger.info("🛑 Auto-backup detenido")

    async def _backup_loop(self):
        """Loop principal de auto-backup"""
        try:
            while self.running:
                # Esperar intervalo
                wait_seconds = self.backup_interval * 3600  # Horas a segundos

                # Si es el primer backup, esperar solo 1 hora
                if self.last_backup_time is None:
                    wait_seconds = 3600  # 1 hora

                logger.debug(f"⏰ Próximo backup en {wait_seconds/3600:.1f} horas")
                await asyncio.sleep(wait_seconds)

                # Realizar backup
                if self.running:  # Verificar que sigue activo
                    await self.perform_backup()

        except asyncio.CancelledError:
            logger.info("Auto-backup loop cancelado")
        except Exception as e:
            logger.error(f"Error en auto-backup loop: {e}", exc_info=True)

    async def perform_backup(self, manual: bool = False) -> bool:
        """
        Realiza backup de inteligencia a Git

        Args:
            manual: True si es backup manual (por comando)

        Returns:
            True si backup fue exitoso
        """
        try:
            logger.info("💾 Iniciando backup de inteligencia a Git...")

            # Verificar que existe el directorio
            if not self.backup_dir.exists():
                logger.warning(f"Directorio no existe: {self.backup_dir}")
                return False

            # 1. Git add
            result = await self._run_git_command(
                ["git", "add", str(self.backup_dir)]
            )
            if not result:
                logger.warning("git add falló")
                return False

            # 2. Verificar si hay cambios
            status_result = await self._run_git_command(
                ["git", "status", "--porcelain", str(self.backup_dir)],
                capture_output=True
            )

            if not status_result or not status_result.strip():
                logger.info("📝 No hay cambios para commitear")
                self.last_backup_time = datetime.now()
                return True  # No es error, simplemente no hay cambios

            # 3. Git commit
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            commit_msg = f"💾 Auto-backup inteligencia aprendida - {timestamp}"

            if manual:
                commit_msg = f"📤 Backup manual de inteligencia - {timestamp}"

            result = await self._run_git_command(
                ["git", "commit", "-m", commit_msg]
            )
            if not result:
                logger.warning("git commit falló")
                return False

            # 4. Git push
            result = await self._run_git_command(
                ["git", "push"],
                timeout=60  # 60 segundos timeout para push
            )
            if not result:
                logger.warning("git push falló")
                # Aún así fue exitoso localmente
                await self._notify_backup_success(timestamp, pushed=False, manual=manual)
                self.last_backup_time = datetime.now()
                return True

            # Todo exitoso
            await self._notify_backup_success(timestamp, pushed=True, manual=manual)
            self.last_backup_time = datetime.now()

            logger.info("✅ Backup completado exitosamente")
            return True

        except Exception as e:
            logger.error(f"❌ Error en backup: {e}", exc_info=True)
            await self._notify_backup_error(str(e))
            return False

    async def _run_git_command(
        self,
        cmd: list,
        capture_output: bool = False,
        timeout: int = 30
    ) -> Optional[str]:
        """
        Ejecuta comando git de forma asíncrona

        Args:
            cmd: Comando a ejecutar
            capture_output: Si debe capturar y retornar output
            timeout: Timeout en segundos

        Returns:
            Output del comando si capture_output=True, sino True/False
        """
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE if capture_output else asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                logger.error(f"Timeout ejecutando: {' '.join(cmd)}")
                return None

            if process.returncode != 0:
                error = stderr.decode() if stderr else "Unknown error"
                logger.warning(f"Comando falló: {' '.join(cmd)}\n{error}")
                return None

            if capture_output:
                return stdout.decode() if stdout else ""

            return "success"

        except Exception as e:
            logger.error(f"Error ejecutando git: {e}")
            return None

    async def _notify_backup_success(
        self,
        timestamp: str,
        pushed: bool = True,
        manual: bool = False
    ):
        """Notifica backup exitoso a Telegram"""
        if not self.telegram_notifier:
            return

        backup_type = "📤 Backup Manual" if manual else "💾 Auto-Backup"
        push_status = "✅ Pusheado a GitHub" if pushed else "⚠️ Solo local (push falló)"

        # Calcular próximo backup
        next_backup = ""
        if not manual and self.last_backup_time:
            next_time = self.last_backup_time + timedelta(hours=self.backup_interval)
            hours_until = (next_time - datetime.now()).total_seconds() / 3600
            next_backup = f"\n⏰ Próximo backup: en {hours_until:.1f} horas"

        message = (
            f"🎉 **{backup_type} Completado**\n\n"
            f"📅 Fecha: {timestamp}\n"
            f"📊 Status: {push_status}\n"
            f"💾 Inteligencia guardada exitosamente\n"
            f"{next_backup}\n\n"
            "El aprendizaje del bot está seguro ✨"
        )

        try:
            await self.telegram_notifier.send_status_message(message)
        except Exception as e:
            logger.warning(f"No se pudo enviar notificación: {e}")

    async def _notify_backup_error(self, error: str):
        """Notifica error de backup a Telegram"""
        if not self.telegram_notifier:
            return

        message = (
            "⚠️ **Error en Auto-Backup**\n\n"
            f"Error: {error}\n\n"
            "La inteligencia local está guardada, pero no se pudo hacer commit a Git.\n"
            "Puedes intentar backup manual con /export_intelligence"
        )

        try:
            await self.telegram_notifier.send_status_message(message)
        except Exception as e:
            logger.warning(f"No se pudo enviar notificación de error: {e}")

    def get_backup_status(self) -> dict:
        """Retorna status del sistema de backup"""
        next_backup = None
        if self.last_backup_time:
            next_backup_time = self.last_backup_time + timedelta(hours=self.backup_interval)
            hours_until = (next_backup_time - datetime.now()).total_seconds() / 3600
            next_backup = f"en {hours_until:.1f} horas"

        return {
            'running': self.running,
            'last_backup': self.last_backup_time.isoformat() if self.last_backup_time else None,
            'next_backup': next_backup,
            'interval_hours': self.backup_interval
        }
