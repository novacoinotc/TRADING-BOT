"""
Learning Persistence - Persistencia de inteligencia aprendida
Guarda y carga TODO el conocimiento adquirido para sobrevivir redeploys
"""
import json
import logging
import gzip
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


class LearningPersistence:
    """
    Gestiona persistencia de inteligencia aprendida
    - Guarda todo el conocimiento del RL Agent
    - Guarda historial de optimización de parámetros
    - Guarda estadísticas y métricas
    - Comprime datos para eficiencia
    - Genera archivo de importación para redeploys
    """

    def __init__(self, storage_dir: str = "data/autonomous"):
        """
        Args:
            storage_dir: Directorio para guardar archivos de persistencia
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Archivos de persistencia
        self.main_file = self.storage_dir / "learned_intelligence.json.gz"
        self.backup_file = self.storage_dir / "learned_intelligence_backup.json.gz"
        self.export_file = self.storage_dir / "intelligence_export.json"  # Para importar fácil

        logger.info(f"💾 Learning Persistence inicializado en: {self.storage_dir}")

    def save_full_state(
        self,
        rl_agent_state: Dict,
        optimizer_state: Dict,
        performance_history: Dict,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Guarda estado completo del sistema autónomo

        Args:
            rl_agent_state: Estado del RL Agent (Q-table, estadísticas, etc.)
            optimizer_state: Estado del Parameter Optimizer (trials, best config, etc.)
            performance_history: Historial de performance del bot
            metadata: Información adicional (versión, timestamp, etc.)

        Returns:
            True si guardado fue exitoso
        """
        try:
            # Crear backup del archivo anterior si existe
            if self.main_file.exists():
                import shutil
                shutil.copy(self.main_file, self.backup_file)
                logger.debug("📦 Backup creado")

            # Construir estado completo
            full_state = {
                'version': '1.0',
                'timestamp': datetime.now().isoformat(),
                'rl_agent': rl_agent_state,
                'parameter_optimizer': optimizer_state,
                'performance_history': performance_history,
                'metadata': metadata or {}
            }

            # Calcular checksum para validación
            state_str = json.dumps(full_state, sort_keys=True)
            checksum = hashlib.sha256(state_str.encode()).hexdigest()
            full_state['checksum'] = checksum

            # Guardar comprimido (ahorra espacio)
            with gzip.open(self.main_file, 'wt', encoding='utf-8') as f:
                json.dump(full_state, f, indent=2)

            # Guardar versión sin comprimir para fácil importación
            with open(self.export_file, 'w', encoding='utf-8') as f:
                json.dump(full_state, f, indent=2)

            file_size = self.main_file.stat().st_size / 1024  # KB

            logger.info(
                f"✅ Inteligencia guardada exitosamente "
                f"({file_size:.1f} KB comprimido)"
            )

            # Log resumen de lo guardado
            self._log_save_summary(full_state)

            return True

        except Exception as e:
            logger.error(f"❌ Error guardando inteligencia: {e}", exc_info=True)
            return False

    def load_full_state(self) -> Optional[Dict]:
        """
        Carga estado completo del sistema autónomo

        Returns:
            Dict con todo el estado guardado, o None si no existe
        """
        try:
            # Intentar cargar archivo principal
            if not self.main_file.exists():
                logger.warning("⚠️ No existe archivo de inteligencia guardado")
                return None

            # Cargar y descomprimir
            with gzip.open(self.main_file, 'rt', encoding='utf-8') as f:
                full_state = json.load(f)

            # Validar checksum
            saved_checksum = full_state.pop('checksum', None)
            state_str = json.dumps(full_state, sort_keys=True)
            calculated_checksum = hashlib.sha256(state_str.encode()).hexdigest()

            if saved_checksum != calculated_checksum:
                logger.warning("⚠️ Checksum no coincide - archivo puede estar corrupto")
                # Intentar cargar backup
                return self._load_backup()

            # Log resumen de lo cargado
            self._log_load_summary(full_state)

            logger.info("✅ Inteligencia cargada exitosamente")

            return full_state

        except Exception as e:
            logger.error(f"❌ Error cargando inteligencia: {e}", exc_info=True)
            # Intentar cargar backup
            return self._load_backup()

    def _load_backup(self) -> Optional[Dict]:
        """Intenta cargar desde archivo de backup"""
        try:
            if not self.backup_file.exists():
                logger.error("❌ No existe archivo de backup")
                return None

            logger.info("🔄 Intentando cargar desde backup...")

            with gzip.open(self.backup_file, 'rt', encoding='utf-8') as f:
                full_state = json.load(f)

            logger.info("✅ Backup cargado exitosamente")
            return full_state

        except Exception as e:
            logger.error(f"❌ Error cargando backup: {e}", exc_info=True)
            return None

    def _log_save_summary(self, state: Dict):
        """Log resumen de lo que se guardó"""
        rl_stats = state.get('rl_agent', {}).get('statistics', {})
        opt_stats = state.get('parameter_optimizer', {})

        logger.info(
            f"📊 Resumen guardado:\n"
            f"  • RL Agent: {rl_stats.get('total_trades', 0)} trades, "
            f"{rl_stats.get('success_rate', 0):.1f}% win rate\n"
            f"  • Parameter Optimizer: {opt_stats.get('total_trials', 0)} trials, "
            f"mejor score: {opt_stats.get('best_performance', 0):.3f}\n"
            f"  • Q-Table: {rl_stats.get('q_table_size', 0)} estados aprendidos\n"
            f"  • Timestamp: {state.get('timestamp', 'N/A')}"
        )

    def _log_load_summary(self, state: Dict):
        """Log resumen de lo que se cargó"""
        rl_stats = state.get('rl_agent', {}).get('statistics', {})
        opt_stats = state.get('parameter_optimizer', {})

        logger.info(
            f"📊 Resumen cargado:\n"
            f"  • RL Agent: {rl_stats.get('total_trades', 0)} trades, "
            f"{rl_stats.get('success_rate', 0):.1f}% win rate\n"
            f"  • Parameter Optimizer: {opt_stats.get('total_trials', 0)} trials\n"
            f"  • Q-Table: {rl_stats.get('q_table_size', 0)} estados\n"
            f"  • Guardado: {state.get('timestamp', 'N/A')}"
        )

    def export_for_import(self) -> str:
        """
        Exporta inteligencia a archivo JSON legible para importar después de redeploy

        Returns:
            Path al archivo exportado
        """
        try:
            if not self.main_file.exists():
                logger.warning("⚠️ No hay inteligencia para exportar")
                return ""

            # Cargar estado actual
            with gzip.open(self.main_file, 'rt', encoding='utf-8') as f:
                full_state = json.load(f)

            # Guardar versión legible sin comprimir
            export_path = self.storage_dir / f"intelligence_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(full_state, f, indent=2)

            file_size = export_path.stat().st_size / 1024  # KB

            logger.info(
                f"📤 Inteligencia exportada a: {export_path}\n"
                f"   Tamaño: {file_size:.1f} KB\n"
                f"   Usar este archivo para importar después de redeploy"
            )

            return str(export_path)

        except Exception as e:
            logger.error(f"❌ Error exportando inteligencia: {e}", exc_info=True)
            return ""

    def import_from_file(self, file_path: str) -> bool:
        """
        Importa inteligencia desde archivo exportado

        Args:
            file_path: Path al archivo de exportación

        Returns:
            True si importación fue exitosa
        """
        try:
            import_path = Path(file_path)

            if not import_path.exists():
                logger.error(f"❌ Archivo no existe: {file_path}")
                return False

            # Cargar estado desde archivo
            with open(import_path, 'r', encoding='utf-8') as f:
                full_state = json.load(f)

            # Validar estructura básica
            if 'rl_agent' not in full_state or 'parameter_optimizer' not in full_state:
                logger.error("❌ Archivo no tiene estructura válida")
                return False

            # Guardar como archivo principal
            with gzip.open(self.main_file, 'wt', encoding='utf-8') as f:
                json.dump(full_state, f, indent=2)

            logger.info(f"✅ Inteligencia importada exitosamente desde: {file_path}")
            self._log_load_summary(full_state)

            return True

        except Exception as e:
            logger.error(f"❌ Error importando inteligencia: {e}", exc_info=True)
            return False

    def get_storage_info(self) -> Dict:
        """Retorna información sobre archivos de persistencia"""
        info = {
            'main_file_exists': self.main_file.exists(),
            'backup_exists': self.backup_file.exists(),
            'export_exists': self.export_file.exists(),
            'storage_dir': str(self.storage_dir)
        }

        if self.main_file.exists():
            info['main_file_size_kb'] = self.main_file.stat().st_size / 1024
            info['main_file_modified'] = datetime.fromtimestamp(
                self.main_file.stat().st_mtime
            ).isoformat()

        return info

    def auto_save_periodic(
        self,
        rl_agent_state: Dict,
        optimizer_state: Dict,
        performance_history: Dict,
        interval_minutes: int = 30
    ) -> bool:
        """
        Guarda automáticamente si ha pasado suficiente tiempo

        Args:
            interval_minutes: Intervalo mínimo entre guardados automáticos

        Returns:
            True si guardó, False si no era necesario
        """
        # Verificar última modificación
        if self.main_file.exists():
            last_modified = datetime.fromtimestamp(self.main_file.stat().st_mtime)
            elapsed = (datetime.now() - last_modified).total_seconds() / 60

            if elapsed < interval_minutes:
                logger.debug(f"⏳ Auto-save: esperando ({elapsed:.0f}/{interval_minutes} min)")
                return False

        # Guardar
        logger.info(f"💾 Auto-save: guardando inteligencia (intervalo: {interval_minutes} min)")
        return self.save_full_state(
            rl_agent_state,
            optimizer_state,
            performance_history,
            metadata={'auto_save': True, 'interval_minutes': interval_minutes}
        )
