# 🤖 Sistema Autónomo de Trading - Control Absoluto

## Descripción

Sistema de IA completamente autónomo que tiene **CONTROL ABSOLUTO** sobre todos los parámetros y decisiones del bot de trading. La IA aprende, se optimiza y evoluciona sin intervención humana.

## Componentes Principales

### 1. **RL Agent** (`rl_agent.py`)
- **Reinforcement Learning** con Q-Learning
- Aprende de cada trade (recompensas/penalizaciones)
- Experience Replay para mejor aprendizaje
- Exploration vs Exploitation balanceado
- Almacena Q-Table con estados y acciones aprendidas

### 2. **Parameter Optimizer** (`parameter_optimizer.py`)
- Optimización de **TODOS** los parámetros sin limitaciones
- Estrategias de exploración y optimización
- Meta-learning: aprende qué parámetros son más importantes
- Búsqueda inteligente basada en historial de performance
- **Parámetros optimizables:**
  - Trading: CHECK_INTERVAL, CONSERVATIVE_THRESHOLD, FLASH_THRESHOLD, etc.
  - Indicadores técnicos: RSI_OVERSOLD, MACD_FAST, EMA_SHORT, etc.
  - Risk Management: BASE_POSITION_SIZE_PCT, MAX_DRAWDOWN_LIMIT, etc.
  - ML Hyperparameters: N_ESTIMATORS, MAX_DEPTH, LEARNING_RATE, etc.

### 3. **Learning Persistence** (`learning_persistence.py`)
- Guarda toda la inteligencia aprendida
- Sobrevive redeploys de Railway
- Compresión de datos para eficiencia
- Backups automáticos
- Exportación para fácil importación
- Validación de integridad (checksum)

### 4. **Autonomy Controller** (`autonomy_controller.py`)
- **Cerebro principal** del sistema
- Coordina todos los componentes
- Decide cuándo optimizar parámetros
- Notifica CADA cambio a Telegram
- Auto-save periódico de inteligencia
- Gestión completa del ciclo de vida

## Características Clave

### 🎯 Control Absoluto
- La IA puede modificar **CUALQUIER** parámetro
- Sin limitaciones ni restricciones humanas
- Decisiones basadas en resultados reales
- Evolución continua

### 📚 Aprendizaje Continuo
- Aprende de cada trade ejecutado
- Mejora estrategia con el tiempo
- Meta-learning sobre importancia de parámetros
- Experience replay para mejor aprendizaje

### 💾 Persistencia de Inteligencia
- **CRÍTICO**: Sobrevive redeploys
- Auto-save cada 30 minutos (configurable)
- Backups automáticos
- Archivo de exportación para importar después de redeploy

### 🔔 Notificaciones Transparentes
- Cada modificación es notificada a Telegram
- Explica **por qué** cambió parámetros
- Reporta mejoras encontradas
- Status de aprendizaje

## Flujo de Funcionamiento

```
1. Bot inicia → Carga inteligencia previa (si existe)
                ↓
2. Trades ejecutados → RL Agent aprende
                ↓
3. Cada N trades → Parameter Optimizer sugiere cambios
                ↓
4. Cambios aplicados → Telegram notificado
                ↓
5. Performance medido → RL Agent y Optimizer actualizados
                ↓
6. Auto-save periódico → Inteligencia guardada
                ↓
7. Ciclo se repite → IA mejora continuamente
```

## Configuración

En `config/config.py` o variables de entorno:

```python
# Habilitar modo autónomo
ENABLE_AUTONOMOUS_MODE = true  # ¡ACTIVAR PARA CONTROL TOTAL!

# Auto-save (minutos)
AUTONOMOUS_AUTO_SAVE_INTERVAL = 30

# Intervalo de optimización (horas)
AUTONOMOUS_OPTIMIZATION_INTERVAL = 2.0

# Mínimo trades antes de optimizar
AUTONOMOUS_MIN_TRADES_BEFORE_OPT = 20
```

## Integración

### En `main.py`
```python
# Sistema autónomo se inicializa automáticamente si está habilitado
autonomy_controller = AutonomyController(
    telegram_notifier=monitor.notifier,
    auto_save_interval_minutes=30,
    optimization_check_interval_hours=2.0,
    min_trades_before_optimization=20
)
await autonomy_controller.initialize()
```

### En `market_monitor.py`
- Trades cerrados se envían al controller para aprendizaje
- RL Agent aprende de outcomes
- Parameter Optimizer ajusta configuración

## Archivos de Persistencia

### Ubicación
`data/autonomous/`

### Archivos generados
- `learned_intelligence.json.gz` - Inteligencia comprimida (principal)
- `learned_intelligence_backup.json.gz` - Backup automático
- `intelligence_export.json` - Export legible para importar
- `intelligence_export_YYYYMMDD_HHMMSS.json` - Exports timestamped

### Contenido guardado
```json
{
  "version": "1.0",
  "timestamp": "2025-11-01T...",
  "rl_agent": {
    "q_table": {...},  // Estados y acciones aprendidas
    "statistics": {...},  // Stats del agente
    "episode_rewards": [...]
  },
  "parameter_optimizer": {
    "trial_history": [...],  // Historial de trials
    "best_config": {...},  // Mejor configuración encontrada
    "parameter_importance": {...}  // Importancia de cada parámetro
  },
  "performance_history": {...},  // Últimos 100 trades
  "metadata": {
    "current_parameters": {...},  // Parámetros actuales
    "total_trades_processed": 1234,
    "total_parameter_changes": 42
  }
}
```

## Importar Inteligencia Después de Redeploy

### Opción 1: Automática
El sistema carga automáticamente `data/autonomous/learned_intelligence.json.gz` al iniciar.

### Opción 2: Manual
```python
from src.autonomous.learning_persistence import LearningPersistence

persistence = LearningPersistence()
persistence.import_from_file('path/to/intelligence_export.json')
```

## Notificaciones de Telegram

El sistema envía notificaciones para:

### Al iniciar
- ✅ Inteligencia cargada (si existe)
- 🆕 Primera ejecución (si es nuevo)

### Durante operación
- 🤖 Parámetros modificados (cada cambio)
- 🎉 Trade exitoso aprendido (profit > 2%)
- 📚 Trade perdedor analizado (loss < -2%)
- 🎉 Nueva mejor configuración encontrada

### Al apagar
- 🛑 Resumen de sesión
- 📊 Trades procesados
- 🔧 Cambios realizados

## Estrategias de Optimización

### Exploración (30%)
- Cambios aleatorios
- Descubre nuevas configuraciones
- Importante al inicio

### Optimización (70%)
- Basado en mejores resultados previos
- Modifica 2-4 parámetros más importantes
- Perturbaciones adaptativas (±20%)

### Meta-Learning
- Aprende qué parámetros tienen mayor impacto
- Prioriza modificar parámetros importantes
- Actualiza importancia con correlaciones

## Métricas de Performance

El sistema considera:
- **Win Rate** (25% peso)
- **ROI** (30% peso)
- **Sharpe Ratio** (15% peso)
- **Profit Factor** (20% peso)
- **Max Drawdown** (10% peso, invertido)

Score total = Suma ponderada de métricas

## Intervenciones Automáticas

La IA optimiza cuando:
1. ⏰ Intervalo de tiempo alcanzado (2 horas)
2. 📊 Suficientes trades procesados (20+)
3. ⚠️ Win rate crítico (< 35%)
4. ⚠️ ROI crítico (< -10%)
5. 🎯 Win rate excelente (> 65%) - maximizar

## Limitaciones y Consideraciones

### Railway (sin Volumes)
- Los archivos en `data/autonomous/` se pierden en redeploy
- **Solución**: Git commit de archivos de inteligencia antes de redeploy
- O descargar `intelligence_export.json` manualmente

### Con Volumes (Railway Pro+)
- Persistencia automática total
- Sin necesidad de commits manuales

## Desarrollo Futuro

Posibles mejoras:
- [ ] Algoritmos más avanzados (PPO, A3C)
- [ ] Multi-armed bandits
- [ ] Bayesian Optimization
- [ ] Auto-tuning de hyperparameters del RL Agent
- [ ] Ensemble de estrategias
- [ ] Transfer learning entre pares

## Logs y Debugging

```bash
# Ver logs del sistema autónomo
tail -f logs/trading_bot.log | grep -E "(🤖|🧠|🎯|💾)"

# Verificar archivos de persistencia
ls -lh data/autonomous/

# Ver contenido de inteligencia guardada
cat data/autonomous/intelligence_export.json | jq
```

## Preguntas Frecuentes

### ¿Qué pasa si el bot crashea?
Auto-save cada 30 min garantiza que máximo se pierden 30 min de aprendizaje.

### ¿Cómo sé qué está cambiando la IA?
Cada cambio es notificado a Telegram con explicación detallada.

### ¿Puedo desactivar el modo autónomo?
Sí, en Railway variables: `ENABLE_AUTONOMOUS_MODE=false`

### ¿Cuánto tiempo tarda en aprender?
- Primeras mejoras: ~50-100 trades
- Optimización significativa: ~500-1000 trades
- Madurez: ~2000+ trades (1-2 meses)

### ¿Puedo modificar parámetros manualmente?
Sí, pero la IA los sobrescribirá en próxima optimización.

## Autor

Sistema diseñado para **control absoluto** y **aprendizaje continuo** sin intervención humana.

**Modo**: AUTONOMÍA TOTAL ✨
