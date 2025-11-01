# 🤖 Sistema de Machine Learning + Paper Trading

## 📋 Descripción General

Este sistema implementa un bot de trading **completamente autónomo** que:

1. **Ejecuta trades virtuales** con $50,000 USDT (paper trading)
2. **Aprende de sus resultados** usando Machine Learning (XGBoost)
3. **Optimiza sus propios parámetros** automáticamente
4. **Predice señales ganadoras** en tiempo real

El objetivo es que la IA **aprenda y mejore continuamente** sin intervención manual.

---

## 🎯 Características Principales

### 💰 Paper Trading
- **Balance inicial**: $50,000 USDT virtual
- **Ejecución automática** de trades basados en señales
- **Stop Loss y Take Profit** automáticos
- **Gestión de riesgo dinámica** (1-10% por posición)
- **Máximo 10 posiciones simultáneas**
- **Tracking completo** de P&L, win rate, drawdown, Sharpe ratio

### 🧠 Machine Learning
- **Modelo**: XGBoost (Gradient Boosting Classifier)
- **Objetivo**: Clasificación binaria (WIN=1 / LOSS=0)
- **Features**: 40+ características extraídas de indicadores técnicos
- **Entrenamiento automático**: Cada 20 trades nuevos
- **Predicción en tiempo real**: Probabilidad de éxito para cada señal

### ⚙️ Auto-Optimización
- **Ajusta parámetros** cada 20 trades basándose en performance
- **Parámetros optimizados**:
  - Flash threshold (umbral de señales flash)
  - Confidence threshold (confianza mínima)
  - Position size (tamaño de posición)
  - Max positions (posiciones simultáneas)
- **Estrategia adaptativa**:
  - Si **pierde** → Aumenta thresholds (más selectivo)
  - Si **gana bien** → Reduce thresholds (más señales)
  - Si **drawdown alto** → Reduce tamaño de posición
  - Si **performance excelente** → Aumenta tamaño de posición

---

## 📁 Estructura del Sistema

```
src/
├── ml/
│   ├── predictor.py          # Predicciones ML en tiempo real
│   ├── model_trainer.py      # Entrenamiento de XGBoost
│   ├── feature_engineer.py   # Creación de features ML
│   ├── optimizer.py          # Auto-optimización de parámetros
│   └── ml_integration.py     # Capa de integración ML
│
├── trading/
│   ├── portfolio.py          # Gestión de $50K USDT
│   ├── position_manager.py   # Apertura/cierre de posiciones
│   ├── risk_manager.py       # Gestión de riesgo
│   └── paper_trader.py       # Motor principal de paper trading
│
└── market_monitor.py         # Integración con señales de mercado

data/
├── trades/
│   └── portfolio.json        # Estado del portfolio
├── models/
│   ├── xgboost_model.pkl     # Modelo ML entrenado
│   └── model_metadata.json   # Métricas del modelo
├── optimization/
│   ├── optimized_params.json # Parámetros optimizados
│   └── optimization_history.json # Historial de ajustes
└── ml/
    └── training_buffer.json  # Features guardadas para entrenamiento
```

---

## 🔄 Flujo de Funcionamiento

### 1. **Análisis de Señal**
```
Mercado → Indicadores Técnicos → Señal (BUY/SELL)
                ↓
        Feature Engineer
                ↓
        40+ Features ML
```

### 2. **Predicción ML**
```
Features → XGBoost Model → Predicción
                               ↓
                    WIN probability: 73%
                    LOSS probability: 27%
                    Confidence: 73%
```

### 3. **Decisión de Trade**
```
IF ml_prediction = WIN AND confidence > 60%:
    → Calculate position size (1-10%)
    → Check risk limits
    → Execute trade
ELSE:
    → Block trade (ML warning)
```

### 4. **Gestión de Posición**
```
Trade abierto → Monitor precio actual
                      ↓
            ¿Alcanzó SL/TP?
                      ↓
               Cerrar trade
                      ↓
            Calcular P&L
                      ↓
        Guardar features + resultado
```

### 5. **Aprendizaje Continuo**
```
Cada 20 trades cerrados:
    → Cargar features guardadas
    → Preparar datos (X, y)
    → Entrenar nuevo modelo XGBoost
    → Evaluar (accuracy, precision, F1)
    → Guardar modelo actualizado
```

### 6. **Optimización Continua**
```
Cada 20 trades:
    → Analizar win rate, ROI, drawdown
    → Ajustar parámetros según performance
    → Guardar nueva configuración
```

---

## 📊 Features de Machine Learning

### Features Básicas (Indicadores)
- **RSI**: Valor, oversold/overbought, extreme levels
- **MACD**: Valor, señal, diferencia, bullish/bearish
- **EMAs**: Short (9), Medium (21), Long (50), trend alignment
- **Bollinger Bands**: Upper, middle, lower, width, position
- **Volume**: Ratio, high/low volume flags
- **ADX**: Trend strength, strong/weak trend flags
- **ATR**: Volatility
- **Divergencias**: RSI/MACD bullish/bearish

### Features Avanzadas (Combinaciones)
- **Momentum Score**: Combinación de RSI + MACD + Volume
- **Trend Strength**: ADX + EMA alignment + BB width
- **Signal Quality**: Score, confidence, strength de la señal

### Features Multi-Timeframe
- **1h/4h/1d trends**: Alineación de tendencias
- **MTF Alignment**: Concordancia entre timeframes

### Features de Contexto
- **Support/Resistance**: Distancia a soportes/resistencias
- **Price to EMA Ratio**: Posición del precio vs EMAs
- **Volatility %**: ATR como % del precio

**Total: 40+ features** procesadas para cada predicción.

---

## 📈 Métricas de Performance

### Trading Metrics
- **ROI**: Return on Investment (%)
- **Win Rate**: % de trades ganadores
- **Profit Factor**: Ganancias totales / Pérdidas totales
- **Sharpe Ratio**: Retorno ajustado por riesgo
- **Max Drawdown**: Pérdida máxima desde peak (%)
- **Avg Win/Loss**: Ganancia/Pérdida promedio por trade

### ML Model Metrics
- **Test Accuracy**: % de predicciones correctas
- **Test Precision**: % de predicciones WIN que fueron correctas
- **Test Recall**: % de trades WIN que fueron predichos
- **F1 Score**: Media armónica de precision y recall

### Risk Metrics
- **Risk Level**: LOW / MEDIUM / HIGH / CRITICAL
- **Current Drawdown**: Drawdown actual (%)
- **Positions Open**: Posiciones abiertas vs máximo
- **Available Balance**: USDT disponible para nuevos trades

---

## 🎮 Configuración

### Variables de Entorno

```bash
# Paper Trading
ENABLE_PAPER_TRADING=true
PAPER_TRADING_INITIAL_BALANCE=50000.0

# Señales Flash (optimizables automáticamente)
FLASH_THRESHOLD=5.0
FLASH_MIN_CONFIDENCE=50
```

### Parámetros Auto-Optimizables

Estos parámetros **se ajustan automáticamente** basados en performance:

```python
{
    'flash_threshold': 5.0,          # Rango: 4.0 - 7.0
    'flash_min_confidence': 50,      # Rango: 40 - 70
    'position_size_pct': 5.0,        # Rango: 1.0 - 10.0
    'max_positions': 10              # Rango: 5 - 15
}
```

---

## 💡 Estrategias de Optimización

### 1. **Threshold Optimization**

**Si win rate < 45%**:
```python
flash_threshold = min(current + 0.5, 7.0)  # Más selectivo
flash_min_confidence = min(current + 5, 70)
```

**Si win rate > 65% y profit_factor > 2.0**:
```python
flash_threshold = max(current - 0.3, 4.0)  # Más señales
flash_min_confidence = max(current - 5, 40)
```

### 2. **Position Sizing**

```python
position_size = base_size * score_factor * confidence_factor *
                drawdown_factor * winrate_factor

# Donde:
score_factor = signal.score / 10.0
confidence_factor = signal.confidence / 100.0
drawdown_factor = 1.0 si DD < 20%, sino max(0.3, 1.0 - DD/100)
winrate_factor = 1.2 si WR > 60%, 0.8 si WR < 40%, sino 1.0
```

**Resultado**: Position size entre **1% y 10%** del equity.

### 3. **Risk Reduction**

Se reduce riesgo automáticamente si:
- Max drawdown > 15%
- Win rate < 35% (con más de 10 trades)
- ROI < -5%

**Acciones**:
- Reducir position size
- Aumentar thresholds
- Pausar trades si drawdown > 30%

---

## 📱 Reportes y Notificaciones

### Telegram Stats (cada ~1 hora)

```
📊 PAPER TRADING STATS

💰 Balance: $51,234.56 USDT
💎 Equity: $52,100.00 USDT
📈 P&L: $2,100.00 (+4.20%)

📊 Trading:
• Total Trades: 45
• Win Rate: 62.2%
• Profit Factor: 1.85
• Sharpe Ratio: 1.42
• Max Drawdown: 3.5%

🔄 Posiciones Abiertas: 3

🧠 ML Model:
• Accuracy: 67.50%
• Precision: 72.30%
• Samples: 45

⚙️ Parámetros:
• Flash Threshold: 4.5
• Min Confidence: 55%
• Position Size: 6.2%
```

### Logs de Optimización

```
🤖 Auto-optimizer analizando performance...
✅ Optimización completada: 2 parámetros ajustados
   flash_threshold: 5.0 → 4.5 (Win rate alto (65.3%), generando más señales)
   position_size_pct: 5.0 → 6.0 (Performance excelente (ROI 8.5%), aumentando tamaño)
```

### Logs de Reentrenamiento

```
🧠 Iniciando reentrenamiento de modelo ML...
📊 Datos preparados: 45 samples | WIN: 28 | LOSS: 17
🧠 Entrenando modelo ML...
   Train: 36 samples | Test: 9 samples
✅ Modelo entrenado exitosamente!
   Train Accuracy: 0.889 | Test Accuracy: 0.778
   Test Precision: 0.857 | Test Recall: 0.750
💾 Modelo guardado: data/models/xgboost_model.pkl
```

---

## 🔍 Debugging y Monitoreo

### Ver Estado del Portfolio
```python
from src.trading.portfolio import Portfolio

portfolio = Portfolio()
stats = portfolio.get_statistics()
print(f"Balance: ${stats['current_balance']:,.2f}")
print(f"ROI: {stats['roi']:+.2f}%")
print(f"Win Rate: {stats['win_rate']:.1f}%")
```

### Ver Modelo ML
```python
from src.ml.model_trainer import ModelTrainer

trainer = ModelTrainer()
info = trainer.get_model_info()
if info['available']:
    print(f"Accuracy: {info['metrics']['test_accuracy']:.3f}")
    print(f"Samples: {info['metrics']['samples_total']}")
```

### Ver Parámetros Optimizados
```python
from src.ml.optimizer import AutoOptimizer

optimizer = AutoOptimizer()
params = optimizer.get_current_params()
history = optimizer.get_optimization_history(limit=5)

print("Parámetros actuales:", params)
print("Últimas optimizaciones:", history)
```

### Forzar Reentrenamiento
```python
from src.ml.ml_integration import MLIntegration

ml_system = MLIntegration()
ml_system.force_retrain()
```

### Forzar Optimización
```python
ml_system.force_optimize()
```

---

## 🚀 Próximos Pasos de Mejora

### 1. **Multi-Model Ensemble**
- Entrenar múltiples modelos (XGBoost + Random Forest + Neural Network)
- Combinar predicciones con voting/stacking
- Mayor robustez en predicciones

### 2. **Reinforcement Learning**
- Implementar Q-Learning o PPO
- El modelo aprende política de trading óptima
- Maximiza recompensa acumulada a largo plazo

### 3. **Sentiment Analysis**
- Analizar noticias de crypto
- Twitter sentiment analysis
- Fear & Greed Index integration

### 4. **Advanced Features**
- Order flow analysis
- Market microstructure features
- Cross-pair correlations
- Funding rates (futuros)

### 5. **Risk Management Avanzado**
- Kelly Criterion para position sizing
- VaR (Value at Risk) dinámico
- Correlation-based diversification
- Dynamic stop-loss trailing

### 6. **Backtesting Riguroso**
- Walk-forward analysis
- Out-of-sample testing
- Stress testing con crashes históricos
- Monte Carlo simulation

---

## 📚 Referencias

### Librerías Utilizadas
- **XGBoost**: https://xgboost.readthedocs.io/
- **Scikit-Learn**: https://scikit-learn.org/
- **Pandas**: https://pandas.pydata.org/
- **NumPy**: https://numpy.org/

### Conceptos de Trading
- **Sharpe Ratio**: Medida de retorno ajustado por riesgo
- **Profit Factor**: Ratio de ganancias vs pérdidas
- **Drawdown**: Pérdida máxima desde un peak
- **Kelly Criterion**: Fórmula para position sizing óptimo

### Machine Learning en Trading
- "Advances in Financial Machine Learning" - Marcos López de Prado
- "Machine Learning for Algorithmic Trading" - Stefan Jansen
- "Quantitative Trading" - Ernest Chan

---

## ⚠️ Advertencias

1. **Esto es PAPER TRADING**: No usa dinero real
2. **Resultados pasados NO garantizan resultados futuros**
3. **El mercado crypto es MUY volátil**: Drawdowns >50% son posibles
4. **La IA puede sobre-optimizar (overfitting)**: Monitorear métricas out-of-sample
5. **Antes de usar dinero real**:
   - Dejar correr el bot **varios meses** en paper trading
   - Verificar que ROI > 0% consistentemente
   - Verificar que Sharpe Ratio > 1.0
   - Verificar que Win Rate > 55%
   - Probar con diferentes condiciones de mercado

---

## 🎓 Cómo Aprende el Sistema

### Ciclo de Aprendizaje

```
1. PREDICCIÓN
   - Recibe señal con 40+ features
   - Modelo predice probabilidad de WIN
   - Si confianza > 60% → Ejecuta trade

2. EJECUCIÓN
   - Abre posición con tamaño dinámico
   - Guarda features en buffer
   - Monitorea SL/TP

3. CIERRE
   - Alcanza SL o TP
   - Calcula P&L
   - Marca trade como WIN/LOSS

4. ALMACENAMIENTO
   - Guarda: features + resultado (WIN/LOSS)
   - Acumula datos para entrenamiento

5. REENTRENAMIENTO (cada 20 trades)
   - Carga trades + features
   - Divide en train/test (80/20)
   - Entrena XGBoost
   - Evalúa métricas
   - Si mejora → Usa nuevo modelo

6. OPTIMIZACIÓN (cada 20 trades)
   - Analiza performance (win rate, ROI, DD)
   - Ajusta parámetros según reglas
   - Guarda nueva configuración

7. VUELTA AL PASO 1
   - Con modelo más inteligente
   - Con parámetros optimizados
   - Aprende continuamente
```

### Ejemplo de Evolución

**Día 1-3** (Primeros 50 trades):
- Win Rate: 45%
- ROI: -2.3%
- Modelo: No disponible (insuficientes datos)
- **Resultado**: Perdiendo ligeramente

**Día 4-7** (Trades 50-100):
- Win Rate: 52%
- ROI: +1.8%
- Modelo: Primera versión entrenada (accuracy: 58%)
- **Optimización**: Flash threshold 5.0 → 5.5 (más selectivo)
- **Resultado**: Mejorando

**Día 8-14** (Trades 100-200):
- Win Rate: 59%
- ROI: +5.2%
- Modelo: 3ra versión (accuracy: 65%, precision: 68%)
- **Optimización**: Position size 5% → 6% (buen performance)
- **Resultado**: Ganando consistentemente

**Día 15-30** (Trades 200-400):
- Win Rate: 63%
- ROI: +12.7%
- Modelo: 8va versión (accuracy: 69%, precision: 73%)
- **Optimización**: Flash threshold 5.5 → 5.2 (más señales)
- **Resultado**: Performance estable y positivo

**Objetivo Final** (3-6 meses):
- Win Rate: 65-70%
- ROI: +20-40%
- Sharpe Ratio: >1.5
- Modelo robusto con 500+ samples

---

## ✅ Checklist de Validación

Antes de considerar el sistema "listo para real trading":

- [ ] **100+ trades** ejecutados en paper trading
- [ ] **Win rate > 60%** consistente por al menos 2 semanas
- [ ] **ROI > +10%** en paper trading
- [ ] **Sharpe Ratio > 1.0**
- [ ] **Max Drawdown < 15%**
- [ ] **Profit Factor > 1.5**
- [ ] **Modelo ML** con accuracy > 65%
- [ ] **Optimizaciones** funcionando correctamente
- [ ] **Sin crashes** por al menos 1 semana
- [ ] **Tested en bear market** (mercado bajista)
- [ ] **Tested en bull market** (mercado alcista)
- [ ] **Tested en sideways market** (mercado lateral)

---

**Creado**: 2025-01-31
**Versión**: 1.0
**Autor**: Claude AI + Usuario
**Estado**: Sistema completo y funcional en Paper Trading
