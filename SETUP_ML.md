# 🚀 Setup - ML Paper Trading System

Guía de instalación y despliegue del sistema de Machine Learning + Paper Trading.

---

## 📋 Requisitos Previos

1. **Python 3.9+** instalado
2. **Cuenta de Telegram** con bot token
3. **Proxy configurado** (Webshare.io, Smartproxy, etc.) - Ver [PROXY_SETUP.md](PROXY_SETUP.md)
4. **Cuenta de Railway** (para despliegue en la nube)

---

## 🛠️ Instalación Local (Desarrollo)

### 1. Clonar el Repositorio

```bash
git clone https://github.com/novacoinotc/TRADING-BOT.git
cd TRADING-BOT
```

### 2. Crear Entorno Virtual

```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 3. Instalar Dependencias

```bash
pip install -r requirements.txt
```

**Nota**: Esto instalará:
- `xgboost==2.0.3` - Machine Learning
- `scikit-learn==1.4.0` - Métricas ML
- `pandas`, `numpy` - Procesamiento de datos
- `ccxt` - Exchange API
- `python-telegram-bot` - Notificaciones
- Y más...

### 4. Configurar Variables de Entorno

Crea un archivo `.env` en la raíz del proyecto:

```bash
# Telegram
TELEGRAM_BOT_TOKEN=tu_bot_token_aqui
TELEGRAM_CHAT_ID=tu_chat_id_aqui

# Exchange (No necesitas API keys para paper trading, pero déjalas vacías)
EXCHANGE_NAME=binance
EXCHANGE_API_KEY=
EXCHANGE_API_SECRET=

# Proxy (Obligatorio para Binance)
USE_PROXY=true
PROXY_HOST=proxy.webshare.io
PROXY_PORT=80
PROXY_USERNAME=tu_usuario
PROXY_PASSWORD=tu_password

# Paper Trading (NUEVO)
ENABLE_PAPER_TRADING=true
PAPER_TRADING_INITIAL_BALANCE=50000.0

# Señales Flash
ENABLE_FLASH_SIGNALS=true
FLASH_THRESHOLD=5.0
FLASH_MIN_CONFIDENCE=50

# Intervalo de Chequeo (segundos)
CHECK_INTERVAL=120
```

### 5. Crear Directorios de Datos

```bash
mkdir -p data/trades
mkdir -p data/models
mkdir -p data/optimization
mkdir -p data/ml
mkdir -p data/training
mkdir -p logs
```

### 6. Ejecutar Bot Localmente

```bash
python main.py
```

Deberías ver:

```
🤖 Paper Trading Engine iniciado
💰 Balance inicial: $50,000.00 USDT
🧠 ML Predictor inicializado (sin modelo entrenado aún)
🤖 Bot de Señales Iniciado
💰 Paper Trading: ✅ Activo ($50,000 USDT)
🧠 Machine Learning: ✅ Activo
```

---

## ☁️ Despliegue en Railway

### 1. Preparar Railway

1. Ve a [Railway.app](https://railway.app)
2. Conecta tu cuenta de GitHub
3. Crea un nuevo proyecto → "Deploy from GitHub repo"
4. Selecciona `novacoinotc/TRADING-BOT`

### 2. Configurar Variables de Entorno en Railway

En el dashboard de Railway, agrega todas las variables del `.env`:

```
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
EXCHANGE_NAME=binance
EXCHANGE_API_KEY=
EXCHANGE_API_SECRET=
USE_PROXY=true
PROXY_HOST=proxy.webshare.io
PROXY_PORT=80
PROXY_USERNAME=...
PROXY_PASSWORD=...
ENABLE_PAPER_TRADING=true
PAPER_TRADING_INITIAL_BALANCE=50000.0
ENABLE_FLASH_SIGNALS=true
FLASH_THRESHOLD=5.0
FLASH_MIN_CONFIDENCE=50
CHECK_INTERVAL=120
```

### 3. Configurar Región

Railway → Settings → Región → **EU West (Amsterdam)**

(Para reducir latencia con Binance)

### 4. Deploy

Railway detectará automáticamente `nixpacks.toml` y construirá el proyecto.

**Tiempo de build**: ~3-5 minutos
**Logs**: Monitorea en Railway Dashboard → Deployments → View Logs

### 5. Verificar Estado

Deberías recibir un mensaje en Telegram:

```
🤖 Bot de Señales Iniciado

📊 Monitoreando: BTC/USDT, ETH/USDT, SOL/USDT y 36 más
⏱️ Intervalo: 120s
📈 Timeframe conservador: 1h (1h/4h/1d)
⚡ Señales flash: ✅ Activas (15m)
💰 Paper Trading: ✅ Activo ($50,000 USDT)
🧠 Machine Learning: ✅ Activo
📍 Reporte diario: 9 PM CDMX
```

---

## 📊 Verificar Funcionamiento

### 1. Primeras Señales

Espera ~5-10 minutos. Deberías recibir señales en Telegram:

```
🟢 SEÑAL DE TRADING FUERTE 🟢

Par: SOL/USDT
Acción: COMPRAR
💎 Calidad: 7.8/10 (ALTA)
Fuerza: ⭐⭐⭐⭐

💰 Precio actual: $142.35
📍 Entrada sugerida: $141.64 - $143.06

🎯 Take Profit:
   TP1: $145.20 (+2.0%)
   TP2: $148.50 (+4.3%)
   TP3: $152.00 (+6.8%)

🛡️ Stop Loss: $138.50 (-2.7%)
📊 Ratio R/R: 1:2.5

📊 Indicadores (7.8 pts):
• RSI: 28.3 (sobreventa fuerte)
• MACD: Alcista ✅
• Volumen: +45% 🔥

⚠️ Riesgo: MEDIUM
💡 Confianza: 78%
```

### 2. Trades Ejecutados

Verás logs como:

```
📊 Trade abierto: SOL/USDT BUY | Size: 5.8% | Score: 7.8/10 | Confidence: 78%
```

### 3. Stats Cada Hora

Cada ~1 hora recibirás:

```
📊 PAPER TRADING STATS

💰 Balance: $49,234.00 USDT
💎 Equity: $50,100.00 USDT
📈 P&L: $100.00 (+0.20%)

📊 Trading:
• Total Trades: 5
• Win Rate: 60.0%
• Profit Factor: 1.35
...
```

### 4. Primer Reentrenamiento ML

Después de ~50 trades (varios días), verás:

```
🧠 Iniciando reentrenamiento de modelo ML...
📊 Datos preparados: 52 samples | WIN: 31 | LOSS: 21
✅ Modelo entrenado exitosamente!
   Train Accuracy: 0.857 | Test Accuracy: 0.727
💾 Modelo guardado
```

### 5. Primera Optimización

Después de 20 trades:

```
🤖 Auto-optimizer analizando performance...
✅ Optimización completada: 1 parámetros ajustados
   flash_threshold: 5.0 → 5.5 (Win rate bajo (42.3%), aumentando selectividad)
```

---

## 🔧 Mantenimiento

### Ver Logs en Railway

```
Railway Dashboard → Deployments → View Logs
```

Busca:
- `📊 Trade abierto` - Trades ejecutados
- `✅ Trade cerrado` - Trades cerrados con P&L
- `🧠 Iniciando reentrenamiento` - ML training
- `🤖 Auto-optimizer` - Optimizaciones

### Descargar Datos Localmente

Para analizar:

```bash
# Railway CLI
railway run bash

# Dentro del contenedor
cat data/trades/portfolio.json
cat data/models/model_metadata.json
cat data/optimization/optimized_params.json
```

O usa Railway's File Browser (si disponible).

### Actualizar Código

```bash
# Local
git pull origin main

# Railway se actualizará automáticamente en el próximo commit a main
```

---

## 🐛 Troubleshooting

### Error: "XGBoost no instalado"

**Solución**:
```bash
pip install xgboost scikit-learn
```

### Error: "Binance 451"

**Causa**: Proxy no configurado o inválido

**Solución**:
1. Verifica variables de proxy en `.env` o Railway
2. Verifica que proxy funcione: https://www.webshare.io/dashboard
3. Prueba cambiar PROXY_PORT de 80 a 443

### Error: "Insuficientes muestras para entrenar"

**Normal**: Necesitas al menos 50 trades cerrados para entrenar modelo

**Solución**: Espera varios días/semanas

### Error: "División por cero" en Bollinger Bands

**Solución**: Ya corregido en última versión. Si persiste:
```bash
git pull origin main
```

### Bot se detiene después de unas horas

**Causa**: Error no manejado o límite de rate

**Solución**:
1. Revisa logs para el error específico
2. Verifica que `CHECK_INTERVAL >= 120` (no menos de 2 minutos)
3. Railway auto-reinicia, pero puedes forzar redeploy

---

## 📈 Optimización de Performance

### 1. Reducir Uso de API

Si alcanzas límites de rate de Binance:

```bash
# En .env
CHECK_INTERVAL=180  # 3 minutos en lugar de 2
```

### 2. Reducir Pares Monitoreados

Edita `config/config.py`:

```python
TRADING_PAIRS = [
    'BTC/USDT',
    'ETH/USDT',
    'SOL/USDT',
    # ... solo los que quieras
]
```

### 3. Ajustar Thresholds Iniciales

Para señales más selectivas desde el inicio:

```bash
# En .env
FLASH_THRESHOLD=6.0  # Más selectivo (menos señales, mayor calidad)
FLASH_MIN_CONFIDENCE=60  # Mayor confianza requerida
```

### 4. Ajustar Balance Inicial

Para simular trading con menos capital:

```bash
PAPER_TRADING_INITIAL_BALANCE=10000.0  # $10K en lugar de $50K
```

---

## 📚 Archivos Importantes

- **ML_SYSTEM.md** - Documentación completa del sistema ML
- **PROXY_SETUP.md** - Guía de configuración de proxy
- **README.md** - Documentación general del bot
- **config/config.py** - Configuración principal
- **main.py** - Punto de entrada
- **requirements.txt** - Dependencias Python

---

## 🎯 Próximos Pasos

1. **Dejar correr el bot 24/7** durante al menos 1-2 semanas
2. **Monitorear stats diarias** vía Telegram
3. **Esperar a tener 100+ trades** para evaluar performance real
4. **Revisar logs de optimización** para ver cómo aprende
5. **Evaluar métricas finales**:
   - Win Rate > 60%
   - ROI > +10%
   - Sharpe Ratio > 1.0
   - Max Drawdown < 15%

Si el sistema cumple estos criterios **consistentemente durante 1-3 meses**, se puede considerar para trading real (con MUCHO cuidado y empezando con cantidades pequeñas).

---

## ⚠️ Advertencia Final

Este sistema usa **DINERO VIRTUAL** (paper trading). No arriesga capital real.

Antes de usar dinero real:
- ✅ Deja correr en paper trading por al menos 3-6 meses
- ✅ Verifica performance consistente en diferentes condiciones de mercado
- ✅ Entiende completamente cómo funciona el sistema
- ✅ Acepta que puedes perder TODO el capital invertido
- ✅ Nunca inviertas más de lo que puedes permitirte perder
- ✅ Consulta con un asesor financiero

**El trading de criptomonedas es de ALTÍSIMO RIESGO.**

---

**¡Buena suerte con tu trading bot! 🚀**
