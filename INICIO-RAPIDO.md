# 🚀 Inicio Rápido - Bot de Señales de Trading

**Para personas sin experiencia en programación**

---

## ⚡ Pasos Rápidos

### 1️⃣ Crear tu Bot de Telegram

1. Abre Telegram en tu celular
2. Busca: `@BotFather`
3. Envía: `/newbot`
4. Elige un nombre para tu bot
5. Elige un username (debe terminar en `bot`)
6. **COPIA EL TOKEN** que te da (algo como: `123456:ABC-DEF1234...`)

### 2️⃣ Obtener tu Chat ID

1. En Telegram, busca: `@userinfobot`
2. Presiona START
3. **COPIA TU ID** (un número como: `123456789`)

### 3️⃣ Configurar el Bot

1. Abre la carpeta `TRADING-BOT`
2. Copia el archivo `.env.example` y renómbralo a `.env`
3. Abre `.env` con un editor de texto (Bloc de notas, etc.)
4. Pega tu TOKEN y CHAT_ID donde dice:
   ```
   TELEGRAM_BOT_TOKEN=aquí_tu_token
   TELEGRAM_CHAT_ID=aquí_tu_chat_id
   ```
5. Guarda el archivo

### 4️⃣ Instalar Python

**Windows:**
1. Ve a: https://www.python.org/downloads/
2. Descarga Python 3.8 o superior
3. **IMPORTANTE:** Marca la casilla "Add Python to PATH" al instalar

**Mac:**
Ya viene instalado. Verifica en Terminal:
```bash
python3 --version
```

**Linux:**
```bash
sudo apt update
sudo apt install python3 python3-pip
```

### 5️⃣ Instalar Dependencias

**Windows:**
1. Abre Command Prompt (cmd)
2. Ve a la carpeta del bot:
   ```
   cd ruta\a\TRADING-BOT
   ```
3. Instala las librerías:
   ```
   pip install -r requirements.txt
   ```

**Mac/Linux:**
```bash
cd /ruta/a/TRADING-BOT
pip3 install -r requirements.txt
```

### 6️⃣ Ejecutar el Bot

**Opción Fácil (Windows):**
- Doble clic en `run.bat`

**Opción Fácil (Mac/Linux):**
```bash
./run.sh
```

**Opción Manual:**
```bash
python main.py
```
o
```bash
python3 main.py
```

---

## ✅ ¿Cómo sé que funciona?

Deberías recibir un mensaje en Telegram que dice:

```
🤖 Bot de Señales Iniciado

📊 Monitoreando: BTC/USDT, ETH/USDT, BNB/USDT, XRP/USDT, SOL/USDT y 5 más
⏱️ Intervalo: 180s
📈 Timeframe: 1h
📍 Reporte diario: 9 PM CDMX
```

---

## 📊 ¿Qué Hace el Bot?

### Análisis Continuo
- Analiza 10+ criptomonedas cada 3 minutos
- Usa indicadores técnicos profesionales (RSI, MACD, EMAs, Bollinger)
- Identifica oportunidades de compra y venta

### Notificaciones Inteligentes
Cuando detecta una oportunidad fuerte, te envía:

```
🟢 SEÑAL DE TRADING 🟢

Par: BTC/USDT
Acción: COMPRAR
Fuerza: ⭐⭐⭐⭐

💰 Precio: $45,230.50

📊 Indicadores:
• RSI: 28.45
• MACD: Cruce alcista
• EMAs: Tendencia alcista

📈 Razones:
• RSI oversold (28.45)
• MACD bullish crossover
• Price below lower Bollinger Band
```

### Reporte Diario (9 PM CDMX)
Cada día a las 9 PM recibirás un reporte completo:

```
📊 REPORTE DIARIO DE TRADING
📅 Fecha: 29/10/2025
━━━━━━━━━━━━━━━━━━━━━━━━━

📈 RESUMEN DEL DÍA
• Señales enviadas: 15
• ✅ Exitosas: 10
• ❌ Fallidas: 3
• ⏳ Pendientes: 2
• 🎯 Precisión: 76.9%
• 💰 Ganancia promedio: +3.45%

🏆 MEJOR SEÑAL:
  BTC/USDT - BUY
  Ganancia: +5.23%
  Precio: $45,230.50 → $47,595.88

📊 POR PAR:
  • BTC/USDT: 5/7 (71%)
  • ETH/USDT: 3/4 (75%)
  • SOL/USDT: 2/2 (100%)
  ...
```

---

## ⚙️ Configuración Avanzada (Opcional)

Puedes editar el archivo `config/config.py` para:

- Cambiar los pares a monitorear
- Ajustar el intervalo de análisis
- Modificar umbrales de indicadores
- Cambiar la hora del reporte diario

---

## 🆘 Problemas Comunes

### ❌ "TELEGRAM_BOT_TOKEN not configured"
**Solución:** Verifica que creaste el archivo `.env` (sin .example) y pegaste tu token correctamente.

### ❌ No recibo mensajes en Telegram
**Solución:**
1. Busca tu bot en Telegram (el username que creaste)
2. Presiona START
3. Reinicia el bot de trading

### ❌ "No module named telegram"
**Solución:**
```bash
pip install python-telegram-bot
```

### ❌ "Symbol MXN/USD not available"
**Solución:** Es normal. No todos los exchanges tienen este par. El bot seguirá funcionando con los demás.

---

## 🔒 Importante

- ⚠️ Este bot **NO opera automáticamente**. Solo envía señales.
- ⚠️ Las señales son sugerencias, **NO consejos de inversión**.
- ⚠️ Siempre haz tu propia investigación antes de operar.
- ⚠️ Nunca inviertas más de lo que puedes perder.

---

## 💡 Consejos

1. **Deja el bot corriendo 24/7** para no perderte señales
2. **Revisa el reporte diario** para evaluar el rendimiento
3. **Empieza con paper trading** (práctica) antes de usar dinero real
4. **Combina las señales con tu propio análisis**

---

## 📞 Ayuda

Si tienes problemas:

1. Lee el archivo `TUTORIAL.md` (guía completa)
2. Revisa `README.md` (documentación técnica)
3. Revisa los logs en `logs/trading_bot.log`

---

**¡Listo! Tu bot está monitoreando el mercado 24/7 🚀**
