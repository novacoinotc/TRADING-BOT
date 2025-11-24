# 📱 Tutorial: Cómo Crear tu Bot de Telegram - Paso a Paso

Esta guía está diseñada para personas sin experiencia en programación. Sigue cada paso exactamente como se indica.

---

## Parte 1: Crear tu Bot de Telegram

### Paso 1: Abrir Telegram en tu celular o computadora

1. Abre la aplicación de Telegram
2. Si no tienes Telegram, descárgalo de:
   - iPhone: App Store
   - Android: Google Play Store
   - Computadora: https://telegram.org/apps

### Paso 2: Buscar a BotFather

1. En Telegram, toca el ícono de búsqueda (🔍) arriba
2. Escribe: `@BotFather`
3. Selecciona el bot verificado con una palomita azul ✓

### Paso 3: Crear tu Bot

1. Toca el botón **START** o **INICIAR**
2. Escribe el comando: `/newbot` y envíalo
3. BotFather te preguntará el nombre de tu bot
   - Ejemplo: `Mi Bot de Trading`
   - Escribe el nombre que quieras y envíalo
4. BotFather te pedirá un **username** (nombre de usuario)
   - **IMPORTANTE**: Debe terminar en `bot`
   - Ejemplo: `MiBotTrading_bot` o `señales_trading_bot`
   - Escribe tu username y envíalo

### Paso 4: Guardar tu Token

BotFather te enviará un mensaje como este:

```
Done! Congratulations on your new bot. You will find it at t.me/TuBot_bot
You can now add a description...

Use this token to access the HTTP API:
1234567890:ABCdefGHIjklMNOpqrsTUVwxyz-1234567890

For a description of the Bot API, see this page: https://core.telegram.org/bots/api
```

**COPIA EL TOKEN** (los números y letras después de "Use this token")
- Ejemplo: `1234567890:ABCdefGHIjklMNOpqrsTUVwxyz-1234567890`
- Guárdalo en un lugar seguro (Notas, bloc de notas, etc.)

### Paso 5: Obtener tu Chat ID

1. En Telegram, busca: `@userinfobot`
2. Toca **START** o **INICIAR**
3. El bot te enviará tu información
4. **COPIA tu ID** (es un número como `123456789`)
5. Guárdalo junto con tu token

---

## Parte 2: Configurar tu Bot de Trading

### Paso 6: Crear el archivo .env

1. Abre el archivo `.env.example` que está en la carpeta TRADING-BOT
2. Verás algo como esto:

```
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

3. Crea un **NUEVO ARCHIVO** llamado `.env` (sin el .example)
4. Copia el contenido y reemplaza:
   - `your_bot_token_here` → Pega tu TOKEN del Paso 4
   - `your_chat_id_here` → Pega tu CHAT ID del Paso 5

**Ejemplo de cómo debe quedar:**

```
TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz-1234567890
TELEGRAM_CHAT_ID=123456789
CHECK_INTERVAL=180
EXCHANGE_NAME=binance
RSI_OVERSOLD=30
RSI_OVERBOUGHT=70
```

5. **GUARDA** el archivo

---

## Parte 3: Instalar Python y Dependencias

### Paso 7: Verificar si tienes Python

**En Windows:**
1. Presiona `Windows + R`
2. Escribe `cmd` y presiona Enter
3. Escribe: `python --version`
4. Si ves algo como "Python 3.8" o superior, ¡genial!
5. Si no, descarga Python de: https://www.python.org/downloads/
   - **IMPORTANTE**: Durante la instalación, marca la casilla "Add Python to PATH"

**En Mac:**
1. Abre Terminal (Aplicaciones → Utilidades → Terminal)
2. Escribe: `python3 --version`
3. Si ves "Python 3.8" o superior, ¡perfecto!
4. Si no, instala desde: https://www.python.org/downloads/

**En Linux:**
```bash
python3 --version
# Si no está instalado:
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

### Paso 8: Instalar las librerías necesarias

**En Windows:**
1. Abre Command Prompt (cmd)
2. Navega a la carpeta del bot:
   ```
   cd ruta\a\TRADING-BOT
   ```
3. Ejecuta:
   ```
   python -m pip install -r requirements.txt
   ```

**En Mac/Linux:**
1. Abre Terminal
2. Navega a la carpeta:
   ```bash
   cd /ruta/a/TRADING-BOT
   ```
3. Ejecuta:
   ```bash
   pip3 install -r requirements.txt
   ```

---

## Parte 4: Ejecutar tu Bot

### Paso 9: Iniciar el Bot de Señales

**Opción 1 - Comando Simple (Windows):**
```
python main.py
```

**Opción 2 - Comando Simple (Mac/Linux):**
```bash
python3 main.py
```

**Opción 3 - Usando el script (Mac/Linux):**
```bash
chmod +x run.sh
./run.sh
```

### Paso 10: Verificar que funciona

1. Deberías ver mensajes en la pantalla como:
   ```
   Trading Signal Bot Starting...
   Connected to binance
   Starting market monitor...
   ```

2. En tu Telegram, deberías recibir un mensaje del bot:
   ```
   🤖 Bot de Señales Iniciado

   Monitoreando: BTC/USDT, ETH/USDT, ...
   Intervalo: 180s
   Timeframe: 1h
   ```

3. **¡Listo!** Tu bot está funcionando

---

## Parte 5: Mantener el Bot Funcionando 24/7

### Para que funcione todo el día:

**Opción A - En tu computadora:**
- Deja la ventana de comando abierta
- No cierres la aplicación
- No apagues tu computadora

**Opción B - En un servidor (Recomendado):**
1. Usa un servicio como:
   - **Raspberry Pi** (si tienes uno)
   - **AWS Free Tier** (gratis 12 meses)
   - **Google Cloud** (gratis $300 crédito)
   - **VPS económico** ($5/mes)

2. Copia todos los archivos al servidor
3. Usa `screen` o `tmux` para mantenerlo corriendo:
   ```bash
   screen -S trading_bot
   python3 main.py
   # Presiona Ctrl+A, luego D para "despegar" la sesión
   ```

---

## ❓ Solución de Problemas Comunes

### Problema: "No module named telegram"
**Solución:**
```bash
pip install python-telegram-bot
```

### Problema: "TELEGRAM_BOT_TOKEN not configured"
**Solución:**
- Verifica que el archivo `.env` exista (sin .example)
- Verifica que pegaste correctamente el token

### Problema: No recibo mensajes en Telegram
**Solución:**
1. Abre Telegram
2. Busca tu bot (el username que creaste)
3. Presiona **START** o **INICIAR**
4. Reinicia el bot de trading

### Problema: "Symbol MXN/USD not available"
**Solución:**
- Es normal, no todos los exchanges tienen MXN/USD
- El bot seguirá funcionando con los otros pares

---

## 📊 Entendiendo las Señales

Cuando el bot detecte una oportunidad, recibirás un mensaje como:

```
🟢 SEÑAL DE TRADING 🟢

Par: BTC/USDT
Acción: COMPRAR
Fuerza: ⭐⭐⭐⭐

💰 Precio: $45,230.50

📊 Indicadores:
• RSI: 28.45 (sobreventa)
• MACD: Cruce alcista
• EMAs: Tendencia alcista

📈 Razones:
• RSI oversold (28.45)
• MACD bullish crossover
• Price below lower Bollinger Band
```

**¿Qué significa?**
- 🟢 = Oportunidad de COMPRA
- 🔴 = Oportunidad de VENTA
- ⭐⭐⭐⭐ = Señal fuerte (más estrellas = más confiable)
- Razones = Por qué el bot sugiere esta acción

---

## 🔒 Seguridad Importante

1. **NUNCA compartas tu Token de Telegram**
2. **NUNCA subas el archivo `.env` a internet**
3. **Estas son solo señales**, no consejos de inversión
4. **Siempre haz tu propia investigación** antes de invertir
5. **Empieza con cantidades pequeñas** para probar

---

## 📞 ¿Necesitas Ayuda?

Si tienes problemas:
1. Lee los mensajes de error en la ventana de comando
2. Revisa los logs en: `logs/trading_bot.log`
3. Verifica que todos los pasos se siguieron correctamente

---

## 🎉 ¡Felicidades!

Tu bot de señales de trading está funcionando. Recibirás notificaciones automáticas cuando el bot detecte oportunidades en el mercado.

**Recuerda**: Este bot es una herramienta de análisis. Las decisiones finales de trading siempre deben ser tuyas después de tu propia investigación.

---

**Última actualización:** 2025-10-29
