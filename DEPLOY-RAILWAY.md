# 🚀 Deploy en Railway.app - Guía Paso a Paso

Esta guía te mostrará cómo desplegar tu bot de señales en Railway para que funcione 24/7 en la nube.

---

## 📋 Lo que Necesitas

- Tu token de Telegram Bot (ya lo tienes)
- Tu Chat ID de Telegram (ya lo tienes)
- Una cuenta de GitHub (la que ya usas)
- 5 minutos de tu tiempo

---

## 🎯 Paso 1: Crear Cuenta en Railway

1. Ve a: **https://railway.app**

2. Click en **"Login"** (arriba a la derecha)

3. Selecciona **"Login with GitHub"**

4. Autoriza Railway para acceder a tu GitHub

5. ✅ **¡Listo!** Ya tienes cuenta (fue 1 click)

---

## 🔗 Paso 2: Conectar tu Repositorio

1. En Railway, click en **"New Project"**

2. Selecciona **"Deploy from GitHub repo"**

3. Busca y selecciona: **`TRADING-BOT`**

4. Railway detectará automáticamente que es un proyecto Python

5. Click en **"Deploy Now"**

⏳ Espera 1-2 minutos mientras Railway prepara todo...

---

## ⚙️ Paso 3: Configurar Variables de Entorno

**IMPORTANTE:** El bot necesita tus credenciales de Telegram.

1. En Railway, click en tu proyecto recién creado

2. Ve a la pestaña **"Variables"** (en el menú lateral)

3. Click en **"New Variable"** y agrega estas dos:

   **Variable 1:**
   ```
   Name: TELEGRAM_BOT_TOKEN
   Value: 8499904153:AAHdRcxNX3NvEXOH6GVCwWXrgUP5kM6C81U
   ```

   **Variable 2:**
   ```
   Name: TELEGRAM_CHAT_ID
   Value: TU_CHAT_ID_AQUI
   ```

   (Reemplaza `TU_CHAT_ID_AQUI` con el número que obtuviste de @userinfobot)

4. **Opcional** - Puedes agregar más variables si quieres personalizar:
   ```
   CHECK_INTERVAL=180
   EXCHANGE_NAME=binance
   RSI_OVERSOLD=30
   RSI_OVERBOUGHT=70
   ```

5. Click en **"Save"**

---

## 🚀 Paso 4: Desplegar el Bot

1. Railway automáticamente re-desplegará con las nuevas variables

2. Ve a la pestaña **"Deployments"**

3. Deberías ver un deployment en proceso

4. Espera a que el estado sea **"SUCCESS"** (1-2 minutos)

---

## ✅ Paso 5: Verificar que Funciona

1. **En Railway:**
   - Ve a la pestaña **"Logs"**
   - Deberías ver mensajes como:
     ```
     Trading Signal Bot Starting...
     Connected to binance
     Starting market monitor...
     ```

2. **En Telegram:**
   - Abre tu bot: `@NovacoinOTCbot`
   - Deberías recibir el mensaje de inicio:
     ```
     🤖 Bot de Señales Iniciado

     📊 Monitoreando: BTC/USDT, ETH/USDT, BNB/USDT...
     ⏱️ Intervalo: 180s
     📈 Timeframe: 1h
     📍 Reporte diario: 9 PM CDMX
     ```

3. **Si NO recibes el mensaje:**
   - Ve a Telegram
   - Busca `@NovacoinOTCbot`
   - Presiona el botón **START**
   - En Railway, ve a "Deployments" y click en **"Restart"**

---

## 📊 Monitorear tu Bot

### Ver Logs en Tiempo Real

1. En Railway, click en tu proyecto
2. Ve a **"Logs"** en el menú lateral
3. Verás todo lo que hace el bot en tiempo real

### Verificar que está Corriendo

- En la pestaña **"Metrics"** puedes ver:
  - CPU usage
  - Memory usage
  - Uptime (tiempo que lleva corriendo)

---

## 💰 Costos

Railway te da **$5 USD de crédito gratis cada mes**.

Tu bot usa aproximadamente:
- **~$3-4 USD/mes** (bien dentro del límite gratuito)

El crédito se renueva automáticamente cada mes. ✅ **Es gratis para siempre.**

---

## 🔧 Comandos Útiles

### Reiniciar el Bot
1. Ve a "Deployments"
2. Click en el deployment activo
3. Click en **"Restart"**

### Detener el Bot
1. Ve a "Settings"
2. Scroll hasta abajo
3. Click en **"Remove Service"**

### Actualizar el Código
1. Haz cambios en GitHub
2. Haz commit y push
3. Railway automáticamente detecta los cambios y re-despliega

---

## 🆘 Solución de Problemas

### El bot no se despliega
- Verifica los logs en la pestaña "Logs"
- Asegúrate de haber agregado las variables de entorno
- Intenta hacer "Restart"

### No recibo mensajes en Telegram
1. Abre `@NovacoinOTCbot` en Telegram
2. Presiona START
3. Verifica que el `TELEGRAM_CHAT_ID` sea correcto
4. Reinicia el deployment en Railway

### Error "TELEGRAM_BOT_TOKEN not configured"
- Ve a Variables en Railway
- Verifica que `TELEGRAM_BOT_TOKEN` esté bien escrito
- Verifica que el token sea correcto
- Guarda y reinicia

### El bot se detiene después de un tiempo
- Ve a Logs para ver el error
- Puede ser un problema de API rate limits
- Aumenta `CHECK_INTERVAL` a 300 (5 minutos)

---

## 📈 Próximos Pasos

Una vez que el bot esté corriendo:

1. **Observa las señales** que te envía durante el día
2. **Espera al reporte de las 9 PM** para ver estadísticas
3. **Revisa los logs en Railway** si quieres ver más detalles
4. **Ajusta la configuración** editando las variables de entorno

---

## 🎉 ¡Listo!

Tu bot está corriendo 24/7 en la nube. Recibirás:
- ✅ Notificaciones de señales en tiempo real
- ✅ Reportes diarios a las 9 PM CDMX
- ✅ Análisis continuo del mercado

**Sin necesidad de tener tu computadora encendida.**

---

## 🔗 Enlaces Útiles

- Railway Dashboard: https://railway.app/dashboard
- Documentación Railway: https://docs.railway.app
- Telegram Bot API: https://core.telegram.org/bots/api

---

**¿Tienes preguntas?** Revisa los logs en Railway o consulta el archivo `README.md`
