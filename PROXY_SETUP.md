# 🔐 Configuración de Proxy para Binance

Este bot ahora usa **Binance** como exchange principal para aprovechar su **alta liquidez** (10-100x más que Kraken).

## ⚠️ ¿Por Qué Necesitas un Proxy?

Binance **bloquea IPs de datacenters** (como Railway, Heroku, AWS, etc.) con error **451 - Restricted Location**.

**Solución**: Usar un proxy con IP residencial.

---

## 📋 Paso 1: Contratar Servicio de Proxy

Recomendamos estos servicios (elige UNO):

### Opción A: Webshare.io (Recomendado - Más Económico)
- **Precio**: $2.99/mes (10 proxies)
- **Velocidad**: Rápido
- **Setup**: Muy fácil
- **Link**: https://www.webshare.io/

**Pasos**:
1. Crear cuenta en https://www.webshare.io/
2. Ir a "Proxy" → "Proxy List"
3. Copiar credenciales del primer proxy:
   - **Host**: Ejemplo: `proxy.webshare.io` o `p.webshare.io`
   - **Port**: Ejemplo: `80` o `443`
   - **Username**: Tu username
   - **Password**: Tu password

---

### Opción B: Smartproxy
- **Precio**: $12.50/mes (8GB)
- **Velocidad**: Muy rápido
- **Link**: https://smartproxy.com/

**Pasos**:
1. Crear cuenta
2. Dashboard → Proxies → HTTP(S) Proxies
3. Copiar:
   - Host: `gate.smartproxy.com`
   - Port: `10000` (HTTP) o `10001` (HTTPS)
   - Username/Password: Desde dashboard

---

### Opción C: Bright Data (Antes Luminati)
- **Precio**: $10+/mes
- **Velocidad**: Excelente
- **Empresarial**: Más robusto
- **Link**: https://brightdata.com/

---

## 📋 Paso 2: Configurar Variables en Railway

Una vez que tengas las credenciales del proxy:

1. Ve a **Railway Dashboard** → Tu Proyecto → **Variables**

2. Agrega estas **5 variables de entorno**:

```bash
USE_PROXY=true
PROXY_HOST=p.webshare.io          # ← Tu proxy host
PROXY_PORT=80                     # ← Tu proxy port
PROXY_USERNAME=tu_username        # ← Tu username
PROXY_PASSWORD=tu_password        # ← Tu password
```

3. **Guarda los cambios**

4. Railway **redesplegará automáticamente** el bot

---

## 📋 Paso 3: Verificar Conexión

Una vez que Railway redespliega, revisa los logs:

### ✅ Conexión Exitosa:
```
Connected to binance via proxy p.webshare.io:80
Analyzing BTC/USDT...
BTC/USDT: HOLD (Score: 4.5/10.0) @ $110,000
```

### ❌ Si Ves Errores:

**Error 451 (aún bloqueado)**:
```
binance {"code":-1022,"msg":"Service unavailable from a restricted location"}
```
→ Verifica que las credenciales del proxy sean correctas

**Error de Proxy**:
```
ProxyError: Cannot connect to proxy
```
→ Revisa `PROXY_HOST` y `PROXY_PORT`

**Error de Autenticación**:
```
407 Proxy Authentication Required
```
→ Revisa `PROXY_USERNAME` y `PROXY_PASSWORD`

---

## 🎯 Configuración Completa en Railway

Tu configuración final debe verse así:

### Variables de Entorno en Railway:

```bash
# Telegram (YA LAS TIENES)
TELEGRAM_BOT_TOKEN=8499904153:AAHdRcxNX3NvEXOH6GVCwWXrgUP5kM6C81U
TELEGRAM_CHAT_ID=7147332663

# Exchange (NUEVO)
EXCHANGE_NAME=binance

# Proxy (NUEVO - obtener de Webshare/Smartproxy/etc)
USE_PROXY=true
PROXY_HOST=p.webshare.io
PROXY_PORT=80
PROXY_USERNAME=tu_username_aqui
PROXY_PASSWORD=tu_password_aqui

# Opcionales
CHECK_INTERVAL=120
ENABLE_FLASH_SIGNALS=true
```

---

## 🚀 Beneficios de Usar Binance + Proxy

### Antes (Kraken):
- ❌ Baja liquidez (~$5M/día en SOL)
- ❌ Solo 14 pares disponibles
- ❌ Spreads más amplios

### Ahora (Binance):
- ✅ **Alta liquidez** (~$2B/día en SOL - 400x más)
- ✅ **30 pares** de alta volatilidad
- ✅ Spreads más ajustados
- ✅ Más señales generadas
- ✅ Pares que Kraken no tiene: PEPE, WIF, BONK, NEAR, MATIC, etc.

---

## 💰 Costos Estimados

| Servicio | Costo Mensual | Total |
|----------|---------------|-------|
| Railway | $5 (con $5 gratis) | $0-5 |
| Webshare.io Proxy | $2.99 | $2.99 |
| **TOTAL** | | **$2.99-8** |

**Alternativa sin proxy**: Mantener Kraken (gratis pero menos liquidez)

---

## ❓ Preguntas Frecuentes

### ¿Puedo usar proxy gratis?
❌ No recomendado. Proxies gratis son:
- Lentos
- Inestables
- Frecuentemente bloqueados por Binance

### ¿Necesito API keys de Binance?
❌ No. El bot solo **lee precios públicos**, no tradea automáticamente.

### ¿El proxy afecta la velocidad?
Mínimamente. Los proxies residenciales agregan ~50-200ms de latencia (insignificante para análisis cada 2 minutos).

### ¿Puedo usar otro exchange?
Sí, pero Binance tiene la mejor liquidez. Alternativas:
- Kraken (ya lo usabas, sin proxy)
- Coinbase (menos liquidez)
- Kucoin (puede funcionar sin proxy)

Para cambiar exchange:
```bash
EXCHANGE_NAME=kraken  # Cambiar a kraken, coinbase, kucoin, etc.
USE_PROXY=false       # Desactivar proxy si no es Binance
```

---

## 🔄 Desactivar Proxy (Volver a Kraken)

Si no quieres usar proxy:

1. En Railway Variables, cambia:
```bash
EXCHANGE_NAME=kraken
USE_PROXY=false
```

2. El bot volverá a usar Kraken (sin proxy, menos liquidez)

---

## 🆘 Soporte

Si tienes problemas:
1. Verifica logs en Railway
2. Confirma credenciales del proxy
3. Prueba con otro servidor proxy de tu proveedor
4. Contacta al soporte del proveedor de proxy

---

## 📊 Próximos Pasos

Una vez configurado el proxy:
1. ✅ El bot se conectará a Binance
2. ✅ Analizará **30 pares** de alta liquidez
3. ✅ Generará más señales flash (mayor volatilidad)
4. ✅ Recibirás notificaciones en Telegram

**¡Disfruta de la liquidez de Binance!** 🚀
