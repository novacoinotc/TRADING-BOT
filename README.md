# Bot de Señales de Trading

Bot automatizado que analiza continuamente pares de criptomonedas y monedas fiat, generando señales de compra/venta basadas en análisis técnico y enviando notificaciones en tiempo real vía Telegram.

## Características

- **Monitoreo 24/7**: Análisis continuo de múltiples pares de trading
- **Análisis Técnico Completo**:
  - RSI (Índice de Fuerza Relativa)
  - MACD (Moving Average Convergence Divergence)
  - EMA (Exponential Moving Averages: 9, 21, 50)
  - Bandas de Bollinger
- **Notificaciones Inteligentes**: Alertas por Telegram solo cuando hay señales relevantes
- **Pares Soportados**: BTC/USDT, ETH/USDT, XRP/USDT, MXN/USD (según disponibilidad del exchange)
- **Sistema de Puntuación**: Evalúa la fuerza de cada señal

## Requisitos

- Python 3.8 o superior
- Cuenta de Telegram y Bot Token
- Conexión a internet estable

## Instalación

1. **Clonar el repositorio**:
```bash
git clone <repository-url>
cd TRADING-BOT
```

2. **Crear entorno virtual**:
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instalar dependencias**:
```bash
pip install -r requirements.txt
```

4. **Configurar variables de entorno**:
```bash
cp .env.example .env
```

Editar el archivo `.env` con tus credenciales:

```env
# Telegram Configuration
TELEGRAM_BOT_TOKEN=tu_token_del_bot
TELEGRAM_CHAT_ID=tu_chat_id

# Bot Configuration
CHECK_INTERVAL=300          # Intervalo de chequeo en segundos (5 minutos por defecto)
EXCHANGE_NAME=binance       # Exchange a utilizar (binance, kraken, etc.)

# Thresholds (Opcional)
RSI_OVERSOLD=30
RSI_OVERBOUGHT=70
```

### Cómo obtener el Token de Telegram

1. Abre Telegram y busca **@BotFather**
2. Envía el comando `/newbot`
3. Sigue las instrucciones para crear tu bot
4. Copia el token que te proporciona
5. Para obtener tu Chat ID:
   - Busca **@userinfobot** en Telegram
   - Inicia una conversación y te dará tu Chat ID

## Uso

### Iniciar el bot

```bash
python main.py
```

El bot comenzará a monitorear los pares configurados y enviará notificaciones cuando detecte oportunidades de trading.

### Ejemplo de Notificación

```
🟢 SEÑAL DE TRADING 🟢

Par: BTC/USDT
Acción: COMPRAR
Fuerza: ⭐⭐⭐⭐

💰 Precio: $45,230.50

📊 Indicadores:
• RSI: 28.45
• MACD: -125.3421
• MACD Señal: -98.2156
• EMA(9): $45,150.20
• EMA(21): $45,890.30
• EMA(50): $46,210.15

📈 Razones:
• RSI oversold (28.45)
• MACD bullish crossover
• Price below lower Bollinger Band

⏰ Análisis automático en timeframe 1h
```

## Estructura del Proyecto

```
TRADING-BOT/
│
├── config/
│   ├── __init__.py
│   └── config.py              # Configuración del bot
│
├── src/
│   ├── __init__.py
│   ├── market_monitor.py      # Monitor principal del mercado
│   ├── technical_analysis.py  # Análisis técnico e indicadores
│   └── telegram_bot.py        # Notificaciones de Telegram
│
├── logs/                      # Logs del bot (generado automáticamente)
│
├── main.py                    # Punto de entrada principal
├── requirements.txt           # Dependencias de Python
├── .env.example              # Plantilla de variables de entorno
├── .gitignore                # Archivos ignorados por git
└── README.md                 # Este archivo
```

## Configuración Avanzada

### Modificar Pares de Trading

Edita `config/config.py`:

```python
TRADING_PAIRS = [
    'BTC/USDT',
    'ETH/USDT',
    'XRP/USDT',
    'SOL/USDT',  # Agregar nuevos pares
]
```

### Ajustar Indicadores Técnicos

En `config/config.py` puedes modificar:

- Periodos de RSI, MACD, EMAs
- Umbrales de sobrecompra/sobreventa
- Timeframe de análisis (1m, 5m, 15m, 1h, 4h, 1d)
- Intervalo de chequeo

### Cambiar Exchange

El bot usa Binance por defecto, pero soporta múltiples exchanges via CCXT:

```env
EXCHANGE_NAME=kraken  # o coinbase, bitfinex, etc.
```

## Funcionamiento del Sistema de Señales

El bot utiliza un sistema de puntuación que evalúa múltiples indicadores:

### Señal de COMPRA (BUY)
- RSI < 30 (sobreventa): +2 puntos
- MACD crossover alcista: +1 punto
- EMAs en tendencia alcista: +1 punto
- Precio por debajo de banda inferior de Bollinger: +1 punto

### Señal de VENTA (SELL)
- RSI > 70 (sobrecompra): +2 puntos
- MACD crossover bajista: +1 punto
- EMAs en tendencia bajista: +1 punto
- Precio por encima de banda superior de Bollinger: +1 punto

**Nota**: Se requiere una puntuación neta de ±2 o más para generar una señal.

## Logs

Los logs se guardan automáticamente en `logs/trading_bot.log` e incluyen:
- Timestamp de cada análisis
- Señales detectadas
- Errores y advertencias
- Estado del bot

## Solución de Problemas

### El bot no envía notificaciones

1. Verifica que `TELEGRAM_BOT_TOKEN` y `TELEGRAM_CHAT_ID` estén correctamente configurados
2. Asegúrate de haber iniciado una conversación con tu bot en Telegram
3. Revisa los logs en `logs/trading_bot.log`

### Error al conectar con el exchange

1. Verifica tu conexión a internet
2. Algunos pares pueden no estar disponibles en todos los exchanges
3. Ajusta `EXCHANGE_NAME` en `.env` si es necesario

### Señales poco frecuentes

Esto es normal. El bot solo notifica cuando hay señales fuertes de compra/venta. Puedes:
- Reducir `CHECK_INTERVAL` para análisis más frecuentes
- Ajustar los umbrales en `config.py`
- Cambiar a un timeframe menor (ej: '15m' en vez de '1h')

## Advertencias Importantes

- **Este bot es solo para fines educativos y de investigación**
- Las señales NO son consejos de inversión
- Siempre realiza tu propio análisis antes de operar
- El trading de criptomonedas conlleva riesgos significativos
- Nunca inviertas más de lo que puedes permitirte perder

## Próximas Mejoras

- [ ] Interfaz web para monitoreo
- [ ] Backtesting de estrategias
- [ ] Múltiples estrategias de trading
- [ ] Integración con más exchanges
- [ ] Comandos interactivos de Telegram
- [ ] Dashboard de métricas
- [ ] Notificaciones por email

## Soporte

Para reportar bugs o solicitar features, por favor abre un issue en el repositorio.

## Licencia

MIT License - Uso libre con atribución

---

**Disclaimer**: Este software se proporciona "tal cual", sin garantías de ningún tipo. El uso de este bot es bajo tu propio riesgo.
