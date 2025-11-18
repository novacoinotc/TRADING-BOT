#!/usr/bin/env python
"""
Test de Conexión a Binance Futures v2.0
Verifica que las credenciales y la integración funcionen correctamente
"""
import os
import sys
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Añadir directorio raíz al path
sys.path.insert(0, os.path.dirname(__file__))

from src.binance_integration.binance_client import BinanceClient
from src.binance_integration.futures_trader import FuturesTrader


def test_connection():
    """Test básico de conexión a Binance"""
    print("=" * 70)
    print("🧪 TEST DE CONEXIÓN A BINANCE FUTURES v2.0")
    print("=" * 70)
    print()

    # 1. Verificar variables de entorno
    print("📋 Paso 1: Verificando variables de entorno...")
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    base_url = os.getenv('BINANCE_BASE_URL')
    testnet = os.getenv('BINANCE_TESTNET', 'true').lower() == 'true'

    if not api_key or not api_secret:
        print("❌ ERROR: BINANCE_API_KEY o BINANCE_API_SECRET no configuradas")
        print("   Verifica tu archivo .env")
        return False

    print(f"✅ Variables configuradas")
    print(f"   Mode: {'🧪 TESTNET' if testnet else '🔴 LIVE'}")
    print(f"   URL: {base_url}")
    print()

    # 2. Crear cliente de Binance
    print("📋 Paso 2: Creando cliente de Binance...")
    try:
        client = BinanceClient(
            api_key=api_key,
            api_secret=api_secret,
            base_url=base_url
        )
        print("✅ Cliente creado exitosamente")
    except Exception as e:
        print(f"❌ ERROR creando cliente: {e}")
        return False
    print()

    # 3. Test: Server time
    print("📋 Paso 3: Probando conexión (server time)...")
    try:
        server_time = client.get_server_time()
        print(f"✅ Conectado - Server time: {server_time}")
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False
    print()

    # 4. Test: Balance
    print("📋 Paso 4: Obteniendo balance de la cuenta...")
    try:
        balance_info = client.get_balance()
        usdt = next((b for b in balance_info if b['asset'] == 'USDT'), None)
        if usdt:
            available = float(usdt['availableBalance'])
            print(f"✅ Balance USDT: ${available:,.2f}")
            if available < 10:
                print(f"   ⚠️  ADVERTENCIA: Balance bajo (mínimo recomendado: $100)")
        else:
            print("⚠️  No se encontró balance USDT")
    except Exception as e:
        print(f"❌ ERROR obteniendo balance: {e}")
        return False
    print()

    # 5. Test: Precio de BTC
    print("📋 Paso 5: Obteniendo precio de mercado...")
    try:
        btc_price = client.get_price('BTCUSDT')
        print(f"✅ Precio BTC: ${btc_price:,.2f}")
    except Exception as e:
        print(f"❌ ERROR obteniendo precio: {e}")
        return False
    print()

    # 6. Test: Exchange info
    print("📋 Paso 6: Verificando información del exchange...")
    try:
        exchange_info = client.get_exchange_info()
        num_symbols = len(exchange_info.get('symbols', []))
        print(f"✅ Exchange info obtenida: {num_symbols} pares disponibles")
    except Exception as e:
        print(f"❌ ERROR obteniendo exchange info: {e}")
        return False
    print()

    # 7. Test: FuturesTrader
    print("📋 Paso 7: Verificando FuturesTrader...")
    try:
        futures_trader = FuturesTrader(
            client=client,
            default_leverage=3,
            use_isolated_margin=True
        )
        print("✅ FuturesTrader inicializado correctamente")
    except Exception as e:
        print(f"❌ ERROR creando FuturesTrader: {e}")
        return False
    print()

    # 8. Test: Posiciones abiertas
    print("📋 Paso 8: Verificando posiciones abiertas...")
    try:
        positions = client.get_position_risk()
        open_positions = [p for p in positions if float(p['positionAmt']) != 0]
        print(f"✅ Posiciones abiertas: {len(open_positions)}")
        if open_positions:
            for pos in open_positions:
                symbol = pos['symbol']
                amt = float(pos['positionAmt'])
                entry = float(pos['entryPrice'])
                pnl = float(pos['unRealizedProfit'])
                print(f"   - {symbol}: {amt:+.4f} @ ${entry:,.2f} (PnL: ${pnl:+.2f})")
    except Exception as e:
        print(f"❌ ERROR obteniendo posiciones: {e}")
        return False
    print()

    print("=" * 70)
    print("✅ TODOS LOS TESTS PASARON - INTEGRACIÓN FUNCIONAL")
    print("=" * 70)
    print()
    print("🚀 Siguiente paso: python main.py")
    print()

    return True


if __name__ == "__main__":
    try:
        success = test_connection()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏸️  Test interrumpido por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR INESPERADO: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
