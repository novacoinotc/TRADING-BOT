"""
API Simple para Dashboard - Bot Trading v2.0
Expone endpoints REST para visualizar estado del bot en tiempo real
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

app = FastAPI(title="Trading Bot API", version="2.0")

# CORS - permitir requests desde cualquier origen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales que se llenarán desde main.py
_market_monitor = None
_autonomy_controller = None

def set_bot_instances(market_monitor, autonomy_controller):
    """Asignar instancias del bot para acceso desde API"""
    global _market_monitor, _autonomy_controller
    _market_monitor = market_monitor
    _autonomy_controller = autonomy_controller
    logger.info("✅ API: Bot instances assigned")

@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "online",
        "service": "Trading Bot API v2.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/status")
async def get_status() -> Dict[str, Any]:
    """Estado general del bot"""
    try:
        if not _market_monitor:
            return {"error": "Bot not initialized"}

        # Balance de Binance
        try:
            balance_info = _market_monitor.binance_client.get_balance()
            usdt = next((b for b in balance_info if b['asset'] == 'USDT'), None)
            balance = float(usdt.get('balance', 0)) if usdt else 0
            available = float(usdt.get('availableBalance', 0)) if usdt else 0
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            balance = 0
            available = 0

        # Posiciones abiertas
        try:
            positions = _market_monitor.position_monitor.get_open_positions()
            open_positions = [
                {
                    'symbol': pos.get('symbol', 'UNKNOWN'),
                    'side': 'LONG' if float(pos.get('positionAmt', 0)) > 0 else 'SHORT',
                    'entry_price': float(pos.get('entryPrice', 0)),
                    'quantity': abs(float(pos.get('positionAmt', 0))),
                    'unrealized_pnl': float(pos.get('unRealizedProfit', 0)),
                    'leverage': int(pos.get('leverage', 1)),
                    'notional': abs(float(pos.get('positionAmt', 0)) * float(pos.get('entryPrice', 0)))
                }
                for pos in positions.values()
                if float(pos.get('positionAmt', 0)) != 0
            ]
        except Exception as e:
            logger.error(f"Error getting positions: {e}", exc_info=True)
            open_positions = []

        # Stats del RL Agent
        rl_stats = {}
        if _autonomy_controller:
            try:
                rl_stats = {
                    'total_trades': _autonomy_controller.total_trades_all_time,
                    'exploration_rate': round(_autonomy_controller.rl_agent.exploration_rate, 4),
                    'q_table_size': len(_autonomy_controller.rl_agent.q_table),
                    'learning_enabled': True
                }
            except Exception as e:
                logger.error(f"Error getting RL stats: {e}")

        # P&L total de posiciones abiertas
        total_unrealized_pnl = sum(pos['unrealized_pnl'] for pos in open_positions)

        return {
            'status': 'online',
            'timestamp': datetime.now().isoformat(),
            'balance': {
                'total': round(balance, 2),
                'available': round(available, 2),
                'in_positions': round(balance - available, 2)
            },
            'positions': {
                'count': len(open_positions),
                'total_unrealized_pnl': round(total_unrealized_pnl, 2),
                'list': open_positions
            },
            'rl_agent': rl_stats,
            'system': {
                'mode': 'Binance Testnet',
                'leverage': 3
            }
        }

    except Exception as e:
        logger.error(f"Error in /api/status: {e}", exc_info=True)
        return {'error': str(e), 'timestamp': datetime.now().isoformat()}


@app.get("/portfolio")
async def get_portfolio() -> Dict[str, Any]:
    """
    Retorna portfolio con historial de trades cerrados
    """
    try:
        if not _market_monitor:
            return {"error": "Bot not initialized"}

        closed_trades = []

        # Intentar obtener trades cerrados del position_monitor
        try:
            if hasattr(_market_monitor, 'position_monitor') and _market_monitor.position_monitor:
                # Si el position_monitor tiene un atributo de closed_trades
                if hasattr(_market_monitor.position_monitor, 'closed_trades'):
                    closed_trades_raw = _market_monitor.position_monitor.closed_trades

                    # Convertir a formato del dashboard
                    for trade in closed_trades_raw:
                        closed_trades.append({
                            'id': trade.get('id', len(closed_trades)),
                            'symbol': trade.get('symbol', 'UNKNOWN'),
                            'side': trade.get('side', 'LONG'),
                            'leverage': trade.get('leverage', 1),
                            'pnl': trade.get('realized_pnl', 0),
                            'pnl_pct': trade.get('realized_pnl_pct', 0),
                            'timestamp': trade.get('close_time', datetime.now().isoformat())
                        })
        except Exception as e:
            logger.error(f"Error getting closed trades: {e}")

        # Si no hay historial, generar datos de ejemplo (opcional - comentar si no quieres)
        # if len(closed_trades) == 0:
        #     # Datos de ejemplo para testing del dashboard
        #     import random
        #     symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        #     for i in range(5):
        #         pnl = random.uniform(-20, 30)
        #         closed_trades.append({
        #             'id': i,
        #             'symbol': random.choice(symbols),
        #             'side': random.choice(['LONG', 'SHORT']),
        #             'leverage': random.choice([2, 3]),
        #             'pnl': pnl,
        #             'pnl_pct': pnl / 50 * 100,
        #             'timestamp': datetime.now().isoformat()
        #         })

        return {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'closed_trades': closed_trades,
            'total_trades': len(closed_trades)
        }

    except Exception as e:
        logger.error(f"Error in /portfolio: {e}", exc_info=True)
        return {
            'error': str(e),
            'timestamp': datetime.now().isoformat(),
            'closed_trades': []
        }
