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
                    'symbol': pos['symbol'],
                    'side': 'LONG' if float(pos['positionAmt']) > 0 else 'SHORT',
                    'entry_price': float(pos['entryPrice']),
                    'quantity': abs(float(pos['positionAmt'])),
                    'unrealized_pnl': float(pos['unRealizedProfit']),
                    'leverage': int(pos['leverage']),
                    'notional': abs(float(pos['positionAmt']) * float(pos['entryPrice']))
                }
                for pos in positions.values()
                if float(pos.get('positionAmt', 0)) != 0
            ]
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
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
