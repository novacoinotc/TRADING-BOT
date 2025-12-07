"""
GPT Meta-Reasoner - Phase 1

Analyzes trading performance and suggests strategic improvements using GPT.
This module provides deep reasoning about:
- Why trades are winning/losing
- Pattern recognition in performance
- Strategic parameter adjustments
- Market condition analysis
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from src.llm.gpt_client import GPTClient

logger = logging.getLogger(__name__)


class GPTMetaReasoner:
    """
    GPT-powered meta-analysis of trading performance.
    Provides strategic insights and improvement suggestions.
    """

    SYSTEM_PROMPT = """You are an elite quantitative trading analyst with 20+ years of experience.
You analyze cryptocurrency trading bot performance data and provide actionable insights.

Your analysis should be:
1. DATA-DRIVEN: Base conclusions on the numbers provided
2. SPECIFIC: Give exact parameter recommendations (e.g., "change threshold from 5.5 to 4.8")
3. ACTIONABLE: Provide clear next steps
4. HONEST: Acknowledge when data is insufficient for conclusions

You understand:
- Technical analysis (RSI, MACD, EMA, Bollinger Bands)
- Market regimes (Bull, Bear, Sideways)
- Risk management (position sizing, stop loss, drawdown)
- Machine Learning trading signals
- Reinforcement Learning Q-learning for trading

CRITICAL: Your recommendations will be automatically applied to the trading bot.
Be conservative with suggestions - wrong parameters can cause significant losses.

Always respond in Spanish."""

    def __init__(self, gpt_client: GPTClient):
        """
        Initialize Meta-Reasoner

        Args:
            gpt_client: Initialized GPT client
        """
        self.gpt = gpt_client
        self.analysis_history: List[Dict] = []
        self.last_analysis_time: Optional[datetime] = None

    async def analyze_performance(
        self,
        trades: List[Dict],
        current_params: Dict,
        portfolio_stats: Dict,
        market_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Perform deep analysis of trading performance

        Args:
            trades: List of recent trades with full data
            current_params: Current bot parameters
            portfolio_stats: Current portfolio statistics
            market_context: Optional market context data

        Returns:
            Analysis with insights and recommendations
        """
        if not trades:
            return {
                "success": False,
                "error": "No trades to analyze",
                "recommendations": []
            }

        # Build analysis prompt
        prompt = self._build_analysis_prompt(
            trades, current_params, portfolio_stats, market_context
        )

        try:
            response = await self.gpt.analyze(
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=prompt,
                temperature=0.7,
                max_tokens=2500,
                json_response=True
            )

            analysis = response["data"]

            # Store in history
            self.analysis_history.append({
                "timestamp": datetime.now().isoformat(),
                "analysis": analysis,
                "trades_count": len(trades),
                "cost": response["cost"]
            })

            self.last_analysis_time = datetime.now()

            # Log summary
            logger.info(
                f"GPT Meta-Reasoner analysis complete: "
                f"{len(analysis.get('recommendations', []))} recommendations, "
                f"${response['cost']:.4f}"
            )

            return {
                "success": True,
                "analysis": analysis,
                "cost": response["cost"],
                "cached": response.get("cached", False)
            }

        except Exception as e:
            logger.error(f"Meta-Reasoner analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "recommendations": []
            }

    def _build_analysis_prompt(
        self,
        trades: List[Dict],
        current_params: Dict,
        portfolio_stats: Dict,
        market_context: Optional[Dict]
    ) -> str:
        """Build detailed analysis prompt"""

        # Calculate trade statistics
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('pnl', 0) < 0]

        win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
        avg_win = sum(t.get('pnl_pct', 0) for t in winning_trades) / max(1, len(winning_trades))
        avg_loss = sum(t.get('pnl_pct', 0) for t in losing_trades) / max(1, len(losing_trades))

        # Recent trades summary (last 20)
        recent_trades = trades[-20:]
        trades_summary = []
        for t in recent_trades:
            trades_summary.append({
                "pair": t.get('pair'),
                "side": t.get('side'),
                "pnl_pct": round(t.get('pnl_pct', 0), 2),
                "duration_min": round(t.get('duration', 0) / 60, 1),
                "reason": t.get('reason', 'UNKNOWN'),
                "trade_type": t.get('trade_type', 'SPOT'),
                "leverage": t.get('leverage', 1)
            })

        # Build the prompt
        prompt = f"""
ANALIZA EL RENDIMIENTO DE ESTE BOT DE TRADING Y PROPORCIONA RECOMENDACIONES CONCRETAS.

== ESTADISTICAS DEL PORTFOLIO ==
- Balance Actual: ${portfolio_stats.get('current_balance', 50000):,.2f} USDT
- P&L Total: ${portfolio_stats.get('net_pnl', 0):,.2f} ({portfolio_stats.get('roi', 0):+.2f}%)
- Win Rate: {portfolio_stats.get('win_rate', 0):.1f}%
- Total Trades: {portfolio_stats.get('total_trades', 0)}
- Max Drawdown: {portfolio_stats.get('max_drawdown', 0):.2f}%
- Profit Factor: {portfolio_stats.get('profit_factor', 0):.2f}
- Sharpe Ratio: {portfolio_stats.get('sharpe_ratio', 0):.2f}

== ANALISIS DE TRADES ==
- Trades Analizados: {len(trades)}
- Win Rate (muestra): {win_rate:.1f}%
- Ganancia Promedio: {avg_win:+.2f}%
- Perdida Promedio: {avg_loss:+.2f}%
- Trades Ganadores: {len(winning_trades)}
- Trades Perdedores: {len(losing_trades)}

== ULTIMOS 20 TRADES ==
{json.dumps(trades_summary, indent=2)}

== PARAMETROS ACTUALES DEL BOT ==
{json.dumps(current_params, indent=2)}

== CONTEXTO DE MERCADO ==
{json.dumps(market_context, indent=2) if market_context else "No disponible"}

== INSTRUCCIONES ==
Analiza los datos y responde en JSON con esta estructura EXACTA:

{{
    "summary": "Resumen ejecutivo de 2-3 oraciones sobre el estado del bot",

    "performance_analysis": {{
        "strengths": ["Lista de fortalezas identificadas"],
        "weaknesses": ["Lista de debilidades identificadas"],
        "patterns": ["Patrones observados en trades ganadores/perdedores"]
    }},

    "losing_trades_analysis": {{
        "main_causes": ["Causas principales de perdidas"],
        "common_conditions": ["Condiciones de mercado donde pierde"],
        "preventable_percentage": 50
    }},

    "recommendations": [
        {{
            "parameter": "NOMBRE_PARAMETRO",
            "current_value": "valor actual",
            "recommended_value": "valor recomendado",
            "reason": "Razon del cambio",
            "expected_impact": "Impacto esperado (+X% win rate)",
            "confidence": 85,
            "priority": "HIGH"
        }}
    ],

    "strategy_insights": {{
        "market_conditions": "Condiciones de mercado actuales",
        "recommended_approach": "Enfoque recomendado",
        "risk_level": "LOW/MEDIUM/HIGH"
    }},

    "next_steps": ["Lista de acciones prioritarias ordenadas"]
}}

IMPORTANTE: Solo recomienda cambios de parametros si tienes confianza >= 70%.
Los parametros disponibles son: CONSERVATIVE_THRESHOLD, FLASH_THRESHOLD, FLASH_MIN_CONFIDENCE,
BASE_POSITION_SIZE_PCT, MAX_POSITION_SIZE_PCT, RSI_OVERSOLD, RSI_OVERBOUGHT,
MIN_CONFIDENCE_FOR_FUTURES, TRAILING_DISTANCE_PCT, BREAKEVEN_AFTER_PCT.
"""
        return prompt

    async def get_quick_insight(
        self,
        trade: Dict,
        market_state: Dict
    ) -> str:
        """
        Get quick insight about a specific trade

        Args:
            trade: Trade data
            market_state: Current market state

        Returns:
            Brief insight string
        """
        prompt = f"""
Trade reciente: {json.dumps(trade, indent=2)}
Estado del mercado: {json.dumps(market_state, indent=2)}

En 1-2 oraciones, explica por que este trade {trade.get('pnl', 0) > 0 and 'gano' or 'perdio'}
y que se puede aprender de el.
"""
        try:
            response = await self.gpt.analyze(
                system_prompt="Eres un analista de trading experto. Responde brevemente en español.",
                user_prompt=prompt,
                temperature=0.5,
                max_tokens=150
            )
            return response["content"]
        except Exception as e:
            logger.error(f"Quick insight failed: {e}")
            return "Analisis no disponible"

    async def analyze_losing_streak(
        self,
        losing_trades: List[Dict],
        current_params: Dict
    ) -> Dict[str, Any]:
        """
        Special analysis for losing streaks

        Args:
            losing_trades: Recent losing trades
            current_params: Current parameters

        Returns:
            Emergency recommendations
        """
        prompt = f"""
ALERTA: El bot ha tenido {len(losing_trades)} trades perdedores consecutivos.

Trades perdedores:
{json.dumps(losing_trades, indent=2)}

Parametros actuales:
{json.dumps(current_params, indent=2)}

Analiza la situacion y responde en JSON:

{{
    "severity": "LOW/MEDIUM/HIGH/CRITICAL",
    "diagnosis": "Diagnostico de que esta pasando",
    "immediate_actions": [
        {{
            "action": "Descripcion de la accion",
            "parameter": "NOMBRE_PARAMETRO o null",
            "new_value": "valor o null",
            "urgency": "IMMEDIATE/SOON"
        }}
    ],
    "should_pause_trading": false,
    "pause_duration_minutes": 0,
    "recovery_strategy": "Estrategia para recuperarse"
}}
"""
        try:
            response = await self.gpt.analyze(
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=prompt,
                temperature=0.3,  # More deterministic for emergencies
                json_response=True
            )
            return {
                "success": True,
                "analysis": response["data"],
                "cost": response["cost"]
            }
        except Exception as e:
            logger.error(f"Losing streak analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "should_pause_trading": True  # Default to safe
            }

    def get_analysis_history(self, limit: int = 10) -> List[Dict]:
        """Get recent analysis history"""
        return self.analysis_history[-limit:]

    def should_analyze(self, min_interval_minutes: int = 60) -> bool:
        """Check if enough time has passed since last analysis"""
        if not self.last_analysis_time:
            return True
        elapsed = datetime.now() - self.last_analysis_time
        return elapsed >= timedelta(minutes=min_interval_minutes)
