"""
GPT Decision Explainer - Phase 2

Explains trading decisions in natural language.
Provides transparency about why the bot takes specific actions.
"""

import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from src.llm.gpt_client import GPTClient

logger = logging.getLogger(__name__)


class GPTDecisionExplainer:
    """
    Explains trading decisions in human-readable language.
    Helps users understand why the bot takes specific actions.
    """

    SYSTEM_PROMPT = """Eres un experto en trading de criptomonedas que explica decisiones de trading de forma clara y concisa.

Tu trabajo es explicar POR QUE el bot tomo una decision especifica.
- Usa lenguaje simple pero tecnico
- Se breve (2-4 oraciones)
- Menciona los factores MAS importantes
- Si hay riesgo, mencionalo claramente

Siempre responde en español."""

    def __init__(self, gpt_client: GPTClient):
        """
        Initialize Decision Explainer

        Args:
            gpt_client: Initialized GPT client
        """
        self.gpt = gpt_client
        self.explanations_cache: Dict[str, str] = {}

    async def explain_trade_decision(
        self,
        pair: str,
        action: str,
        signal: Dict,
        ml_prediction: Optional[Dict] = None,
        rl_decision: Optional[Dict] = None,
        market_state: Optional[Dict] = None,
        arsenal_data: Optional[Dict] = None
    ) -> str:
        """
        Explain why a trade was opened

        Args:
            pair: Trading pair
            action: BUY or SELL
            signal: Signal data with score, reasons, etc.
            ml_prediction: ML prediction data
            rl_decision: RL agent decision
            market_state: Current market state
            arsenal_data: Advanced arsenal analysis

        Returns:
            Human-readable explanation
        """
        prompt = self._build_trade_prompt(
            pair, action, signal, ml_prediction, rl_decision, market_state, arsenal_data
        )

        try:
            response = await self.gpt.analyze(
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=prompt,
                temperature=0.5,
                max_tokens=250
            )
            return response["content"]

        except Exception as e:
            logger.error(f"Trade explanation failed: {e}")
            return self._fallback_explanation(action, signal)

    def _build_trade_prompt(
        self,
        pair: str,
        action: str,
        signal: Dict,
        ml_prediction: Optional[Dict],
        rl_decision: Optional[Dict],
        market_state: Optional[Dict],
        arsenal_data: Optional[Dict]
    ) -> str:
        """Build trade explanation prompt"""

        prompt = f"""
El bot acaba de abrir un trade:

TRADE:
- Par: {pair}
- Accion: {action}
- Score de senal: {signal.get('score', 0):.1f}/10
- Confianza: {signal.get('confidence', 0)}%

RAZONES DE LA SENAL:
{chr(10).join('- ' + r for r in signal.get('reasons', [])[:5])}

"""
        if ml_prediction:
            prompt += f"""
ML PREDICTION:
- Prediccion: {ml_prediction.get('prediction', 'N/A')}
- Probabilidad de Win: {ml_prediction.get('win_probability', 0)*100:.1f}%
- Confianza ML: {ml_prediction.get('confidence', 0)}%
"""

        if rl_decision:
            prompt += f"""
RL AGENT:
- Accion elegida: {rl_decision.get('chosen_action', 'N/A')}
- Multiplicador de posicion: {rl_decision.get('position_size_multiplier', 1.0):.2f}x
- Trade type: {rl_decision.get('trade_type', 'SPOT')}
"""

        if market_state:
            prompt += f"""
ESTADO DEL MERCADO:
- RSI: {market_state.get('rsi', 50):.1f}
- Regimen: {market_state.get('regime', 'SIDEWAYS')}
- Fear & Greed: {market_state.get('fear_greed_index', 50)}
- Order Flow: {market_state.get('order_flow_bias', 'neutral')}
"""

        if arsenal_data:
            prompt += f"""
ARSENAL AVANZADO:
- Session: {arsenal_data.get('current_session', 'N/A')}
- Funding Rate: {arsenal_data.get('funding_sentiment', 'neutral')}
- Pattern: {arsenal_data.get('pattern_type', 'NONE')}
- Near POC: {arsenal_data.get('near_poc', False)}
"""

        prompt += """
Explica en 2-4 oraciones por que el bot tomo esta decision.
Menciona los factores mas importantes que influyeron.
"""
        return prompt

    def _fallback_explanation(self, action: str, signal: Dict) -> str:
        """Generate fallback explanation without GPT"""
        reasons = signal.get('reasons', [])[:2]
        score = signal.get('score', 0)
        confidence = signal.get('confidence', 0)

        if action == 'BUY':
            base = f"Senal de compra detectada (score {score:.1f}/10, {confidence}% confianza)"
        else:
            base = f"Senal de venta detectada (score {score:.1f}/10, {confidence}% confianza)"

        if reasons:
            base += f". Razones principales: {', '.join(reasons)}."

        return base

    async def explain_trade_closed(
        self,
        trade: Dict,
        market_conditions: Optional[Dict] = None
    ) -> str:
        """
        Explain why a trade was closed

        Args:
            trade: Closed trade data
            market_conditions: Market conditions at close

        Returns:
            Explanation of the close
        """
        prompt = f"""
El bot cerro un trade:

RESULTADO:
- Par: {trade.get('pair', 'N/A')}
- Direccion: {trade.get('side', 'N/A')}
- P&L: {trade.get('pnl_pct', 0):+.2f}%
- Razon de cierre: {trade.get('reason', 'UNKNOWN')}
- Duracion: {trade.get('duration', 0)/60:.1f} minutos
- Tipo: {trade.get('trade_type', 'SPOT')}
- Leverage: {trade.get('leverage', 1)}x

{f"Condiciones de mercado: {json.dumps(market_conditions, indent=2)}" if market_conditions else ""}

Explica brevemente (2-3 oraciones) por que el trade termino asi y que se puede aprender.
"""
        try:
            response = await self.gpt.analyze(
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=prompt,
                temperature=0.5,
                max_tokens=200
            )
            return response["content"]

        except Exception as e:
            logger.error(f"Close explanation failed: {e}")
            reason = trade.get('reason', 'UNKNOWN')
            pnl = trade.get('pnl_pct', 0)
            if pnl > 0:
                return f"Trade cerrado con ganancia de {pnl:+.2f}% por {reason}."
            else:
                return f"Trade cerrado con perdida de {pnl:+.2f}% por {reason}."

    async def explain_trade_blocked(
        self,
        pair: str,
        signal: Dict,
        block_reason: str,
        blocker: str
    ) -> str:
        """
        Explain why a trade was blocked

        Args:
            pair: Trading pair
            signal: Original signal
            block_reason: Reason for blocking
            blocker: What blocked it (RL Agent, Sentiment, Arsenal, etc.)

        Returns:
            Explanation of why trade was blocked
        """
        prompt = f"""
El bot detecto una senal pero decidio NO abrir el trade:

SENAL ORIGINAL:
- Par: {pair}
- Accion: {signal.get('action', 'N/A')}
- Score: {signal.get('score', 0):.1f}/10

BLOQUEADO POR: {blocker}
RAZON: {block_reason}

Explica brevemente (1-2 oraciones) por que fue buena decision bloquear este trade.
"""
        try:
            response = await self.gpt.analyze(
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=prompt,
                temperature=0.5,
                max_tokens=150
            )
            return response["content"]

        except Exception as e:
            logger.error(f"Block explanation failed: {e}")
            return f"Trade bloqueado por {blocker}: {block_reason}"

    async def generate_daily_summary(
        self,
        trades: List[Dict],
        portfolio_stats: Dict,
        notable_events: List[str]
    ) -> str:
        """
        Generate daily trading summary

        Args:
            trades: Today's trades
            portfolio_stats: Current portfolio stats
            notable_events: Notable events that occurred

        Returns:
            Daily summary in natural language
        """
        if not trades:
            return "Hoy no se ejecutaron trades. El bot continua monitoreando el mercado."

        winning = sum(1 for t in trades if t.get('pnl', 0) > 0)
        losing = len(trades) - winning
        total_pnl = sum(t.get('pnl', 0) for t in trades)
        win_rate = winning / len(trades) * 100

        prompt = f"""
Genera un resumen del dia de trading:

TRADES DEL DIA:
- Total: {len(trades)}
- Ganadores: {winning}
- Perdedores: {losing}
- Win Rate: {win_rate:.1f}%
- P&L Total: ${total_pnl:,.2f}

PORTFOLIO ACTUAL:
- Balance: ${portfolio_stats.get('current_balance', 0):,.2f}
- ROI Total: {portfolio_stats.get('roi', 0):+.2f}%
- Max Drawdown: {portfolio_stats.get('max_drawdown', 0):.2f}%

EVENTOS NOTABLES:
{chr(10).join('- ' + e for e in notable_events) if notable_events else '- Ningun evento notable'}

TOP TRADES:
{json.dumps(sorted(trades, key=lambda x: x.get('pnl', 0), reverse=True)[:3], indent=2)}

Escribe un resumen de 4-6 oraciones sobre el dia, incluyendo:
1. Como fue el rendimiento general
2. Que salio bien
3. Que se puede mejorar
4. Perspectiva para manana
"""
        try:
            response = await self.gpt.analyze(
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=prompt,
                temperature=0.5,  # Moderate temperature for explanations
                max_tokens=400
            )
            return response["content"]

        except Exception as e:
            logger.error(f"Daily summary failed: {e}")
            return (
                f"Resumen del dia: {len(trades)} trades ejecutados, "
                f"{winning} ganadores ({win_rate:.1f}% win rate), "
                f"P&L: ${total_pnl:+,.2f}"
            )
