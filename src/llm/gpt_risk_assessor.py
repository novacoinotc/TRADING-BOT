"""
GPT Risk Assessor - Phase 3

Evaluates risk before executing trades using GPT reasoning.
Provides a sophisticated layer of risk analysis beyond traditional metrics.
"""

import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from src.llm.gpt_client import GPTClient

logger = logging.getLogger(__name__)


class GPTRiskAssessor:
    """
    GPT-powered risk assessment for trading decisions.
    Evaluates potential risks before trade execution.
    """

    SYSTEM_PROMPT = """Eres un experto en gestion de riesgo para trading de criptomonedas.
Tu trabajo es evaluar si un trade propuesto es seguro de ejecutar.

Considera:
1. RIESGO DE MERCADO: Volatilidad, tendencia, correlaciones
2. RIESGO DE PORTFOLIO: Concentracion, drawdown actual, posiciones abiertas
3. RIESGO DE TIMING: Hora del dia, eventos proximos, liquidez
4. RIESGO DE SENAL: Calidad de la senal, confianza del ML/RL

CRITICO: Tu evaluacion puede bloquear trades. Se conservador pero no paranoico.
Un trade bloqueado injustamente es mejor que una perdida evitable.

Responde siempre en español."""

    def __init__(self, gpt_client: GPTClient):
        """
        Initialize Risk Assessor

        Args:
            gpt_client: Initialized GPT client
        """
        self.gpt = gpt_client
        self.risk_history: List[Dict] = []
        self.blocked_count = 0
        self.approved_count = 0

    async def assess_trade_risk(
        self,
        pair: str,
        action: str,
        position_size_pct: float,
        signal: Dict,
        portfolio: Dict,
        market_state: Dict,
        open_positions: List[str]
    ) -> Dict[str, Any]:
        """
        Comprehensive risk assessment for a proposed trade

        Args:
            pair: Trading pair
            action: BUY or SELL
            position_size_pct: Proposed position size as % of portfolio
            signal: Signal data
            portfolio: Portfolio statistics
            market_state: Current market state
            open_positions: List of currently open positions

        Returns:
            Risk assessment with approve/block decision
        """
        prompt = self._build_risk_prompt(
            pair, action, position_size_pct, signal, portfolio, market_state, open_positions
        )

        try:
            response = await self.gpt.analyze(
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=prompt,
                temperature=0.3,  # Low temperature for consistent risk assessment
                max_tokens=4000,  # Must be enough for reasoning + response
                json_response=True
            )

            assessment = response["data"]

            # Track statistics
            if assessment.get("approved", True):
                self.approved_count += 1
            else:
                self.blocked_count += 1

            # Store in history
            self.risk_history.append({
                "timestamp": datetime.now().isoformat(),
                "pair": pair,
                "action": action,
                "assessment": assessment,
                "cost": response["cost"]
            })

            logger.info(
                f"GPT Risk Assessment for {pair} {action}: "
                f"{'APPROVED' if assessment.get('approved', True) else 'BLOCKED'} "
                f"(risk_level={assessment.get('risk_level', 'UNKNOWN')})"
            )

            return {
                "success": True,
                "assessment": assessment,
                "cost": response["cost"]
            }

        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            # Default to approve on error to not block trading
            return {
                "success": False,
                "error": str(e),
                "assessment": {
                    "approved": True,
                    "risk_level": "UNKNOWN",
                    "reason": "Assessment failed, defaulting to approve"
                }
            }

    def _build_risk_prompt(
        self,
        pair: str,
        action: str,
        position_size_pct: float,
        signal: Dict,
        portfolio: Dict,
        market_state: Dict,
        open_positions: List[str]
    ) -> str:
        """Build risk assessment prompt"""

        prompt = f"""
EVALUA EL RIESGO DE ESTE TRADE PROPUESTO:

== TRADE PROPUESTO ==
- Par: {pair}
- Accion: {action}
- Tamano Posicion: {position_size_pct:.1f}% del portfolio
- Score Senal: {signal.get('score', 0):.1f}/10
- Confianza: {signal.get('confidence', 0)}%
- Trade Type: {signal.get('trade_type', 'SPOT')}
- Leverage: {signal.get('leverage', 1)}x

== ESTADO DEL PORTFOLIO ==
- Balance: ${portfolio.get('current_balance', 50000):,.2f}
- P&L Actual: {portfolio.get('roi', 0):+.2f}%
- Drawdown Actual: {portfolio.get('max_drawdown', 0):.2f}%
- Win Rate: {portfolio.get('win_rate', 0):.1f}%
- Posiciones Abiertas: {len(open_positions)}
- Pares Abiertos: {', '.join(open_positions) if open_positions else 'Ninguno'}

== ESTADO DEL MERCADO ==
- RSI: {market_state.get('rsi', 50):.1f}
- Regimen: {market_state.get('regime', 'SIDEWAYS')} ({market_state.get('regime_strength', 'MEDIUM')})
- Fear & Greed: {market_state.get('fear_greed_index', 50)}
- Volatilidad: {market_state.get('volatility', 'medium')}
- Order Flow: {market_state.get('order_flow_bias', 'neutral')}
- Session: {market_state.get('current_session', 'UNKNOWN')}

== FACTORES DE SENAL ==
Razones: {', '.join(signal.get('reasons', [])[:3])}

== INSTRUCCIONES ==
Evalua el riesgo y responde en JSON con esta estructura EXACTA:

{{
    "approved": true,
    "risk_level": "LOW/MEDIUM/HIGH/CRITICAL",
    "risk_score": 35,
    "confidence": 85,

    "risk_factors": [
        {{
            "factor": "Nombre del factor de riesgo",
            "severity": "LOW/MEDIUM/HIGH",
            "description": "Descripcion breve"
        }}
    ],

    "mitigations": [
        "Acciones recomendadas para mitigar riesgo"
    ],

    "position_adjustment": {{
        "recommended_size_pct": {position_size_pct},
        "reason": "Razon del ajuste o null si no hay ajuste"
    }},

    "warnings": ["Advertencias importantes"],

    "reason": "Razon principal de la decision (1 oracion)"
}}

REGLAS DE DECISION:
- APPROVED=false si risk_score > 70
- APPROVED=false si hay factor de riesgo CRITICAL
- APPROVED=false si drawdown > 8% y position_size > 5%
- APPROVED=false si hay > 3 posiciones abiertas correlacionadas
- Reduce position_size si risk_score > 50
"""
        return prompt

    async def assess_portfolio_risk(
        self,
        portfolio: Dict,
        open_positions: List[Dict],
        market_context: Dict
    ) -> Dict[str, Any]:
        """
        Assess overall portfolio risk

        Args:
            portfolio: Portfolio statistics
            open_positions: List of open positions with details
            market_context: Current market context

        Returns:
            Portfolio risk assessment
        """
        prompt = f"""
EVALUA EL RIESGO ACTUAL DEL PORTFOLIO:

== PORTFOLIO ==
- Balance: ${portfolio.get('current_balance', 50000):,.2f}
- Equity: ${portfolio.get('equity', 50000):,.2f}
- P&L: {portfolio.get('roi', 0):+.2f}%
- Drawdown: {portfolio.get('max_drawdown', 0):.2f}%
- Posiciones Abiertas: {len(open_positions)}

== POSICIONES ABIERTAS ==
{json.dumps(open_positions, indent=2) if open_positions else "Ninguna"}

== CONTEXTO DE MERCADO ==
{json.dumps(market_context, indent=2)}

Responde en JSON:
{{
    "portfolio_risk_level": "LOW/MEDIUM/HIGH/CRITICAL",
    "risk_score": 0-100,
    "exposure_analysis": {{
        "total_exposure_pct": 0,
        "largest_position_pct": 0,
        "correlation_risk": "LOW/MEDIUM/HIGH"
    }},
    "recommendations": [
        "Lista de recomendaciones"
    ],
    "should_reduce_exposure": false,
    "emergency_close_pairs": []
}}
"""
        try:
            response = await self.gpt.analyze(
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=prompt,
                temperature=0.3,
                json_response=True
            )
            return {
                "success": True,
                "assessment": response["data"],
                "cost": response["cost"]
            }
        except Exception as e:
            logger.error(f"Portfolio risk assessment failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def check_correlation_risk(
        self,
        new_pair: str,
        open_positions: List[str],
        correlation_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Quick correlation risk check

        Args:
            new_pair: Pair being considered
            open_positions: Currently open positions
            correlation_data: Optional correlation matrix data

        Returns:
            Correlation risk assessment
        """
        if not open_positions:
            return {
                "risk": "LOW",
                "reason": "No hay posiciones abiertas",
                "approved": True
            }

        prompt = f"""
Evalua rapidamente el riesgo de correlacion:

Nueva posicion: {new_pair}
Posiciones abiertas: {', '.join(open_positions)}
{f"Datos de correlacion: {json.dumps(correlation_data)}" if correlation_data else ""}

Responde en JSON:
{{
    "correlation_risk": "LOW/MEDIUM/HIGH",
    "most_correlated_with": "par mas correlacionado o null",
    "approved": true,
    "reason": "Razon en 1 oracion"
}}
"""
        try:
            response = await self.gpt.analyze(
                system_prompt="Eres un analista de correlaciones. Responde brevemente en español.",
                user_prompt=prompt,
                temperature=0.2,
                max_tokens=2000,  # Must be enough for reasoning + response
                json_response=True
            )
            return response["data"]
        except Exception as e:
            logger.error(f"Correlation check failed: {e}")
            return {
                "correlation_risk": "UNKNOWN",
                "approved": True,
                "reason": "Check failed"
            }

    def get_risk_stats(self) -> Dict[str, Any]:
        """Get risk assessment statistics"""
        return {
            "total_assessments": self.approved_count + self.blocked_count,
            "approved": self.approved_count,
            "blocked": self.blocked_count,
            "block_rate": (
                self.blocked_count / max(1, self.approved_count + self.blocked_count) * 100
            ),
            "recent_history": self.risk_history[-10:]
        }
