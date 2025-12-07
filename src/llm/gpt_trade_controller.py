"""
GPT Trade Controller - Absolute Control Over Trading

This is the MASTER CONTROLLER that gives GPT complete control over:
1. Signal evaluation and generation
2. Trade entry decisions
3. Position sizing and leverage
4. Risk management (SL/TP)
5. Trade exit decisions
6. Learning from outcomes

GPT has VETO power over all traditional systems (ML, RL).
Traditional systems provide INPUT, GPT makes DECISIONS.
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from src.llm.gpt_client import GPTClient
from src.llm.gpt_experience_learner import GPTExperienceLearner

logger = logging.getLogger(__name__)


class GPTTradeController:
    """
    Master controller for GPT-driven trading.

    Hierarchy:
    1. GPT has FINAL say on all decisions
    2. ML/RL systems provide recommendations (can be overridden)
    3. Technical indicators provide data (not decisions)

    Flow:
    Signal → GPT Analysis → GPT Decision → Execution → GPT Learning
    """

    CONTROLLER_SYSTEM_PROMPT = """Eres el CEREBRO CENTRAL de un bot de trading de criptomonedas.
Tienes CONTROL ABSOLUTO sobre todas las decisiones de trading.

Tu trabajo es:
1. Evaluar señales de trading holísticamente
2. Decidir si abrir o no abrir trades
3. Determinar tamaño de posición y leverage
4. Establecer stop-loss y take-profit óptimos
5. Decidir cuándo cerrar trades
6. Aprender de cada resultado

FILOSOFÍA DE TRADING:
- Preservar capital es PRIORIDAD #1
- Solo trades con alta probabilidad (>65%)
- Risk/Reward mínimo 1:1.5
- Mejor perder una oportunidad que perder dinero
- Un "NO TRADE" es una decisión válida

TIENES ACCESO A:
- Indicadores técnicos (RSI, MACD, EMA, etc.)
- Sentiment del mercado (Fear & Greed, noticias)
- Order book (presión de compra/venta)
- Predicciones ML (como referencia, puedes ignorarlas)
- Recomendaciones RL (como referencia, puedes ignorarlas)
- Sabiduría aprendida de trades pasados

Responde SIEMPRE en español y en JSON estructurado."""

    def __init__(
        self,
        gpt_client: GPTClient,
        experience_learner: GPTExperienceLearner,
        param_update_callback: Optional[Callable[[str, Any], bool]] = None,
        notification_callback: Optional[Callable[[str], Any]] = None,
        config: Optional[Any] = None
    ):
        """
        Initialize GPT Trade Controller

        Args:
            gpt_client: Initialized GPT client
            experience_learner: GPT Experience Learner instance
            param_update_callback: Callback to update parameters
            notification_callback: Callback to send notifications
            config: Configuration object
        """
        self.gpt = gpt_client
        self.learner = experience_learner
        self.param_callback = param_update_callback
        self.notify_callback = notification_callback
        self.config = config

        # State
        self.is_enabled = True
        self.total_decisions = 0
        self.trades_approved = 0
        self.trades_rejected = 0
        self.total_cost = 0.0

        # Trade tracking
        self.active_trades: Dict[str, Dict] = {}  # pair -> trade info
        self.pending_reviews: List[Dict] = []

        # Performance
        self.session_pnl = 0.0
        self.consecutive_wins = 0
        self.consecutive_losses = 0

        logger.info("GPT Trade Controller initialized - ABSOLUTE CONTROL MODE")

    async def evaluate_signal(
        self,
        pair: str,
        signal: Dict,
        indicators: Dict,
        sentiment_data: Optional[Dict] = None,
        orderbook_data: Optional[Dict] = None,
        regime_data: Optional[Dict] = None,
        ml_prediction: Optional[Dict] = None,
        rl_recommendation: Optional[Dict] = None,
        portfolio: Optional[Dict] = None,
        open_positions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        MASTER DECISION POINT - Evaluate a trading signal

        GPT has FINAL say. ML/RL are just advisors.

        Args:
            pair: Trading pair
            signal: Generated signal from traditional system
            indicators: Technical indicators
            sentiment_data: Sentiment analysis
            orderbook_data: Order book analysis
            regime_data: Market regime
            ml_prediction: ML system prediction (advisory)
            rl_recommendation: RL agent recommendation (advisory)
            portfolio: Current portfolio state
            open_positions: List of open positions

        Returns:
            Complete trading decision
        """
        if not self.is_enabled:
            return self._default_approval(signal)

        # Get relevant wisdom
        wisdom = self.learner.get_relevant_wisdom(
            pair=pair,
            market_conditions={
                "regime": regime_data.get("regime", "") if regime_data else "",
                "rsi": indicators.get("rsi", 50)
            }
        )

        prompt = self._build_evaluation_prompt(
            pair=pair,
            signal=signal,
            indicators=indicators,
            sentiment_data=sentiment_data,
            orderbook_data=orderbook_data,
            regime_data=regime_data,
            ml_prediction=ml_prediction,
            rl_recommendation=rl_recommendation,
            portfolio=portfolio,
            open_positions=open_positions or [],
            wisdom=wisdom
        )

        try:
            response = await self.gpt.analyze(
                system_prompt=self.CONTROLLER_SYSTEM_PROMPT,
                user_prompt=prompt,
                temperature=0.4,  # Lower for more consistent decisions
                max_tokens=1500,
                json_response=True
            )

            decision = response["data"]
            self.total_cost += response["cost"]
            self.total_decisions += 1

            approved = decision.get("approved", False)
            if approved:
                self.trades_approved += 1
            else:
                self.trades_rejected += 1

            # Log decision
            logger.info(
                f"GPT Trade Decision for {pair}: "
                f"{'APPROVED' if approved else 'REJECTED'} "
                f"(confidence={decision.get('confidence', 0)}%)"
            )

            # Send notification for important decisions
            if self.notify_callback:
                if approved and decision.get("confidence", 0) >= 80:
                    await self._notify(
                        f"🎯 **GPT Aprobó Trade**\n\n"
                        f"Par: {pair}\n"
                        f"Acción: {signal.get('action', 'N/A')}\n"
                        f"Confianza GPT: {decision.get('confidence', 0)}%\n"
                        f"Tamaño: {decision.get('position_size', {}).get('recommendation', 'NORMAL')}\n\n"
                        f"📝 {decision.get('reasoning', 'N/A')}"
                    )
                elif not approved:
                    await self._notify(
                        f"🚫 **GPT Rechazó Trade**\n\n"
                        f"Par: {pair}\n"
                        f"Razón: {decision.get('rejection_reason', 'Riesgo elevado')}"
                    )

            return {
                "success": True,
                "approved": approved,
                "decision": decision,
                "cost": response["cost"],
                "overrides": {
                    "ml_override": decision.get("overrides", {}).get("ml", False),
                    "rl_override": decision.get("overrides", {}).get("rl", False)
                }
            }

        except Exception as e:
            logger.error(f"Signal evaluation failed: {e}")
            # On error, defer to traditional system with reduced size
            return {
                "success": False,
                "approved": True,
                "decision": {
                    "approved": True,
                    "position_size": {"modifier": 0.5},
                    "reason": "GPT unavailable, using conservative fallback"
                },
                "error": str(e)
            }

    def _build_evaluation_prompt(
        self,
        pair: str,
        signal: Dict,
        indicators: Dict,
        sentiment_data: Optional[Dict],
        orderbook_data: Optional[Dict],
        regime_data: Optional[Dict],
        ml_prediction: Optional[Dict],
        rl_recommendation: Optional[Dict],
        portfolio: Optional[Dict],
        open_positions: List[str],
        wisdom: Dict
    ) -> str:
        """Build comprehensive evaluation prompt"""

        current_price = indicators.get("current_price", 0)

        prompt = f"""
EVALÚA ESTE TRADE Y TOMA LA DECISIÓN FINAL

== SEÑAL RECIBIDA ==
Par: {pair}
Acción: {signal.get('action', 'N/A')}
Score del sistema: {signal.get('score', 0)}/10
Confianza del sistema: {signal.get('confidence', 0)}%
Razones: {', '.join(signal.get('reasons', [])[:5])}

== INDICADORES TÉCNICOS ==
Precio: ${current_price:,.2f}
RSI (14): {indicators.get('rsi', 50):.1f}
MACD: {indicators.get('macd', 0):.4f}
MACD Signal: {indicators.get('macd_signal', 0):.4f}
EMA 9/21/50: ${indicators.get('ema_9', 0):,.2f} / ${indicators.get('ema_21', 0):,.2f} / ${indicators.get('ema_50', 0):,.2f}
ATR: {indicators.get('atr', 0):.4f}
ADX: {indicators.get('adx', 0):.1f}
Volumen: {indicators.get('volume_ratio', 1):.1f}x promedio
"""

        if sentiment_data:
            fg = sentiment_data.get('fear_greed_index', 0.5) * 100
            prompt += f"""
== SENTIMENT ==
Fear & Greed: {fg:.0f}/100 ({sentiment_data.get('fear_greed_label', 'Neutral')})
Noticias: {sentiment_data.get('news_sentiment_overall', 0.5):.2f}
Impacto alto: {sentiment_data.get('high_impact_news_count', 0)} noticias
"""

        if orderbook_data:
            prompt += f"""
== ORDER BOOK ==
Presión: {orderbook_data.get('market_pressure', 'NEUTRAL')}
Imbalance: {orderbook_data.get('imbalance', 0):+.2f}
"""

        if regime_data:
            prompt += f"""
== RÉGIMEN DE MERCADO ==
Régimen: {regime_data.get('regime', 'SIDEWAYS')}
Fuerza: {regime_data.get('regime_strength', 'MEDIUM')}
Volatilidad: {regime_data.get('volatility', 'NORMAL')}
"""

        if ml_prediction:
            prompt += f"""
== PREDICCIÓN ML (REFERENCIA - puedes ignorar) ==
Probabilidad WIN: {ml_prediction.get('win_probability', 0.5) * 100:.1f}%
Recomendación: {ml_prediction.get('recommendation', 'N/A')}
Confianza: {ml_prediction.get('confidence', 0)}%
"""

        if rl_recommendation:
            prompt += f"""
== RECOMENDACIÓN RL (REFERENCIA - puedes ignorar) ==
Acción sugerida: {rl_recommendation.get('action', 'N/A')}
Q-Value: {rl_recommendation.get('q_value', 0):.3f}
Exploración: {'Sí' if rl_recommendation.get('is_exploration') else 'No'}
"""

        if portfolio:
            prompt += f"""
== PORTFOLIO ==
Equity: ${portfolio.get('equity', 50000):,.2f}
Posiciones abiertas: {len(open_positions)}
Win Rate sesión: {portfolio.get('win_rate', 0):.1f}%
Drawdown actual: {portfolio.get('current_drawdown', 0):.2f}%
"""

        # Add wisdom
        prompt += "\n" + self.learner.format_wisdom_for_prompt(wisdom)

        prompt += """
== TOMA LA DECISIÓN ==
Responde en JSON:

{
    "approved": true/false,
    "confidence": 75,
    "reasoning": "Por qué apruebas o rechazas este trade (2-3 oraciones)",
    "rejection_reason": "Si rechazas, razón principal",

    "position_size": {
        "recommendation": "FULL/HALF/QUARTER/SKIP",
        "modifier": 1.0,
        "reason": "Por qué este tamaño"
    },

    "leverage": {
        "recommended": 3,
        "max_safe": 5,
        "reason": "Por qué este leverage"
    },

    "risk_management": {
        "stop_loss_pct": 2.0,
        "take_profit_pct": 4.0,
        "trailing_stop": true,
        "risk_reward_ratio": 2.0
    },

    "timing": {
        "urgency": "IMMEDIATE/WAIT/SKIP",
        "wait_for": "Condición si debe esperar"
    },

    "overrides": {
        "ml": false,
        "rl": false,
        "override_reason": "Si ignoras ML/RL, por qué"
    },

    "warnings": [
        "Advertencia 1 si hay riesgos"
    ],

    "alternative_action": "Si rechazas, qué hacer en su lugar"
}

REGLAS:
- approved=true SOLO si confianza >= 65%
- Risk/Reward mínimo 1:1.5
- Máximo 2% riesgo por trade
- Considera la sabiduría aprendida
- Puedes IGNORAR ML/RL si tienes buena razón
"""
        return prompt

    async def manage_open_trade(
        self,
        pair: str,
        trade: Dict,
        current_price: float,
        indicators: Dict,
        market_context: Dict
    ) -> Dict[str, Any]:
        """
        GPT manages an open trade - decide if to hold, adjust, or close

        Args:
            pair: Trading pair
            trade: Current trade data
            current_price: Current market price
            indicators: Current indicators
            market_context: Current market context

        Returns:
            Trade management decision
        """
        if not self.is_enabled:
            return {"action": "HOLD", "reason": "GPT disabled"}

        entry_price = trade.get("entry_price", current_price)
        side = trade.get("side", "LONG")
        current_pnl_pct = self._calculate_pnl_pct(entry_price, current_price, side)

        prompt = f"""
GESTIONA ESTE TRADE ABIERTO

== TRADE ACTUAL ==
Par: {pair}
Lado: {side}
Precio entrada: ${entry_price:,.2f}
Precio actual: ${current_price:,.2f}
P&L actual: {current_pnl_pct:+.2f}%
Tiempo abierto: {trade.get('duration_hours', 0):.1f} horas
Stop Loss: ${trade.get('stop_loss', 0):,.2f}
Take Profit: ${trade.get('take_profit', 0):,.2f}

== INDICADORES ACTUALES ==
RSI: {indicators.get('rsi', 50):.1f}
MACD: {indicators.get('macd', 0):.4f}
Volumen: {indicators.get('volume_ratio', 1):.1f}x

== CONTEXTO ==
Régimen: {market_context.get('regime', 'SIDEWAYS')}
Volatilidad: {market_context.get('volatility', 'NORMAL')}

Responde en JSON:
{{
    "action": "HOLD/CLOSE/ADJUST_SL/ADJUST_TP/PARTIAL_CLOSE",
    "reason": "Por qué esta acción",
    "new_stop_loss": precio (si ADJUST_SL),
    "new_take_profit": precio (si ADJUST_TP),
    "close_percentage": 50 (si PARTIAL_CLOSE),
    "urgency": "IMMEDIATE/SOON/LOW",
    "confidence": 75
}}
"""

        try:
            response = await self.gpt.analyze(
                system_prompt="Eres un gestor de posiciones experto. Protege el capital primero.",
                user_prompt=prompt,
                temperature=0.3,
                max_tokens=500,
                json_response=True
            )

            self.total_cost += response["cost"]
            return {
                "success": True,
                "management": response["data"],
                "cost": response["cost"]
            }

        except Exception as e:
            logger.error(f"Trade management failed: {e}")
            return {"success": False, "action": "HOLD", "error": str(e)}

    async def process_trade_close(
        self,
        trade: Dict,
        market_context: Dict,
        signal_data: Dict
    ) -> Dict[str, Any]:
        """
        Process a closed trade - learn and update

        Args:
            trade: Closed trade data
            market_context: Market context when closed
            signal_data: Original signal data

        Returns:
            Processing result
        """
        pnl = trade.get("pnl", 0)

        # Update streaks
        if pnl > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0

        self.session_pnl += pnl

        # Store in learner
        await self.learner.learn_from_trade(
            trade=trade,
            market_context=market_context,
            signal_data=signal_data
        )

        # Trigger learning session if needed
        result = {"learned": True}

        if self.consecutive_losses >= 3:
            # Emergency learning
            analysis = await self.learner.analyze_losing_streak(
                recent_losses=self.pending_reviews[-5:] if len(self.pending_reviews) >= 5 else [],
                current_params=self._get_current_params()
            )
            result["emergency_analysis"] = analysis

            if analysis.get("success") and self.param_callback:
                for change in analysis.get("analysis", {}).get("parameter_changes", []):
                    self.param_callback(
                        change["parameter"],
                        change["recommended"]
                    )

        # Check if time for learning session
        if len(self.learner.trade_memory) % 10 == 0:
            # Run learning session every 10 trades
            learning = await self.learner.run_learning_session(
                trades=[m["trade"] for m in self.learner.trade_memory[-20:]]
            )
            result["learning_session"] = learning

        return result

    def _calculate_pnl_pct(self, entry: float, current: float, side: str) -> float:
        """Calculate P&L percentage"""
        if side == "LONG":
            return ((current - entry) / entry) * 100
        else:  # SHORT
            return ((entry - current) / entry) * 100

    def _get_current_params(self) -> Dict:
        """Get current parameters from config"""
        if self.config:
            return {
                "RSI_OVERSOLD": getattr(self.config, "RSI_OVERSOLD", 30),
                "RSI_OVERBOUGHT": getattr(self.config, "RSI_OVERBOUGHT", 70),
                "CONSERVATIVE_THRESHOLD": getattr(self.config, "CONSERVATIVE_THRESHOLD", 5.5),
                "BASE_POSITION_SIZE_PCT": getattr(self.config, "BASE_POSITION_SIZE_PCT", 10),
                "MAX_DRAWDOWN_LIMIT": getattr(self.config, "MAX_DRAWDOWN_LIMIT", 10),
            }
        return {}

    def _default_approval(self, signal: Dict) -> Dict:
        """Default approval when GPT is disabled"""
        return {
            "success": True,
            "approved": True,
            "decision": {
                "approved": True,
                "confidence": signal.get("confidence", 50),
                "reasoning": "GPT disabled - using system defaults",
                "position_size": {"recommendation": "FULL", "modifier": 1.0}
            }
        }

    async def _notify(self, message: str):
        """Send notification"""
        if self.notify_callback:
            try:
                import asyncio
                if asyncio.iscoroutinefunction(self.notify_callback):
                    await self.notify_callback(message)
                else:
                    self.notify_callback(message)
            except Exception as e:
                logger.error(f"Notification failed: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get controller statistics"""
        return {
            "enabled": self.is_enabled,
            "total_decisions": self.total_decisions,
            "trades_approved": self.trades_approved,
            "trades_rejected": self.trades_rejected,
            "approval_rate": (
                self.trades_approved / max(1, self.total_decisions) * 100
            ),
            "total_cost": self.total_cost,
            "session_pnl": self.session_pnl,
            "consecutive_wins": self.consecutive_wins,
            "consecutive_losses": self.consecutive_losses,
            "active_trades": len(self.active_trades),
            "learner_stats": self.learner.get_stats()
        }

    def enable(self):
        """Enable GPT control"""
        self.is_enabled = True
        logger.info("GPT Trade Controller ENABLED - Full control active")

    def disable(self):
        """Disable GPT control (fallback to traditional systems)"""
        self.is_enabled = False
        logger.info("GPT Trade Controller DISABLED - Traditional systems active")
