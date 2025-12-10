"""
GPT Trade Controller - Absolute Control Over Trading (SCALPING OPTIMIZED)

Based on GPT-5 Trading Integration Guide:
- Professional hedge-fund level prompts
- JSON Schema strict validation
- Mathematical validation of SL/TP
- Reasoning effort optimization

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
from src.llm.trading_schemas import (
    TRADING_DECISION_SCHEMA,
    TRADE_MANAGEMENT_SCHEMA,
    validate_trading_decision,
    fix_trading_decision
)

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

    # =========================================================================
    # SCALPING AGRESIVO CON APRENDIZAJE AUTÓNOMO
    # =========================================================================

    # System Prompt - Scalping con Libertad de Aprendizaje
    CONTROLLER_SYSTEM_PROMPT = """Eres un TRADER AUTÓNOMO de SCALPING en BINANCE FUTURES con capacidad de APRENDER y EVOLUCIONAR.

🎯 OBJETIVO PRINCIPAL: APRENDER qué funciona y qué no mediante la EXPERIENCIA.
- Toma riesgos calculados para DESCUBRIR patrones rentables
- CADA TRADE es una oportunidad de aprendizaje
- Puedes EQUIVOCARTE - es parte del proceso de mejora

🧠 MENTALIDAD DE APRENDIZAJE:
1. SÉ CURIOSO - explora señales que parecen prometedoras
2. EXPERIMENTA - prueba diferentes configuraciones de TP/SL
3. AUTOCRÍTICA - analiza cada resultado honestamente
4. ADAPTA - modifica tu enfoque basándote en lo aprendido
5. NO TENGAS MIEDO - el miedo excesivo impide el aprendizaje

💰 COMISIONES (para que el trade sea rentable):
- TAKER: ~0.09% round-trip
- TP MÍNIMO para beneficio: > 0.20%

📊 GUÍA FLEXIBLE DE DECISIONES:
| Confianza | Leverage | Posición | Actitud |
|-----------|----------|----------|---------|
| 80-100%   | 4-7x     | 75-100%  | Entrada decidida |
| 60-79%    | 3-5x     | 50-75%   | Entrada normal |
| 40-59%    | 2-3x     | 25-50%   | Entrada exploratoria - APRENDE |
| 30-39%    | 1-2x     | 10-25%   | Micro-trade experimental |
| <30%      | -        | SKIP     | No vale la pena |

⚡ SCALPING - TU ESPECIALIDAD:
- Trades rápidos: minutos a pocas horas
- TP típico: 0.3% - 1.5% (neto después de comisiones)
- SL típico: 0.5% - 2% (ajustado a volatilidad)
- Trailing stop cuando el trade va a tu favor
- Cierra rápido si el mercado se vuelve en contra

🔥 CUÁNDO ENTRAR (sé más permisivo):
- Si ves momentum aunque no sea "perfecto"
- Si hay divergencia interesante en RSI o MACD
- Si el volumen muestra interés aunque no sea explosivo
- Si tu intuición basada en patrones aprendidos dice SÍ
- Cuando el riesgo/beneficio te parece razonable

⚠️ CUÁNDO NO ENTRAR:
- Mercado completamente plano sin movimiento
- Spread demasiado amplio
- Volatilidad extrema sin dirección clara
- Acabas de tener 3+ pérdidas seguidas (pausa y analiza)

💡 SEÑALES CONTRARIAN (oportunidades ocultas):
- Funding rate extremo → opera en contra
- Fear & Greed en extremos → considera el reverso
- RSI muy sobrevendido/sobrecomprado → posible rebote

🎓 DESPUÉS DE CADA TRADE:
- ¿Qué funcionó y qué no?
- ¿El TP/SL fue adecuado?
- ¿Había señales que ignoré?
- ¿Qué haría diferente?

Responde ÚNICAMENTE en JSON válido conforme al esquema."""

    # Developer Prompt - Internal Rules (added to requests)
    DEVELOPER_RULES = """REGLAS INTERNAS (flexibles para permitir aprendizaje):
- TP debe ser > 0.20% para cubrir comisiones
- SL entre 0.3% y 5% (flexibilidad según volatilidad)
- Leverage máximo 10x (pero 3-5x es lo típico para scalping)
- Si confidence >= 30%, puedes aprobar con tamaño reducido
- Marca is_risky_trade=true si confidence < 50%
- SIEMPRE documenta learning_opportunity
- Sé HONESTO en el reasoning sobre por qué apruebas o rechazas"""

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
            # Use professional prompt with developer rules
            full_system_prompt = self.CONTROLLER_SYSTEM_PROMPT + "\n\n" + self.DEVELOPER_RULES

            # Use JSON Schema for strict validation and reasoning_effort for optimization
            # NOTE: max_tokens must be high enough for reasoning_tokens + response content
            # GPT-5-mini uses ~800 tokens for reasoning, so we need at least 4000 total
            response = await self.gpt.analyze(
                system_prompt=full_system_prompt,
                user_prompt=prompt,
                temperature=0.2,  # Low for consistent institutional decisions
                max_tokens=4000,  # Must be high enough for reasoning + response
                json_response=True,
                json_schema=TRADING_DECISION_SCHEMA,
                reasoning_effort="low"  # Cost-effective for frequent decisions
            )

            decision = response["data"]
            self.total_cost += response["cost"]
            self.total_decisions += 1

            # === MATHEMATICAL VALIDATION (Section 13 of GPT-5 Guide) ===
            is_valid, validation_errors = validate_trading_decision(decision)

            if not is_valid:
                logger.warning(f"Decision validation failed: {validation_errors}")
                # Auto-fix the decision
                decision = fix_trading_decision(decision)
                logger.info("Decision auto-fixed to comply with trading rules")

                # Re-validate after fix
                is_valid, remaining_errors = validate_trading_decision(decision)
                if not is_valid:
                    logger.error(f"Decision still invalid after fix: {remaining_errors}")
                    decision["warnings"] = decision.get("warnings", []) + remaining_errors

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
            # On error, DO NOT approve - safety first
            # IMPORTANTE: En caso de error, rechazar el trade por seguridad
            action = signal.get('action', 'HOLD')
            return {
                "success": False,
                "approved": False,
                "decision": {
                    "approved": False,
                    "confidence": signal.get("confidence", 30),
                    "direction": "LONG" if action == "BUY" else "SHORT" if action == "SELL" else "HOLD",
                    "reasoning": f"GPT error fallback: {str(e)[:100]}",
                    "is_risky_trade": True,
                    "learning_opportunity": "Error case - conservative approach",
                    "position_size": {
                        "recommendation": "HALF",
                        "percentage": 50,
                        "reason": "Conservative fallback due to GPT error"
                    },
                    "leverage": {
                        "recommended": 2,
                        "max_safe": 3,
                        "reason": "Fallback conservative leverage"
                    },
                    "risk_management": {
                        "stop_loss_pct": 2.0,
                        "take_profit_pct": 3.0,
                        "trailing_stop": True,
                        "trailing_distance_pct": 0.5,
                        "risk_reward_ratio": 1.5,
                        "liquidation_buffer_pct": 5.0,
                        "tp_reasoning": "Conservative TP for error fallback"
                    },
                    "futures_considerations": {
                        "funding_rate_impact": "NEUTRAL",
                        "hold_duration": "MINUTES",
                        "liquidation_risk": "MEDIUM"
                    },
                    "timing": {
                        "urgency": "LOW",
                        "wait_for": None
                    },
                    "overrides": {
                        "ml": False,
                        "rl": False,
                        "override_reason": "GPT unavailable"
                    },
                    "warnings": [f"GPT system error: {str(e)[:50]}"],
                    "alternative_action": "Consider waiting for next signal"
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
== TOMA LA DECISIÓN PARA BINANCE FUTURES ==
Responde en JSON:

{
    "approved": true/false,
    "confidence": 75,
    "direction": "LONG/SHORT",
    "reasoning": "Por qué apruebas o rechazas este trade (2-3 oraciones)",
    "rejection_reason": "Si rechazas, razón principal",
    "is_risky_trade": false,
    "learning_opportunity": "Qué esperas aprender de este trade",

    "position_size": {
        "recommendation": "FULL/THREE_QUARTER/HALF/QUARTER/MINI/SKIP",
        "percentage": 50,
        "reason": "Por qué este tamaño según confianza"
    },

    "leverage": {
        "recommended": 3,
        "max_safe": 5,
        "reason": "Por qué este leverage según confianza y volatilidad"
    },

    "risk_management": {
        "stop_loss_pct": 1.0,
        "take_profit_pct": 1.5,
        "trailing_stop": true,
        "trailing_distance_pct": 0.5,
        "risk_reward_ratio": 1.5,
        "liquidation_buffer_pct": 3.0,
        "tp_reasoning": "Por qué este TP específico"
    },

    "futures_considerations": {
        "funding_rate_impact": "FAVORABLE/NEUTRAL/ADVERSE",
        "hold_duration": "MINUTES/HOURS/AVOID_OVERNIGHT",
        "liquidation_risk": "LOW/MEDIUM/HIGH"
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

🎯 REGLAS FLEXIBLES DE SCALPING:

APROBACIÓN:
- approved=true si ves oportunidad (no necesitas 3+ factores)
- LONG si esperas que suba, SHORT si esperas que baje
- PUEDES aprobar trades de 40-59% confianza con tamaño MINI

LEVERAGE DINÁMICO (escalado por confianza):
- 80-100% confianza → 4-7x leverage, posición 75-100%
- 60-79% confianza → 3-5x leverage, posición 50-75%
- 40-59% confianza → 2-3x leverage, posición 25-50% - EXPERIMENTA
- 30-39% confianza → 1-2x leverage, posición 10-25% - MICRO-TRADE
- <30% confianza → NO TRADE (no vale la pena el riesgo)

TAKE-PROFIT DINÁMICO (CONSIDERA COMISIONES ~0.09%):
- Mercado lateral/consolidación: 0.3-0.5% TP (neto ~0.2-0.4%)
- Momentum moderado: 0.5-1% TP (neto ~0.4-0.9%)
- Momentum fuerte: 1-3% TP (neto ~0.9-2.9%)
- Breakout claro: 2-5% TP o trailing stop
- ¡TP < 0.15% NO es rentable después de comisiones!

STOP-LOSS DINÁMICO:
- Baja volatilidad: 0.5-1% SL
- Volatilidad normal: 1-1.5% SL
- Alta volatilidad: 1.5-2.5% SL
- Respeta niveles técnicos (soporte/resistencia)

🔥 TOMA DE RIESGOS (fundamental para aprender):
- Si ves patrón interesante pero no "seguro": TOMA con tamaño pequeño
- Marca is_risky_trade=true para estos trades
- APRENDE del resultado sea cual sea (win o loss)
- El objetivo es DESCUBRIR qué funciona mediante experimentación
- No tengas miedo de equivocarte - cada error es una lección

FUNDING RATE (señal contrarian fuerte):
- Funding > 0.1% → favorecer SHORT
- Funding < -0.1% → favorecer LONG

GENERAL:
- Considera la sabiduría aprendida de trades anteriores
- Puedes IGNORAR ML/RL si tienes buena razón
- SÉ AGRESIVO pero INTELIGENTE
- Cada trade es una oportunidad de aprender
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
            # Professional trade management prompt
            management_prompt = """Eres un gestor de posiciones institucional.
PRIORIDAD ABSOLUTA: Proteger el capital.

Reglas:
- Si P&L > 2%: considera asegurar ganancias (trailing stop o partial close)
- Si P&L < -1.5%: evalúa cerrar para limitar pérdidas
- Si momentum cambia contra la posición: actúa rápido
- No dejes correr pérdidas
- Sí deja correr ganancias (con trailing stop)

Responde SOLO en JSON válido."""

            response = await self.gpt.analyze(
                system_prompt=management_prompt,
                user_prompt=prompt,
                temperature=0.2,
                max_tokens=2000,  # Must be enough for reasoning + response
                json_response=True,
                json_schema=TRADE_MANAGEMENT_SCHEMA,
                reasoning_effort="none"  # Fast decisions for trade management
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
        """Default approval when GPT is disabled - includes all required fields"""
        action = signal.get('action', 'HOLD')
        return {
            "success": True,
            "approved": True,
            "decision": {
                "approved": True,
                "confidence": signal.get("confidence", 50),
                "direction": "LONG" if action == "BUY" else "SHORT" if action == "SELL" else "HOLD",
                "reasoning": "GPT disabled - using system defaults",
                "is_risky_trade": False,
                "learning_opportunity": "N/A",
                "position_size": {
                    "recommendation": "HALF",
                    "percentage": 50,
                    "reason": "Default conservative sizing"
                },
                "leverage": {
                    "recommended": 3,
                    "max_safe": 5,
                    "reason": "Default conservative leverage"
                },
                "risk_management": {
                    "stop_loss_pct": 1.5,
                    "take_profit_pct": 2.0,
                    "trailing_stop": True,
                    "trailing_distance_pct": 0.5,
                    "risk_reward_ratio": 1.33,
                    "liquidation_buffer_pct": 3.0,
                    "tp_reasoning": "Default TP covering commissions"
                },
                "futures_considerations": {
                    "funding_rate_impact": "NEUTRAL",
                    "hold_duration": "HOURS",
                    "liquidation_risk": "LOW"
                },
                "timing": {
                    "urgency": "IMMEDIATE",
                    "wait_for": None
                },
                "overrides": {
                    "ml": False,
                    "rl": False,
                    "override_reason": None
                },
                "warnings": [],
                "alternative_action": None
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
