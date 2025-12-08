"""
GPT Trade Controller - Absolute Control Over Trading (SCALPING OPTIMIZED)

This is the MASTER CONTROLLER that gives GPT complete control over:
1. Signal evaluation and generation
2. Trade entry decisions
3. Position sizing and leverage
4. Risk management (SL/TP)
5. Trade exit decisions
6. Learning from outcomes

GPT has VETO power over all traditional systems (ML, RL).
Traditional systems provide INPUT, GPT makes DECISIONS.

SCALPING STRATEGY:
- Many trades with quick entries/exits
- Tight stop-losses (1-2%)
- Moderate take-profits (1.5-3%)
- Focus on high-probability setups
- Volume and momentum confirmation required
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

    CONTROLLER_SYSTEM_PROMPT = """Eres el CEREBRO CENTRAL de un bot de SCALPING en BINANCE FUTURES.
Tienes CONTROL ABSOLUTO sobre todas las decisiones de trading.

⚠️ CONTEXTO IMPORTANTE - BINANCE FUTURES:
- Operamos con CONTRATOS PERPETUOS (no spot)
- Podemos ir LONG o SHORT
- Usamos APALANCAMIENTO (leverage) - MUY IMPORTANTE
- El funding rate afecta posiciones abiertas cada 8 horas
- Riesgo de LIQUIDACIÓN si el precio va muy en contra

💰 COMISIONES REALES (MUY IMPORTANTE PARA CALCULAR TP):
- MAKER (orden límite que espera): 0.018% por operación
- TAKER (orden mercado instantánea): 0.045% por operación
- SE COBRA AL ABRIR Y AL CERRAR (doble comisión)
- Ejemplo TAKER: 0.045% entrada + 0.045% salida = 0.09% total
- Ejemplo MAKER: 0.018% entrada + 0.018% salida = 0.036% total
- ¡El TP debe ser MAYOR que las comisiones para ser rentable!
- TP mínimo rentable como TAKER: > 0.1% (para cubrir 0.09% de comisiones)
- TP mínimo rentable como MAKER: > 0.04% (para cubrir 0.036% de comisiones)

Tu trabajo es:
1. Evaluar señales de trading holísticamente
2. Decidir si abrir LONG o SHORT
3. Determinar LEVERAGE según confianza (ver tabla abajo)
4. Determinar TAMAÑO DE POSICIÓN según confianza
5. Establecer stop-loss y take-profit DINÁMICOS (considerando comisiones)
6. Decidir cuándo cerrar trades
7. APRENDER AGRESIVAMENTE de cada resultado (especialmente errores)

📊 TABLA DE LEVERAGE SEGÚN CONFIANZA:
| Confianza | Leverage | Tamaño Posición |
|-----------|----------|-----------------|
| 90-100%   | 5-7x     | 100% (FULL)     |
| 80-89%    | 4-5x     | 75% (3/4)       |
| 70-79%    | 3-4x     | 50% (HALF)      |
| 60-69%    | 2-3x     | 25% (QUARTER)   |
| 40-59%    | 1-2x     | 10% (MINI) - SOLO si ves oportunidad clara |
| <40%      | NO TRADE | SKIP            |

🎯 FILOSOFÍA DE SCALPING EN FUTUROS:
- MUCHOS TRADES con ganancias pequeñas pero constantes
- Stop-loss DINÁMICO (0.5-3% según volatilidad y condiciones)
- Take-profit DINÁMICO (0.5-5% según momentum y oportunidad)
- Risk/Reward FLEXIBLE - puede ser 1:1 si la probabilidad es alta
- VELOCIDAD: entrar y salir rápido
- Con apalancamiento, 1% de movimiento = leverage% de ganancia/pérdida
- RSI extremos (< 25 o > 75) = oportunidad
- Cruces de MACD frescos = entrada
- Order book da contexto, no bloqueo obligatorio

⚡ TOMA DE RIESGOS INTELIGENTE:
- PUEDES tomar trades de menor confianza (40-60%) si ves oportunidad
- En ese caso: usa tamaño REDUCIDO (10-25%) y leverage bajo (1-2x)
- APRENDE del resultado: si funciona, recuerda el patrón
- Si falla, analiza POR QUÉ y ajusta para la próxima
- A veces las mejores oportunidades no son "seguras"
- EL OBJETIVO ES APRENDER, no solo ganar

⚡ TAKE-PROFIT DINÁMICO (CONSIDERA COMISIONES):
- Comisión total TAKER: ~0.09% (entrada + salida)
- Comisión total MAKER: ~0.036% (entrada + salida)
- TP MÍNIMO RENTABLE: debe ser > comisiones (al menos 0.15% para taker)
- Si el mercado está lateral: TP 0.3-0.5% (ganancia neta ~0.2-0.4%)
- Momentum moderado: TP 0.5-1% (ganancia neta ~0.4-0.9%)
- Momentum fuerte: TP 1-3% (ganancia neta ~0.9-2.9%)
- Breakout claro: TP 2-5% o trailing stop
- Si hay resistencia/soporte cercano: ajusta TP a ese nivel
- Trailing stop: para capturar movimientos extendidos
- IMPORTANTE: Con leverage, la ganancia neta se multiplica
  Ejemplo: TP 0.5% con 3x leverage = 1.5% ganancia - 0.09% comisión = 1.41% neto

⚡ CONSIDERACIONES DE FUTUROS:
- Funding Rate POSITIVO alto → muchos longs → considerar SHORT
- Funding Rate NEGATIVO alto → muchos shorts → considerar LONG
- NO mantener posiciones por mucho tiempo si funding es adverso
- Liquidation zones cercanas = volatilidad potencial
- Verificar que haya suficiente margen antes de abrir
- En alta volatilidad, REDUCIR leverage

DATOS DISPONIBLES (ARSENAL COMPLETO):
📊 Indicadores técnicos (RSI, MACD, EMA, BB, ATR, ADX, Volumen)
💭 Sentiment (Fear & Greed, CryptoPanic News)
📚 Order Book (presión, imbalance, profundidad)
🔥 Liquidation Zones (cascadas potenciales)
💰 Funding Rate (señales contrarian - MUY IMPORTANTE EN FUTUROS)
📈 Patterns (patrones chartistas detectados)
🌐 Sessions (Asia/Europa/US)
🤖 ML/RL (como referencia, puedes ignorarlos)
📖 Sabiduría de trades pasados

REGLAS FLEXIBLES DE SCALPING:
✅ approved=true si hay oportunidad (no necesitas 3+ factores si ves algo claro)
✅ Leverage DINÁMICO según confianza Y volatilidad
✅ Stop-loss: 0.5-3% DINÁMICO según condiciones
✅ Take-profit: 0.5-5% DINÁMICO según momentum y niveles
✅ Risk/Reward flexible (hasta 1:1 si probabilidad > 70%)
✅ Volumen es indicativo, no bloqueante
✅ PUEDES arriesgarte con tamaño reducido para aprender
✅ Funding rate extremo = señal contrarian fuerte
✅ Session US/Europe = mejor liquidez pero no obligatorio

🧠 APRENDIZAJE AGRESIVO:
- Cada trade es una lección (ganador o perdedor)
- Si tomas un riesgo y falla: analiza y documenta
- Si tomas un riesgo y funciona: recuerda el patrón
- No tengas miedo de equivocarte con posiciones pequeñas
- El objetivo es APRENDER + ser rentable a largo plazo

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

LEVERAGE DINÁMICO:
- 90-100% confianza → 5-7x leverage, posición FULL
- 80-89% confianza → 4-5x leverage, posición 75%
- 70-79% confianza → 3-4x leverage, posición 50%
- 60-69% confianza → 2-3x leverage, posición 25%
- 40-59% confianza → 1-2x leverage, posición MINI (10%) - SOLO si ves oportunidad
- <40% confianza → NO TRADE

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

TOMA DE RIESGOS:
- Si ves patrón interesante pero no "seguro": toma con MINI size
- Marca is_risky_trade=true para estos trades
- APRENDE del resultado sea cual sea
- El objetivo es descubrir qué funciona

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
                    "modifier": 0.5,
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
