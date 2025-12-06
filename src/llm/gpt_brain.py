"""
GPT Brain - Central Orchestrator

The master controller that orchestrates all GPT components.
This is the BRAIN of the trading bot, coordinating:
- Meta-reasoning for performance analysis
- Decision explanation for transparency
- Risk assessment for protection
- Strategy optimization for improvement

This module integrates with the existing AutonomyController and provides
a unified interface for GPT-powered trading intelligence.
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from src.llm.gpt_client import GPTClient
from src.llm.gpt_meta_reasoner import GPTMetaReasoner
from src.llm.gpt_decision_explainer import GPTDecisionExplainer
from src.llm.gpt_risk_assessor import GPTRiskAssessor
from src.llm.gpt_strategy_advisor import GPTStrategyAdvisor

logger = logging.getLogger(__name__)


class GPTBrain:
    """
    Central orchestrator for all GPT-powered trading intelligence.
    This is the BRAIN that controls the bot's advanced reasoning.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        param_update_callback: Optional[Callable[[str, Any], bool]] = None,
        notification_callback: Optional[Callable[[str], Any]] = None,
        config: Optional[Any] = None
    ):
        """
        Initialize GPT Brain

        Args:
            api_key: OpenAI API key
            model: GPT model to use
            param_update_callback: Callback to apply parameter changes
            notification_callback: Callback to send notifications (Telegram)
            config: Configuration object
        """
        self.config = config
        self.api_key = api_key
        self.model = model

        # Initialize GPT Client
        self.gpt_client = GPTClient(
            api_key=api_key,
            model=model,
            temperature=0.7,
            max_tokens=2000
        )

        # Initialize all components
        self.meta_reasoner = GPTMetaReasoner(self.gpt_client)
        self.decision_explainer = GPTDecisionExplainer(self.gpt_client)
        self.risk_assessor = GPTRiskAssessor(self.gpt_client)
        self.strategy_advisor = GPTStrategyAdvisor(
            self.gpt_client,
            param_update_callback=param_update_callback
        )

        # Callbacks
        self.param_update_callback = param_update_callback
        self.notification_callback = notification_callback

        # State tracking
        self.is_enabled = True
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.last_trade_result: Optional[str] = None
        self.trades_since_optimization = 0
        self.notable_events: List[str] = []

        # Performance tracking
        self.total_gpt_cost = 0.0
        self.decisions_made = 0
        self.trades_blocked = 0
        self.trades_approved = 0
        self.optimizations_performed = 0

        # Timing
        self.last_optimization_time: Optional[datetime] = None
        self.last_analysis_time: Optional[datetime] = None

        logger.info(f"GPT Brain initialized with model: {model}")

    async def initialize(self):
        """Async initialization tasks"""
        logger.info("GPT Brain online and ready")
        if self.notification_callback:
            await self._notify(
                "🧠 **GPT Brain Activado**\n\n"
                f"Modelo: {self.model}\n"
                "Componentes:\n"
                "• Meta-Reasoner: Analisis de performance\n"
                "• Decision Explainer: Explicar trades\n"
                "• Risk Assessor: Evaluacion de riesgo\n"
                "• Strategy Advisor: Optimizacion autonoma\n\n"
                "El bot ahora tiene razonamiento avanzado."
            )

    async def evaluate_trade(
        self,
        pair: str,
        action: str,
        signal: Dict,
        market_state: Dict,
        portfolio: Dict,
        open_positions: List[str]
    ) -> Dict[str, Any]:
        """
        Comprehensive trade evaluation using GPT

        This is the main entry point for trade decisions.

        Args:
            pair: Trading pair
            action: BUY or SELL
            signal: Signal data
            market_state: Current market state
            portfolio: Portfolio statistics
            open_positions: List of open positions

        Returns:
            Evaluation result with approval/rejection and explanation
        """
        if not self.is_enabled:
            return {
                "approved": True,
                "reason": "GPT Brain disabled",
                "explanation": None,
                "risk_assessment": None
            }

        try:
            # Step 1: Risk Assessment
            position_size = signal.get('position_size_pct', 10.0)

            risk_result = await self.risk_assessor.assess_trade_risk(
                pair=pair,
                action=action,
                position_size_pct=position_size,
                signal=signal,
                portfolio=portfolio,
                market_state=market_state,
                open_positions=open_positions
            )

            self.total_gpt_cost += risk_result.get("cost", 0)

            if not risk_result.get("success"):
                logger.warning(f"Risk assessment failed for {pair}")
                return {
                    "approved": True,
                    "reason": "Risk assessment unavailable",
                    "risk_assessment": None
                }

            assessment = risk_result["assessment"]
            approved = assessment.get("approved", True)

            # Step 2: Generate Explanation
            explanation = None
            if approved:
                explanation = await self.decision_explainer.explain_trade_decision(
                    pair=pair,
                    action=action,
                    signal=signal,
                    ml_prediction=signal.get('ml'),
                    rl_decision=signal.get('rl_decision'),
                    market_state=market_state,
                    arsenal_data=signal.get('arsenal_analysis')
                )

            # Track statistics
            self.decisions_made += 1
            if approved:
                self.trades_approved += 1
            else:
                self.trades_blocked += 1
                self.notable_events.append(
                    f"Trade bloqueado en {pair}: {assessment.get('reason', 'Unknown')}"
                )

            # Step 3: Apply position adjustment if recommended
            position_adjustment = assessment.get("position_adjustment", {})
            adjusted_size = position_adjustment.get("recommended_size_pct", position_size)

            result = {
                "approved": approved,
                "reason": assessment.get("reason", ""),
                "risk_level": assessment.get("risk_level", "MEDIUM"),
                "risk_score": assessment.get("risk_score", 50),
                "risk_factors": assessment.get("risk_factors", []),
                "warnings": assessment.get("warnings", []),
                "explanation": explanation,
                "position_adjustment": adjusted_size if adjusted_size != position_size else None,
                "risk_assessment": assessment
            }

            # Log result
            logger.info(
                f"GPT Trade Evaluation for {pair} {action}: "
                f"{'APPROVED' if approved else 'BLOCKED'} "
                f"(risk={assessment.get('risk_level', 'UNKNOWN')})"
            )

            # Send notification if blocked
            if not approved and self.notification_callback:
                await self._notify(
                    f"🚫 **Trade Bloqueado por GPT**\n\n"
                    f"Par: {pair}\n"
                    f"Accion: {action}\n"
                    f"Riesgo: {assessment.get('risk_level', 'HIGH')}\n"
                    f"Razon: {assessment.get('reason', 'N/A')}"
                )

            return result

        except Exception as e:
            logger.error(f"Trade evaluation failed: {e}")
            return {
                "approved": True,
                "reason": f"Evaluation error: {str(e)}",
                "risk_assessment": None
            }

    async def process_trade_result(
        self,
        trade: Dict,
        market_state: Dict,
        portfolio: Dict
    ):
        """
        Process trade result for learning and adaptation

        Args:
            trade: Closed trade data
            market_state: Market state at close
            portfolio: Current portfolio stats
        """
        if not self.is_enabled:
            return

        try:
            pnl = trade.get('pnl', 0)
            pnl_pct = trade.get('pnl_pct', 0)

            # Update streaks
            if pnl > 0:
                self.consecutive_wins += 1
                self.consecutive_losses = 0
                self.last_trade_result = "WIN"
            else:
                self.consecutive_losses += 1
                self.consecutive_wins = 0
                self.last_trade_result = "LOSS"

            self.trades_since_optimization += 1

            # Generate explanation for closed trade
            explanation = await self.decision_explainer.explain_trade_closed(
                trade=trade,
                market_conditions=market_state
            )

            # React to losing streak
            if self.consecutive_losses >= 3:
                logger.warning(f"Losing streak detected: {self.consecutive_losses} losses")
                await self._handle_losing_streak(trade, portfolio)

            # React to winning streak
            elif self.consecutive_wins >= 5:
                logger.info(f"Winning streak detected: {self.consecutive_wins} wins")
                await self._handle_winning_streak(portfolio)

            # Send notification with explanation
            if self.notification_callback:
                emoji = "✅" if pnl > 0 else "❌"
                await self._notify(
                    f"{emoji} **Trade Cerrado**\n\n"
                    f"Par: {trade.get('pair', 'N/A')}\n"
                    f"P&L: {pnl_pct:+.2f}%\n\n"
                    f"📝 {explanation}"
                )

            # Check if optimization is needed
            await self._check_optimization_trigger(portfolio)

        except Exception as e:
            logger.error(f"Error processing trade result: {e}")

    async def _handle_losing_streak(self, trade: Dict, portfolio: Dict):
        """Handle losing streak with strategic adjustments"""
        try:
            # Get current params
            current_params = self._get_current_params()

            # Get GPT reaction
            reaction = await self.strategy_advisor.react_to_loss(
                trade=trade,
                consecutive_losses=self.consecutive_losses,
                current_params=current_params
            )

            self.total_gpt_cost += reaction.get("cost", 0)

            if reaction.get("should_pause"):
                pause_minutes = reaction.get("pause_minutes", 30)
                self.notable_events.append(
                    f"Trading pausado por {pause_minutes} min debido a {self.consecutive_losses} perdidas"
                )
                if self.notification_callback:
                    await self._notify(
                        f"⚠️ **Alerta de Perdidas**\n\n"
                        f"Perdidas consecutivas: {self.consecutive_losses}\n"
                        f"Severidad: {reaction.get('severity', 'HIGH')}\n\n"
                        f"📝 {reaction.get('message', 'Evaluando situacion...')}"
                    )

            # Apply changes
            for change in reaction.get("changes", []):
                if self.param_update_callback:
                    self.param_update_callback(
                        change["parameter"],
                        change["new_value"]
                    )
                    logger.info(
                        f"GPT adjusted {change['parameter']}: {change['new_value']} "
                        f"(reason: {change['reason']})"
                    )

        except Exception as e:
            logger.error(f"Error handling losing streak: {e}")

    async def _handle_winning_streak(self, portfolio: Dict):
        """Handle winning streak with potential aggressive adjustments"""
        try:
            current_params = self._get_current_params()

            reaction = await self.strategy_advisor.react_to_win_streak(
                consecutive_wins=self.consecutive_wins,
                current_params=current_params
            )

            if reaction.get("should_adjust"):
                for change in reaction.get("changes", []):
                    if self.param_update_callback:
                        self.param_update_callback(
                            change["parameter"],
                            change["new_value"]
                        )
                        logger.info(
                            f"GPT adjusted {change['parameter']}: {change['new_value']} "
                            f"(win streak adjustment)"
                        )

                if self.notification_callback:
                    await self._notify(
                        f"🔥 **Racha Ganadora: {self.consecutive_wins}**\n\n"
                        f"GPT ha ajustado parametros para aprovechar el momentum.\n"
                        f"⚠️ {reaction.get('warning', 'Mantener precaucion')}"
                    )

        except Exception as e:
            logger.error(f"Error handling winning streak: {e}")

    async def _check_optimization_trigger(self, portfolio: Dict):
        """Check if we should trigger a full optimization"""
        should_optimize = False
        reason = ""

        # Trigger conditions
        if self.trades_since_optimization >= 20:
            should_optimize = True
            reason = f"{self.trades_since_optimization} trades since last optimization"

        elif self.strategy_advisor.should_optimize(min_interval_hours=2.0):
            should_optimize = True
            reason = "Time-based optimization trigger"

        elif self.consecutive_losses >= 5:
            should_optimize = True
            reason = f"Emergency: {self.consecutive_losses} consecutive losses"

        if should_optimize:
            logger.info(f"Triggering GPT optimization: {reason}")
            await self.run_full_optimization(portfolio, reason)
            self.trades_since_optimization = 0

    async def run_full_optimization(
        self,
        portfolio: Dict,
        trigger_reason: str = "Manual trigger"
    ) -> Dict[str, Any]:
        """
        Run full strategy optimization

        Args:
            portfolio: Current portfolio stats
            trigger_reason: Why optimization was triggered

        Returns:
            Optimization results
        """
        if not self.is_enabled:
            return {"success": False, "reason": "GPT Brain disabled"}

        try:
            logger.info(f"Running full GPT optimization: {trigger_reason}")

            # Get recent trades (this would need to be passed or fetched)
            trades = []  # TODO: Get from paper_trader

            # Get current params
            current_params = self._get_current_params()

            # Get market context
            market_context = {
                "trigger_reason": trigger_reason,
                "consecutive_losses": self.consecutive_losses,
                "consecutive_wins": self.consecutive_wins,
                "trades_since_optimization": self.trades_since_optimization
            }

            # Run optimization
            result = await self.strategy_advisor.optimize_strategy(
                trades=trades,
                current_params=current_params,
                portfolio_stats=portfolio,
                market_context=market_context,
                auto_apply=True
            )

            self.total_gpt_cost += result.get("cost", 0)
            self.optimizations_performed += 1
            self.last_optimization_time = datetime.now()

            # Notify about changes
            if result.get("applied_changes") and self.notification_callback:
                changes_text = "\n".join([
                    f"• {c['parameter']}: {c.get('old_value')} → {c['new_value']}"
                    for c in result["applied_changes"][:5]
                ])
                await self._notify(
                    f"🧠 **GPT Optimization Complete**\n\n"
                    f"Trigger: {trigger_reason}\n"
                    f"Direccion: {result.get('strategy_direction', 'N/A')}\n"
                    f"Confianza: {result.get('confidence', 0)}%\n\n"
                    f"**Cambios Aplicados:**\n{changes_text}"
                )

            return result

        except Exception as e:
            logger.error(f"Full optimization failed: {e}")
            return {"success": False, "error": str(e)}

    async def run_performance_analysis(
        self,
        trades: List[Dict],
        portfolio: Dict
    ) -> Dict[str, Any]:
        """
        Run comprehensive performance analysis

        Args:
            trades: List of trades to analyze
            portfolio: Current portfolio stats

        Returns:
            Analysis results
        """
        if not self.is_enabled:
            return {"success": False, "reason": "GPT Brain disabled"}

        try:
            current_params = self._get_current_params()
            market_context = {
                "consecutive_losses": self.consecutive_losses,
                "consecutive_wins": self.consecutive_wins
            }

            result = await self.meta_reasoner.analyze_performance(
                trades=trades,
                current_params=current_params,
                portfolio_stats=portfolio,
                market_context=market_context
            )

            self.total_gpt_cost += result.get("cost", 0)
            self.last_analysis_time = datetime.now()

            # Apply recommendations if high confidence
            if result.get("success"):
                analysis = result.get("analysis", {})
                for rec in analysis.get("recommendations", []):
                    if rec.get("confidence", 0) >= 80 and rec.get("priority") == "HIGH":
                        if self.param_update_callback:
                            self.param_update_callback(
                                rec["parameter"],
                                rec["recommended_value"]
                            )
                            logger.info(
                                f"Applied high-confidence recommendation: "
                                f"{rec['parameter']} = {rec['recommended_value']}"
                            )

            return result

        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            return {"success": False, "error": str(e)}

    async def get_market_insight(
        self,
        pair: str,
        market_data: Dict
    ) -> str:
        """
        Get GPT insight about current market conditions

        Args:
            pair: Trading pair
            market_data: Current market data

        Returns:
            Market insight text
        """
        try:
            prompt = f"""
Analiza brevemente el estado actual del mercado para {pair}:

Datos: {json.dumps(market_data, indent=2)}

Responde en 2-3 oraciones con:
1. Estado actual del mercado
2. Principales riesgos u oportunidades
3. Recomendacion general
"""
            response = await self.gpt_client.analyze(
                system_prompt="Eres un analista de mercados crypto. Responde en español.",
                user_prompt=prompt,
                temperature=0.5,
                max_tokens=200
            )
            self.total_gpt_cost += response.get("cost", 0)
            return response["content"]

        except Exception as e:
            logger.error(f"Market insight failed: {e}")
            return "Analisis de mercado no disponible"

    async def generate_daily_report(
        self,
        trades: List[Dict],
        portfolio: Dict
    ) -> str:
        """
        Generate comprehensive daily report

        Args:
            trades: Today's trades
            portfolio: Current portfolio stats

        Returns:
            Daily report text
        """
        try:
            summary = await self.decision_explainer.generate_daily_summary(
                trades=trades,
                portfolio_stats=portfolio,
                notable_events=self.notable_events[-10:]
            )

            # Clear notable events after report
            self.notable_events = []

            return summary

        except Exception as e:
            logger.error(f"Daily report generation failed: {e}")
            return "Reporte diario no disponible"

    def _get_current_params(self) -> Dict:
        """Get current parameter values"""
        if self.config:
            return {
                "CONSERVATIVE_THRESHOLD": getattr(self.config, 'CONSERVATIVE_THRESHOLD', 5.5),
                "FLASH_THRESHOLD": getattr(self.config, 'FLASH_THRESHOLD', 4.5),
                "FLASH_MIN_CONFIDENCE": getattr(self.config, 'FLASH_MIN_CONFIDENCE', 55),
                "BASE_POSITION_SIZE_PCT": getattr(self.config, 'BASE_POSITION_SIZE_PCT', 10.0),
                "MAX_POSITION_SIZE_PCT": getattr(self.config, 'MAX_POSITION_SIZE_PCT', 12.0),
                "RSI_OVERSOLD": getattr(self.config, 'RSI_OVERSOLD', 35),
                "RSI_OVERBOUGHT": getattr(self.config, 'RSI_OVERBOUGHT', 65),
                "MIN_CONFIDENCE_FOR_FUTURES": getattr(self.config, 'MIN_CONFIDENCE_FOR_FUTURES', 70.0),
                "TRAILING_DISTANCE_PCT": getattr(self.config, 'TRAILING_DISTANCE_PCT', 0.4),
                "MAX_DRAWDOWN_LIMIT": getattr(self.config, 'MAX_DRAWDOWN_LIMIT', 10.0),
                "MAX_POSITIONS": getattr(self.config, 'MAX_POSITIONS', 5),
            }
        return {}

    async def _notify(self, message: str):
        """Send notification"""
        if self.notification_callback:
            try:
                if asyncio.iscoroutinefunction(self.notification_callback):
                    await self.notification_callback(message)
                else:
                    self.notification_callback(message)
            except Exception as e:
                logger.error(f"Notification failed: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get GPT Brain statistics"""
        return {
            "enabled": self.is_enabled,
            "model": self.model,
            "total_gpt_cost": self.total_gpt_cost,
            "decisions_made": self.decisions_made,
            "trades_approved": self.trades_approved,
            "trades_blocked": self.trades_blocked,
            "block_rate": (
                self.trades_blocked / max(1, self.decisions_made) * 100
            ),
            "optimizations_performed": self.optimizations_performed,
            "consecutive_losses": self.consecutive_losses,
            "consecutive_wins": self.consecutive_wins,
            "last_optimization": (
                self.last_optimization_time.isoformat()
                if self.last_optimization_time else None
            ),
            "last_analysis": (
                self.last_analysis_time.isoformat()
                if self.last_analysis_time else None
            ),
            "notable_events_count": len(self.notable_events),
            "gpt_client_stats": self.gpt_client.get_usage_stats()
        }

    def enable(self):
        """Enable GPT Brain"""
        self.is_enabled = True
        logger.info("GPT Brain enabled")

    def disable(self):
        """Disable GPT Brain"""
        self.is_enabled = False
        logger.info("GPT Brain disabled")

    def reset_streaks(self):
        """Reset win/loss streaks"""
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.last_trade_result = None

    async def emergency_stop(self, reason: str):
        """Emergency stop with notification"""
        self.disable()
        if self.notification_callback:
            await self._notify(
                f"🚨 **GPT Brain Emergency Stop**\n\n"
                f"Razon: {reason}\n\n"
                f"El razonamiento GPT ha sido desactivado. "
                f"El bot continuara operando sin analisis avanzado."
            )
