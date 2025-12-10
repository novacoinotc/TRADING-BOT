"""
GPT Strategy Advisor - Phase 4

The most powerful component: GPT has FULL CONTROL to modify bot parameters.
This module enables autonomous strategy optimization through GPT reasoning.
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from src.llm.gpt_client import GPTClient

logger = logging.getLogger(__name__)


class GPTStrategyAdvisor:
    """
    GPT-powered strategy advisor with FULL CONTROL over bot parameters.
    Can autonomously modify any trading parameter based on reasoning.
    """

    SYSTEM_PROMPT = """Eres el CEREBRO ESTRATEGICO de un bot de trading autonomo de criptomonedas.
Tienes CONTROL TOTAL sobre todos los parametros del bot.

Tu objetivo: MAXIMIZAR GANANCIAS mientras CONTROLAS EL RIESGO.

Puedes modificar:
1. THRESHOLDS: Cuando entrar/salir de trades
2. POSITION SIZING: Cuanto capital arriesgar
3. INDICADORES: Configuracion de RSI, MACD, etc.
4. RISK MANAGEMENT: Stop loss, take profit, trailing stops
5. LEVERAGE: Cuanto apalancamiento usar
6. ML/RL PARAMS: Parametros de machine learning

REGLAS CRITICAS:
- Cambios pequenos (<20% del valor actual) son seguros
- Cambios grandes (>50%) requieren alta confianza
- Si el bot esta perdiendo, se MAS conservador
- Si el bot esta ganando, puedes ser MAS agresivo (con cuidado)
- NUNCA pongas position_size > 15%
- NUNCA pongas leverage > 10x (límite de seguridad del sistema)
- SIEMPRE justifica cada cambio

Responde siempre en español con formato JSON estructurado."""

    # Parameters that can be modified with their valid ranges
    MODIFIABLE_PARAMS = {
        # Trading Thresholds
        "CONSERVATIVE_THRESHOLD": {"min": 4.0, "max": 8.0, "default": 5.5},
        "FLASH_THRESHOLD": {"min": 3.0, "max": 7.0, "default": 4.5},
        "FLASH_MIN_CONFIDENCE": {"min": 40, "max": 80, "default": 55},

        # Position Sizing
        "BASE_POSITION_SIZE_PCT": {"min": 1.0, "max": 15.0, "default": 10.0},
        "MAX_POSITION_SIZE_PCT": {"min": 5.0, "max": 15.0, "default": 12.0},

        # Technical Indicators
        "RSI_OVERSOLD": {"min": 20, "max": 40, "default": 35},
        "RSI_OVERBOUGHT": {"min": 60, "max": 80, "default": 65},

        # Futures/Leverage
        "MIN_CONFIDENCE_FOR_FUTURES": {"min": 50.0, "max": 90.0, "default": 70.0},
        "MIN_WINRATE_FOR_FUTURES": {"min": 40.0, "max": 70.0, "default": 55.0},
        "CONSERVATIVE_LEVERAGE": {"min": 2, "max": 5, "default": 3},
        "BALANCED_LEVERAGE": {"min": 5, "max": 12, "default": 8},
        "AGGRESSIVE_LEVERAGE": {"min": 10, "max": 20, "default": 15},

        # Trailing Stops
        "TRAILING_DISTANCE_PCT": {"min": 0.2, "max": 1.0, "default": 0.4},
        "BREAKEVEN_AFTER_PCT": {"min": 0.3, "max": 1.5, "default": 0.5},
        "LOCK_PROFIT_STEP_PCT": {"min": 0.2, "max": 1.0, "default": 0.5},

        # Risk Management
        "MAX_DRAWDOWN_LIMIT": {"min": 5.0, "max": 20.0, "default": 10.0},
        "MAX_POSITIONS": {"min": 3, "max": 10, "default": 5},

        # Arsenal Advanced
        "HIGH_CORRELATION_THRESHOLD": {"min": 0.5, "max": 0.9, "default": 0.7},
        "FUNDING_EXTREME_POSITIVE": {"min": 0.05, "max": 0.20, "default": 0.10},
        "FUNDING_EXTREME_NEGATIVE": {"min": -0.20, "max": -0.05, "default": -0.10},
        "MIN_PATTERN_CONFIDENCE": {"min": 0.5, "max": 0.9, "default": 0.7},
    }

    def __init__(
        self,
        gpt_client: GPTClient,
        param_update_callback: Optional[Callable[[str, Any], bool]] = None
    ):
        """
        Initialize Strategy Advisor

        Args:
            gpt_client: Initialized GPT client
            param_update_callback: Callback to apply parameter changes
        """
        self.gpt = gpt_client
        self.param_update_callback = param_update_callback
        self.change_history: List[Dict] = []
        self.pending_changes: List[Dict] = []
        self.last_optimization: Optional[datetime] = None
        self.optimization_count = 0

    async def optimize_strategy(
        self,
        trades: List[Dict],
        current_params: Dict,
        portfolio_stats: Dict,
        market_context: Dict,
        auto_apply: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze performance and optimize strategy parameters

        Args:
            trades: Recent trades for analysis
            current_params: Current parameter values
            portfolio_stats: Portfolio statistics
            market_context: Current market context
            auto_apply: Whether to automatically apply changes

        Returns:
            Optimization results with parameter changes
        """
        prompt = self._build_optimization_prompt(
            trades, current_params, portfolio_stats, market_context
        )

        try:
            response = await self.gpt.analyze(
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=prompt,
                temperature=0.3,  # Low temperature for consistent strategy decisions
                max_tokens=2000,
                json_response=True
            )

            result = response["data"]

            # Validate and apply changes
            applied_changes = []
            rejected_changes = []

            for change in result.get("parameter_changes", []):
                param = change.get("parameter")
                new_value = change.get("new_value")

                if self._validate_change(param, new_value):
                    if auto_apply and self.param_update_callback:
                        success = self.param_update_callback(param, new_value)
                        if success:
                            applied_changes.append(change)
                            self._record_change(param, change.get("old_value"), new_value, change.get("reason"))
                        else:
                            rejected_changes.append({**change, "rejection_reason": "Callback failed"})
                    else:
                        self.pending_changes.append(change)
                        applied_changes.append({**change, "status": "pending"})
                else:
                    rejected_changes.append({**change, "rejection_reason": "Validation failed"})

            self.last_optimization = datetime.now()
            self.optimization_count += 1

            logger.info(
                f"GPT Strategy Optimization: "
                f"{len(applied_changes)} applied, {len(rejected_changes)} rejected, "
                f"${response['cost']:.4f}"
            )

            return {
                "success": True,
                "analysis": result.get("analysis", {}),
                "applied_changes": applied_changes,
                "rejected_changes": rejected_changes,
                "strategy_direction": result.get("strategy_direction", "MAINTAIN"),
                "confidence": result.get("confidence", 50),
                "next_review": result.get("next_review_hours", 2),
                "cost": response["cost"]
            }

        except Exception as e:
            logger.error(f"Strategy optimization failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "applied_changes": [],
                "rejected_changes": []
            }

    def _build_optimization_prompt(
        self,
        trades: List[Dict],
        current_params: Dict,
        portfolio_stats: Dict,
        market_context: Dict
    ) -> str:
        """Build optimization prompt"""

        # Calculate trade metrics
        if trades:
            winning = [t for t in trades if t.get('pnl', 0) > 0]
            losing = [t for t in trades if t.get('pnl', 0) < 0]
            win_rate = len(winning) / len(trades) * 100
            avg_win = sum(t.get('pnl_pct', 0) for t in winning) / max(1, len(winning))
            avg_loss = sum(t.get('pnl_pct', 0) for t in losing) / max(1, len(losing))
        else:
            win_rate = avg_win = avg_loss = 0

        prompt = f"""
OPTIMIZA LA ESTRATEGIA DEL BOT BASANDOTE EN EL RENDIMIENTO RECIENTE.

== RENDIMIENTO ACTUAL ==
- Total Trades: {len(trades)}
- Win Rate: {win_rate:.1f}%
- Ganancia Promedio: {avg_win:+.2f}%
- Perdida Promedio: {avg_loss:+.2f}%
- ROI Total: {portfolio_stats.get('roi', 0):+.2f}%
- Max Drawdown: {portfolio_stats.get('max_drawdown', 0):.2f}%
- Sharpe Ratio: {portfolio_stats.get('sharpe_ratio', 0):.2f}
- Profit Factor: {portfolio_stats.get('profit_factor', 1.0):.2f}

== PARAMETROS ACTUALES ==
{json.dumps(current_params, indent=2)}

== CONTEXTO DE MERCADO ==
{json.dumps(market_context, indent=2)}

== ULTIMOS TRADES ==
{json.dumps(trades[-10:], indent=2) if trades else "[]"}

== PARAMETROS MODIFICABLES ==
{json.dumps(self.MODIFIABLE_PARAMS, indent=2)}

== INSTRUCCIONES ==
Analiza el rendimiento y decide si modificar parametros.
Responde en JSON con esta estructura EXACTA:

{{
    "analysis": {{
        "performance_summary": "Resumen del rendimiento en 2-3 oraciones",
        "main_issues": ["Lista de problemas identificados"],
        "opportunities": ["Lista de oportunidades de mejora"]
    }},

    "strategy_direction": "AGGRESSIVE/MAINTAIN/CONSERVATIVE/DEFENSIVE",

    "parameter_changes": [
        {{
            "parameter": "NOMBRE_DEL_PARAMETRO",
            "old_value": valor_actual,
            "new_value": valor_nuevo,
            "reason": "Razon del cambio",
            "expected_impact": "Impacto esperado",
            "confidence": 85,
            "priority": "HIGH/MEDIUM/LOW"
        }}
    ],

    "no_change_reasons": ["Razones para no cambiar ciertos parametros"],

    "confidence": 75,

    "next_review_hours": 2,

    "warnings": ["Advertencias importantes"],

    "rationale": "Explicacion general de la estrategia en 3-4 oraciones"
}}

REGLAS:
1. Solo cambia parametros si tienes confianza >= 70%
2. No hagas mas de 5 cambios a la vez
3. Cambios pequenos (±10-20%) son preferibles
4. Si el bot esta perdiendo, se MAS conservador
5. Respeta los rangos min/max de cada parametro
"""
        return prompt

    def _validate_change(self, param: str, value: Any) -> bool:
        """Validate parameter change against allowed ranges"""
        if param not in self.MODIFIABLE_PARAMS:
            logger.warning(f"Parameter {param} not in modifiable list")
            return False

        constraints = self.MODIFIABLE_PARAMS[param]
        min_val = constraints["min"]
        max_val = constraints["max"]

        try:
            numeric_value = float(value)
            if min_val <= numeric_value <= max_val:
                return True
            else:
                logger.warning(f"Value {value} for {param} outside range [{min_val}, {max_val}]")
                return False
        except (ValueError, TypeError):
            logger.warning(f"Invalid value type for {param}: {value}")
            return False

    def _record_change(
        self,
        param: str,
        old_value: Any,
        new_value: Any,
        reason: str
    ):
        """Record parameter change in history"""
        self.change_history.append({
            "timestamp": datetime.now().isoformat(),
            "parameter": param,
            "old_value": old_value,
            "new_value": new_value,
            "reason": reason
        })

    async def get_quick_adjustment(
        self,
        situation: str,
        current_params: Dict
    ) -> Dict[str, Any]:
        """
        Get quick parameter adjustment for specific situation

        Args:
            situation: Description of the situation
            current_params: Current parameters

        Returns:
            Quick adjustment recommendation
        """
        prompt = f"""
SITUACION: {situation}

PARAMETROS ACTUALES:
{json.dumps(current_params, indent=2)}

Responde en JSON:
{{
    "adjustment_needed": true/false,
    "changes": [
        {{
            "parameter": "PARAM",
            "new_value": valor,
            "reason": "razon"
        }}
    ],
    "urgency": "LOW/MEDIUM/HIGH/IMMEDIATE"
}}
"""
        try:
            response = await self.gpt.analyze(
                system_prompt="Eres un asesor de trading rapido. Responde en español.",
                user_prompt=prompt,
                temperature=0.3,
                max_tokens=2000,  # Must be enough for reasoning + response
                json_response=True
            )
            return response["data"]
        except Exception as e:
            logger.error(f"Quick adjustment failed: {e}")
            return {"adjustment_needed": False, "changes": []}

    async def react_to_loss(
        self,
        trade: Dict,
        consecutive_losses: int,
        current_params: Dict
    ) -> Dict[str, Any]:
        """
        React to a losing trade with parameter adjustments

        Args:
            trade: The losing trade
            consecutive_losses: Number of consecutive losses
            current_params: Current parameters

        Returns:
            Reactive adjustments
        """
        prompt = f"""
ALERTA: Trade perdedor detectado.

TRADE:
{json.dumps(trade, indent=2)}

PERDIDAS CONSECUTIVAS: {consecutive_losses}

PARAMETROS ACTUALES:
{json.dumps(current_params, indent=2)}

Si hay {consecutive_losses}+ perdidas consecutivas, considera:
- Reducir position size
- Aumentar thresholds
- Reducir leverage
- Ser mas selectivo

Responde en JSON:
{{
    "severity": "LOW/MEDIUM/HIGH/CRITICAL",
    "should_adjust": true/false,
    "changes": [
        {{
            "parameter": "PARAM",
            "new_value": valor,
            "reason": "razon"
        }}
    ],
    "should_pause": false,
    "pause_minutes": 0,
    "message": "Mensaje explicativo"
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
                **response["data"]
            }
        except Exception as e:
            logger.error(f"Loss reaction failed: {e}")
            # Default conservative response
            if consecutive_losses >= 3:
                return {
                    "success": False,
                    "severity": "HIGH",
                    "should_adjust": True,
                    "changes": [{
                        "parameter": "BASE_POSITION_SIZE_PCT",
                        "new_value": max(1.0, current_params.get("BASE_POSITION_SIZE_PCT", 10) * 0.7),
                        "reason": "Reduccion automatica por perdidas consecutivas"
                    }],
                    "should_pause": consecutive_losses >= 5,
                    "pause_minutes": 30 if consecutive_losses >= 5 else 0
                }
            return {"success": False, "should_adjust": False, "changes": []}

    async def react_to_win_streak(
        self,
        consecutive_wins: int,
        current_params: Dict
    ) -> Dict[str, Any]:
        """
        React to a winning streak with parameter adjustments

        Args:
            consecutive_wins: Number of consecutive wins
            current_params: Current parameters

        Returns:
            Aggressive adjustments if warranted
        """
        if consecutive_wins < 3:
            return {"should_adjust": False, "changes": []}

        prompt = f"""
BUENAS NOTICIAS: {consecutive_wins} trades ganadores consecutivos!

PARAMETROS ACTUALES:
{json.dumps(current_params, indent=2)}

Considera si es seguro ser mas agresivo:
- Aumentar position size (con cuidado)
- Reducir thresholds ligeramente
- Aumentar leverage moderadamente

ADVERTENCIA: No te excedas. Las rachas terminan.

Responde en JSON:
{{
    "should_adjust": true/false,
    "changes": [
        {{
            "parameter": "PARAM",
            "new_value": valor,
            "reason": "razon"
        }}
    ],
    "confidence": 0-100,
    "warning": "Advertencia si aplica"
}}
"""
        try:
            response = await self.gpt.analyze(
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=prompt,
                temperature=0.4,
                json_response=True
            )
            return response["data"]
        except Exception as e:
            logger.error(f"Win streak reaction failed: {e}")
            return {"should_adjust": False, "changes": []}

    def get_change_history(self, limit: int = 20) -> List[Dict]:
        """Get recent change history"""
        return self.change_history[-limit:]

    def get_pending_changes(self) -> List[Dict]:
        """Get pending changes awaiting approval"""
        return self.pending_changes

    def apply_pending_changes(self) -> List[Dict]:
        """Apply all pending changes"""
        applied = []
        for change in self.pending_changes:
            if self.param_update_callback:
                success = self.param_update_callback(
                    change["parameter"],
                    change["new_value"]
                )
                if success:
                    applied.append(change)
                    self._record_change(
                        change["parameter"],
                        change.get("old_value"),
                        change["new_value"],
                        change.get("reason", "Applied from pending")
                    )
        self.pending_changes = [c for c in self.pending_changes if c not in applied]
        return applied

    def clear_pending_changes(self):
        """Clear all pending changes"""
        self.pending_changes = []

    def should_optimize(self, min_interval_hours: float = 2.0) -> bool:
        """Check if enough time has passed since last optimization"""
        if not self.last_optimization:
            return True
        elapsed = datetime.now() - self.last_optimization
        return elapsed >= timedelta(hours=min_interval_hours)
