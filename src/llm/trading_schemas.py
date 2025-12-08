"""
Trading JSON Schemas - Strict validation for GPT responses

Based on GPT-5 Trading Integration Guide:
- Ensures consistent, parseable responses
- Validates all required fields
- Prevents malformed trading decisions
"""

# JSON Schema for Trading Decisions (used with response_format)
TRADING_DECISION_SCHEMA = {
    "name": "trading_decision",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "approved": {
                "type": "boolean",
                "description": "Whether to approve this trade"
            },
            "confidence": {
                "type": "integer",
                "description": "Confidence level 0-100"
            },
            "direction": {
                "type": "string",
                "enum": ["LONG", "SHORT", "HOLD"],
                "description": "Trade direction"
            },
            "reasoning": {
                "type": "string",
                "description": "Brief explanation of the decision"
            },
            "rejection_reason": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "description": "Reason if trade is rejected"
            },
            "is_risky_trade": {
                "type": "boolean",
                "description": "Whether this is a risky/learning trade"
            },
            "learning_opportunity": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "description": "What to learn from this trade"
            },
            "position_size": {
                "type": "object",
                "properties": {
                    "recommendation": {
                        "type": "string",
                        "enum": ["FULL", "THREE_QUARTER", "HALF", "QUARTER", "MINI", "SKIP"]
                    },
                    "percentage": {
                        "type": "integer",
                        "description": "Position size as percentage 0-100"
                    },
                    "reason": {
                        "type": "string"
                    }
                },
                "required": ["recommendation", "percentage", "reason"],
                "additionalProperties": False
            },
            "leverage": {
                "type": "object",
                "properties": {
                    "recommended": {
                        "type": "integer",
                        "description": "Recommended leverage 1-10"
                    },
                    "max_safe": {
                        "type": "integer",
                        "description": "Maximum safe leverage"
                    },
                    "reason": {
                        "type": "string"
                    }
                },
                "required": ["recommended", "max_safe", "reason"],
                "additionalProperties": False
            },
            "risk_management": {
                "type": "object",
                "properties": {
                    "stop_loss_pct": {
                        "type": "number",
                        "description": "Stop loss percentage (0.5-5.0)"
                    },
                    "take_profit_pct": {
                        "type": "number",
                        "description": "Take profit percentage (0.2-10.0)"
                    },
                    "trailing_stop": {
                        "type": "boolean"
                    },
                    "trailing_distance_pct": {
                        "type": "number"
                    },
                    "risk_reward_ratio": {
                        "type": "number"
                    },
                    "liquidation_buffer_pct": {
                        "type": "number"
                    },
                    "tp_reasoning": {
                        "type": "string",
                        "description": "Justification for TP level"
                    }
                },
                "required": ["stop_loss_pct", "take_profit_pct", "trailing_stop", "trailing_distance_pct", "risk_reward_ratio", "liquidation_buffer_pct", "tp_reasoning"],
                "additionalProperties": False
            },
            "futures_considerations": {
                "type": "object",
                "properties": {
                    "funding_rate_impact": {
                        "type": "string",
                        "enum": ["FAVORABLE", "NEUTRAL", "ADVERSE"]
                    },
                    "hold_duration": {
                        "type": "string",
                        "enum": ["MINUTES", "HOURS", "AVOID_OVERNIGHT"]
                    },
                    "liquidation_risk": {
                        "type": "string",
                        "enum": ["LOW", "MEDIUM", "HIGH"]
                    }
                },
                "required": ["funding_rate_impact", "hold_duration", "liquidation_risk"],
                "additionalProperties": False
            },
            "timing": {
                "type": "object",
                "properties": {
                    "urgency": {
                        "type": "string",
                        "enum": ["IMMEDIATE", "WAIT", "SKIP"]
                    },
                    "wait_for": {
                        "anyOf": [{"type": "string"}, {"type": "null"}]
                    }
                },
                "required": ["urgency", "wait_for"],
                "additionalProperties": False
            },
            "overrides": {
                "type": "object",
                "properties": {
                    "ml": {
                        "type": "boolean"
                    },
                    "rl": {
                        "type": "boolean"
                    },
                    "override_reason": {
                        "anyOf": [{"type": "string"}, {"type": "null"}]
                    }
                },
                "required": ["ml", "rl", "override_reason"],
                "additionalProperties": False
            },
            "warnings": {
                "type": "array",
                "items": {
                    "type": "string"
                }
            },
            "alternative_action": {
                "anyOf": [{"type": "string"}, {"type": "null"}]
            }
        },
        "required": [
            "approved",
            "confidence",
            "direction",
            "reasoning",
            "rejection_reason",
            "is_risky_trade",
            "learning_opportunity",
            "position_size",
            "leverage",
            "risk_management",
            "futures_considerations",
            "timing",
            "overrides",
            "warnings",
            "alternative_action"
        ],
        "additionalProperties": False
    }
}

# Simplified schema for trade management decisions
TRADE_MANAGEMENT_SCHEMA = {
    "name": "trade_management",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["HOLD", "CLOSE", "ADJUST_SL", "ADJUST_TP", "PARTIAL_CLOSE"]
            },
            "reason": {
                "type": "string"
            },
            "new_stop_loss": {
                "anyOf": [{"type": "number"}, {"type": "null"}]
            },
            "new_take_profit": {
                "anyOf": [{"type": "number"}, {"type": "null"}]
            },
            "close_percentage": {
                "anyOf": [{"type": "integer"}, {"type": "null"}],
                "description": "Percentage to close if PARTIAL_CLOSE"
            },
            "urgency": {
                "type": "string",
                "enum": ["IMMEDIATE", "SOON", "LOW"]
            },
            "confidence": {
                "type": "integer"
            }
        },
        "required": ["action", "reason", "new_stop_loss", "new_take_profit", "close_percentage", "urgency", "confidence"],
        "additionalProperties": False
    }
}


def validate_trading_decision(decision: dict) -> tuple[bool, list[str]]:
    """
    Validate a trading decision mathematically.

    Based on GPT-5 Trading Guide Section 13: Mathematical Validations

    Returns:
        (is_valid, list of validation errors)
    """
    errors = []

    # Skip validation if not approved
    if not decision.get("approved", False):
        return True, []

    risk_mgmt = decision.get("risk_management", {})
    direction = decision.get("direction", "HOLD")

    # === Validation 1: TP must be > commissions (minimum 0.20% for taker) ===
    tp_pct = risk_mgmt.get("take_profit_pct", 0)
    MIN_TP_FOR_PROFIT = 0.20  # Minimum to cover taker fees (0.09% x2) + slippage (~0.02%)
    if tp_pct < MIN_TP_FOR_PROFIT:
        errors.append(f"TP {tp_pct}% is below minimum profitable {MIN_TP_FOR_PROFIT}% (doesn't cover fees)")

    # === Validation 2: SL must be reasonable (0.3% - 5%) ===
    sl_pct = risk_mgmt.get("stop_loss_pct", 0)
    if sl_pct < 0.3:
        errors.append(f"SL {sl_pct}% is too tight (minimum 0.3%)")
    if sl_pct > 5.0:
        errors.append(f"SL {sl_pct}% is too wide (maximum 5%)")

    # === Validation 3: Risk/Reward check (flexible para scalping) ===
    confidence = decision.get("confidence", 0)
    # En scalping, a veces TP < SL está bien si la probabilidad de win es alta
    # Solo advertimos en casos extremos
    if tp_pct < sl_pct * 0.5 and confidence < 70:
        errors.append(f"TP {tp_pct}% muy bajo vs SL {sl_pct}% (R:R < 0.5:1)")

    # === Validation 4: Leverage must be within bounds ===
    leverage = decision.get("leverage", {}).get("recommended", 1)
    if leverage < 1:
        errors.append(f"Leverage {leverage}x is invalid (minimum 1x)")
    if leverage > 10:
        errors.append(f"Leverage {leverage}x is too high (maximum 10x)")

    # === Validation 5: Position size vs confidence (permisivo para aprendizaje) ===
    pos_size = decision.get("position_size", {})
    pos_pct = pos_size.get("percentage", 0)

    # Límites más flexibles para permitir experimentación
    # Solo advertimos en casos extremos, no rechazamos
    if confidence < 30 and pos_pct > 25:
        errors.append(f"Position {pos_pct}% too large for confidence {confidence}% (max 25%)")
    elif confidence < 50 and pos_pct > 50:
        errors.append(f"Position {pos_pct}% too large for confidence {confidence}% (max 50%)")
    # Para confianza >= 50%, permitimos posiciones más grandes

    # === Validation 6: Liquidation buffer (advertencia, no bloqueo) ===
    liq_buffer = risk_mgmt.get("liquidation_buffer_pct", 0)
    # Solo advertimos si es muy bajo, pero permitimos el trade
    # El trader autónomo puede decidir con menos buffer si tiene razones
    if liq_buffer < 1.0 and liq_buffer > 0:
        errors.append(f"Liquidation buffer {liq_buffer}% is very low (recommended 2%+)")

    # === Validation 7: Direction must be valid ===
    if direction not in ["LONG", "SHORT", "HOLD"]:
        errors.append(f"Invalid direction: {direction}")

    return len(errors) == 0, errors


def fix_trading_decision(decision: dict) -> dict:
    """
    Auto-fix common validation issues in trading decisions.

    Returns corrected decision dict.
    """
    fixed = decision.copy()

    risk_mgmt = fixed.get("risk_management", {}).copy()
    leverage_data = fixed.get("leverage", {}).copy()
    pos_size = fixed.get("position_size", {}).copy()

    # Fix TP if too low
    if risk_mgmt.get("take_profit_pct", 0) < 0.20:
        risk_mgmt["take_profit_pct"] = 0.20
        risk_mgmt["tp_reasoning"] = f"[AUTO-FIXED] Original TP too low, set to minimum 0.20%"

    # Fix SL if out of bounds
    sl = risk_mgmt.get("stop_loss_pct", 1.0)
    if sl < 0.3:
        risk_mgmt["stop_loss_pct"] = 0.3
    elif sl > 5.0:
        risk_mgmt["stop_loss_pct"] = 5.0

    # Fix leverage if out of bounds
    lev = leverage_data.get("recommended", 3)
    if lev < 1:
        leverage_data["recommended"] = 1
    elif lev > 10:
        leverage_data["recommended"] = 10

    # Fix liquidation buffer
    if risk_mgmt.get("liquidation_buffer_pct", 0) < 2.0:
        risk_mgmt["liquidation_buffer_pct"] = 2.0

    # Ensure trailing distance exists
    if "trailing_distance_pct" not in risk_mgmt:
        risk_mgmt["trailing_distance_pct"] = 0.5

    fixed["risk_management"] = risk_mgmt
    fixed["leverage"] = leverage_data
    fixed["position_size"] = pos_size

    return fixed
