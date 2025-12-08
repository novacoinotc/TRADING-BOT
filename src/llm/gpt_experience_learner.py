"""
GPT Experience Learner - Persistent Learning from Trading Experience

This module enables GPT to "learn" from trade history by:
1. Tracking trade outcomes with full context
2. Identifying patterns in wins/losses
3. Maintaining a persistent "wisdom file" with learned lessons
4. Using experience to inform future decisions

Key insight: GPT doesn't have persistent memory, but we can:
- Store insights in a JSON wisdom file
- Include relevant wisdom in prompts
- Update wisdom after analyzing trades
"""

import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
from src.llm.gpt_client import GPTClient

logger = logging.getLogger(__name__)


class GPTExperienceLearner:
    """
    GPT-powered experience learning system.

    Maintains a "wisdom file" that stores:
    - Patterns that lead to winning trades
    - Patterns that lead to losing trades
    - Parameter insights
    - Market condition insights
    - Anti-patterns to avoid
    """

    WISDOM_FILE = "data/gpt_wisdom.json"
    TRADE_MEMORY_FILE = "data/gpt_trade_memory.json"

    LEARNER_SYSTEM_PROMPT = """Eres un analista de trading experto que aprende de la experiencia.
Tu trabajo es analizar trades cerrados y extraer LECCIONES VALIOSAS que se puedan aplicar en el futuro.

IMPORTANTE:
- Busca PATRONES, no eventos aislados
- Identifica QUÉ funcionó y POR QUÉ
- Identifica QUÉ falló y POR QUÉ
- Sé específico y actionable
- Las lecciones deben ser prácticas

Responde siempre en español y en formato JSON estructurado."""

    PATTERN_ANALYSIS_PROMPT = """Eres un detector de patrones de trading.
Tu trabajo es identificar patrones en secuencias de trades.

Busca:
1. Combinaciones de indicadores que siempre ganan/pierden
2. Condiciones de mercado problemáticas
3. Horarios o días que afectan resultados
4. Patrones de señales que engañan
5. Anti-patrones a evitar

Responde en español con JSON estructurado."""

    def __init__(self, gpt_client: GPTClient, config: Optional[Any] = None):
        """
        Initialize GPT Experience Learner

        Args:
            gpt_client: Initialized GPT client
            config: Configuration object
        """
        self.gpt = gpt_client
        self.config = config

        # Load or initialize wisdom
        self.wisdom = self._load_wisdom()
        self.trade_memory = self._load_trade_memory()

        # Stats
        self.lessons_learned = len(self.wisdom.get("lessons", []))
        self.patterns_identified = len(self.wisdom.get("patterns", []))
        self.last_learning_session: Optional[datetime] = None

        logger.info(f"GPT Experience Learner initialized with {self.lessons_learned} lessons")

    def _load_wisdom(self) -> Dict:
        """Load wisdom file or create default"""
        try:
            if os.path.exists(self.WISDOM_FILE):
                with open(self.WISDOM_FILE, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load wisdom file: {e}")

        # Default wisdom structure
        return {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "total_trades_analyzed": 0,
            "lessons": [],
            "patterns": {
                "winning_patterns": [],
                "losing_patterns": [],
                "anti_patterns": []
            },
            "parameter_insights": [],
            "market_insights": [],
            "pair_specific": {},
            "time_based": {
                "best_hours": [],
                "worst_hours": [],
                "best_days": [],
                "worst_days": []
            },
            "golden_rules": [],
            "mistakes_to_avoid": []
        }

    def _save_wisdom(self):
        """Save wisdom to file"""
        try:
            os.makedirs(os.path.dirname(self.WISDOM_FILE), exist_ok=True)
            self.wisdom["updated_at"] = datetime.now().isoformat()
            with open(self.WISDOM_FILE, 'w') as f:
                json.dump(self.wisdom, f, indent=2, ensure_ascii=False)
            logger.info("Wisdom file saved")
        except Exception as e:
            logger.error(f"Could not save wisdom file: {e}")

    def _load_trade_memory(self) -> List[Dict]:
        """Load trade memory for pattern analysis"""
        try:
            if os.path.exists(self.TRADE_MEMORY_FILE):
                with open(self.TRADE_MEMORY_FILE, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load trade memory: {e}")
        return []

    def _save_trade_memory(self):
        """Save trade memory"""
        try:
            os.makedirs(os.path.dirname(self.TRADE_MEMORY_FILE), exist_ok=True)
            # Keep only last 500 trades
            self.trade_memory = self.trade_memory[-500:]
            with open(self.TRADE_MEMORY_FILE, 'w') as f:
                json.dump(self.trade_memory, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save trade memory: {e}")

    async def learn_from_trade(
        self,
        trade: Dict,
        market_context: Dict,
        signal_data: Dict
    ) -> Dict[str, Any]:
        """
        Learn from a single closed trade

        Args:
            trade: Closed trade data
            market_context: Market conditions when trade was opened
            signal_data: Original signal that triggered the trade

        Returns:
            Learning result with any new insights
        """
        # Store in memory
        memory_entry = {
            "timestamp": datetime.now().isoformat(),
            "trade": trade,
            "market_context": market_context,
            "signal_data": signal_data,
            "outcome": "WIN" if trade.get("pnl", 0) > 0 else "LOSS"
        }
        self.trade_memory.append(memory_entry)
        self._save_trade_memory()

        self.wisdom["total_trades_analyzed"] += 1

        # Don't call GPT for every trade - batch learning
        # But return quick insights
        return {
            "stored": True,
            "trades_until_analysis": max(0, 10 - (len(self.trade_memory) % 10))
        }

    async def run_learning_session(
        self,
        trades: List[Dict],
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Run a full learning session analyzing multiple trades

        Args:
            trades: List of closed trades to analyze
            force: Force analysis even if not enough trades

        Returns:
            Learning session results
        """
        if len(trades) < 10 and not force:
            return {
                "success": False,
                "reason": f"Need at least 10 trades, have {len(trades)}"
            }

        prompt = self._build_learning_prompt(trades)

        try:
            response = await self.gpt.analyze(
                system_prompt=self.LEARNER_SYSTEM_PROMPT,
                user_prompt=prompt,
                temperature=0.4,  # Moderate temperature for learning insights
                max_tokens=2000,
                json_response=True
            )

            learning = response["data"]

            # Process and store lessons
            new_lessons = learning.get("lessons", [])
            new_patterns = learning.get("patterns", {})
            golden_rules = learning.get("golden_rules", [])
            mistakes = learning.get("mistakes_to_avoid", [])

            # Add to wisdom (avoiding duplicates)
            for lesson in new_lessons:
                if lesson not in self.wisdom["lessons"]:
                    self.wisdom["lessons"].append(lesson)

            for pattern in new_patterns.get("winning", []):
                if pattern not in self.wisdom["patterns"]["winning_patterns"]:
                    self.wisdom["patterns"]["winning_patterns"].append(pattern)

            for pattern in new_patterns.get("losing", []):
                if pattern not in self.wisdom["patterns"]["losing_patterns"]:
                    self.wisdom["patterns"]["losing_patterns"].append(pattern)

            for rule in golden_rules:
                if rule not in self.wisdom["golden_rules"]:
                    self.wisdom["golden_rules"].append(rule)

            for mistake in mistakes:
                if mistake not in self.wisdom["mistakes_to_avoid"]:
                    self.wisdom["mistakes_to_avoid"].append(mistake)

            # Save updated wisdom
            self._save_wisdom()

            self.last_learning_session = datetime.now()
            self.lessons_learned = len(self.wisdom["lessons"])
            self.patterns_identified = (
                len(self.wisdom["patterns"]["winning_patterns"]) +
                len(self.wisdom["patterns"]["losing_patterns"])
            )

            logger.info(
                f"Learning session complete: {len(new_lessons)} new lessons, "
                f"{len(golden_rules)} golden rules"
            )

            return {
                "success": True,
                "new_lessons": len(new_lessons),
                "new_patterns": len(new_patterns.get("winning", [])) + len(new_patterns.get("losing", [])),
                "golden_rules": golden_rules,
                "mistakes": mistakes,
                "summary": learning.get("summary", ""),
                "cost": response["cost"]
            }

        except Exception as e:
            logger.error(f"Learning session failed: {e}")
            return {"success": False, "error": str(e)}

    def _build_learning_prompt(self, trades: List[Dict]) -> str:
        """Build learning prompt from trades"""

        # Calculate stats
        wins = [t for t in trades if t.get("pnl", 0) > 0]
        losses = [t for t in trades if t.get("pnl", 0) <= 0]

        total_pnl = sum(t.get("pnl", 0) for t in trades)
        win_rate = len(wins) / len(trades) * 100 if trades else 0

        # Format trades
        trades_summary = []
        for t in trades[-20:]:  # Last 20 trades
            trades_summary.append({
                "pair": t.get("pair", "N/A"),
                "side": t.get("side", "N/A"),
                "pnl": t.get("pnl", 0),
                "pnl_pct": t.get("pnl_pct", 0),
                "duration_hours": t.get("duration_hours", 0),
                "leverage": t.get("leverage", 1),
                "rsi_at_entry": t.get("rsi_at_entry", 50),
                "regime": t.get("regime", "UNKNOWN"),
                "news_triggered": t.get("news_triggered", False),
                "ml_confidence": t.get("ml_confidence", 0),
                "entry_time": t.get("entry_time", ""),
                "exit_reason": t.get("exit_reason", "")
            })

        prompt = f"""
ANALIZA ESTOS {len(trades)} TRADES Y EXTRAE LECCIONES:

== ESTADÍSTICAS GENERALES ==
- Win Rate: {win_rate:.1f}%
- Total P&L: ${total_pnl:,.2f}
- Trades ganadores: {len(wins)}
- Trades perdedores: {len(losses)}

== TRADES RECIENTES ==
{json.dumps(trades_summary, indent=2)}

== SABIDURÍA ACTUAL (lo que ya sabemos) ==
Lecciones previas: {len(self.wisdom.get('lessons', []))}
Patrones ganadores conocidos: {len(self.wisdom['patterns']['winning_patterns'])}
Patrones perdedores conocidos: {len(self.wisdom['patterns']['losing_patterns'])}

== INSTRUCCIONES ==
Analiza los trades y responde en JSON:

{{
    "summary": "Resumen de 2-3 oraciones sobre lo observado",

    "lessons": [
        "Lección específica 1 (qué aprendimos)",
        "Lección específica 2"
    ],

    "patterns": {{
        "winning": [
            {{
                "pattern": "Descripción del patrón ganador",
                "conditions": ["RSI < 35", "Régimen BULLISH", etc.],
                "confidence": 80,
                "occurrences": 5
            }}
        ],
        "losing": [
            {{
                "pattern": "Descripción del patrón perdedor",
                "conditions": ["Condición 1", "Condición 2"],
                "confidence": 75,
                "occurrences": 4
            }}
        ]
    }},

    "golden_rules": [
        "Regla de oro 1: Siempre/Nunca hacer X cuando Y"
    ],

    "mistakes_to_avoid": [
        "Error común 1: Descripción y cómo evitarlo"
    ],

    "parameter_suggestions": [
        {{
            "parameter": "RSI_OVERSOLD",
            "current": 30,
            "suggested": 35,
            "reason": "Los trades con RSI 30-35 tienen mejor win rate"
        }}
    ],

    "pair_insights": [
        {{
            "pair": "BTC/USDT",
            "insight": "Observación específica sobre este par"
        }}
    ]
}}

IMPORTANTE:
- Solo incluye patrones con al menos 3 ocurrencias
- Las lecciones deben ser ACCIONABLES
- Las reglas de oro deben ser CLARAS y ESPECÍFICAS
"""
        return prompt

    async def analyze_losing_streak(
        self,
        recent_losses: List[Dict],
        current_params: Dict
    ) -> Dict[str, Any]:
        """
        Deep analysis of a losing streak

        Args:
            recent_losses: List of recent losing trades
            current_params: Current trading parameters

        Returns:
            Analysis with recommendations
        """
        prompt = f"""
ANÁLISIS DE EMERGENCIA - RACHA PERDEDORA

Se han registrado {len(recent_losses)} pérdidas consecutivas.
Necesitamos entender QUÉ ESTÁ FALLANDO.

== TRADES PERDEDORES ==
{json.dumps([{
    'pair': t.get('pair'),
    'side': t.get('side'),
    'pnl': t.get('pnl'),
    'pnl_pct': t.get('pnl_pct'),
    'rsi': t.get('rsi_at_entry'),
    'regime': t.get('regime'),
    'exit_reason': t.get('exit_reason')
} for t in recent_losses], indent=2)}

== PARÁMETROS ACTUALES ==
{json.dumps(current_params, indent=2)}

== SABIDURÍA CONOCIDA ==
Patrones perdedores conocidos:
{json.dumps(self.wisdom['patterns']['losing_patterns'][:5], indent=2)}

Errores a evitar:
{json.dumps(self.wisdom['mistakes_to_avoid'][:5], indent=2)}

== RESPONDE EN JSON ==
{{
    "diagnosis": "Qué está causando las pérdidas",
    "root_cause": "La causa raíz más probable",
    "immediate_actions": [
        "Acción inmediata 1",
        "Acción inmediata 2"
    ],
    "parameter_changes": [
        {{
            "parameter": "NOMBRE",
            "current": valor_actual,
            "recommended": valor_nuevo,
            "reason": "Por qué este cambio"
        }}
    ],
    "should_pause": true,
    "pause_duration_minutes": 30,
    "confidence": 75
}}
"""

        try:
            response = await self.gpt.analyze(
                system_prompt="Eres un consultor de trading de emergencia. Diagnostica problemas y da soluciones.",
                user_prompt=prompt,
                temperature=0.4,
                max_tokens=1000,
                json_response=True
            )

            return {
                "success": True,
                "analysis": response["data"],
                "cost": response["cost"]
            }

        except Exception as e:
            logger.error(f"Losing streak analysis failed: {e}")
            return {"success": False, "error": str(e)}

    def get_relevant_wisdom(
        self,
        pair: str,
        market_conditions: Dict,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Get relevant wisdom for current trading context

        Args:
            pair: Trading pair
            market_conditions: Current market conditions
            limit: Max items per category

        Returns:
            Relevant wisdom for the context
        """
        relevant = {
            "golden_rules": self.wisdom.get("golden_rules", [])[:limit],
            "mistakes_to_avoid": self.wisdom.get("mistakes_to_avoid", [])[:limit],
            "winning_patterns": [],
            "losing_patterns": [],
            "pair_specific": None
        }

        # Get pair-specific wisdom
        if pair in self.wisdom.get("pair_specific", {}):
            relevant["pair_specific"] = self.wisdom["pair_specific"][pair]

        # Filter patterns by current conditions
        regime = market_conditions.get("regime", "")
        rsi = market_conditions.get("rsi", 50)

        for pattern in self.wisdom["patterns"].get("winning_patterns", [])[:limit]:
            # Include patterns that might apply
            conditions = pattern.get("conditions", [])
            if any(regime.lower() in c.lower() for c in conditions):
                relevant["winning_patterns"].append(pattern)
            elif len(relevant["winning_patterns"]) < 3:
                relevant["winning_patterns"].append(pattern)

        for pattern in self.wisdom["patterns"].get("losing_patterns", [])[:limit]:
            conditions = pattern.get("conditions", [])
            if any(regime.lower() in c.lower() for c in conditions):
                relevant["losing_patterns"].append(pattern)
            elif len(relevant["losing_patterns"]) < 3:
                relevant["losing_patterns"].append(pattern)

        return relevant

    def format_wisdom_for_prompt(self, wisdom: Dict) -> str:
        """Format wisdom dictionary for inclusion in prompts"""
        formatted = "== SABIDURÍA APRENDIDA ==\n"

        if wisdom.get("golden_rules"):
            formatted += "\n🏆 REGLAS DE ORO:\n"
            for rule in wisdom["golden_rules"][:5]:
                formatted += f"  • {rule}\n"

        if wisdom.get("mistakes_to_avoid"):
            formatted += "\n⚠️ ERRORES A EVITAR:\n"
            for mistake in wisdom["mistakes_to_avoid"][:5]:
                formatted += f"  • {mistake}\n"

        if wisdom.get("winning_patterns"):
            formatted += "\n✅ PATRONES GANADORES:\n"
            for pattern in wisdom["winning_patterns"][:3]:
                if isinstance(pattern, dict):
                    formatted += f"  • {pattern.get('pattern', str(pattern))}\n"
                else:
                    formatted += f"  • {pattern}\n"

        if wisdom.get("losing_patterns"):
            formatted += "\n❌ PATRONES PERDEDORES:\n"
            for pattern in wisdom["losing_patterns"][:3]:
                if isinstance(pattern, dict):
                    formatted += f"  • {pattern.get('pattern', str(pattern))}\n"
                else:
                    formatted += f"  • {pattern}\n"

        if wisdom.get("pair_specific"):
            formatted += f"\n📊 ESPECÍFICO DEL PAR:\n  • {wisdom['pair_specific']}\n"

        return formatted

    def get_stats(self) -> Dict[str, Any]:
        """Get learner statistics"""
        return {
            "total_trades_analyzed": self.wisdom.get("total_trades_analyzed", 0),
            "lessons_learned": self.lessons_learned,
            "patterns_identified": self.patterns_identified,
            "winning_patterns": len(self.wisdom["patterns"]["winning_patterns"]),
            "losing_patterns": len(self.wisdom["patterns"]["losing_patterns"]),
            "golden_rules": len(self.wisdom.get("golden_rules", [])),
            "mistakes_cataloged": len(self.wisdom.get("mistakes_to_avoid", [])),
            "trades_in_memory": len(self.trade_memory),
            "last_learning_session": (
                self.last_learning_session.isoformat()
                if self.last_learning_session else None
            ),
            "wisdom_file_exists": os.path.exists(self.WISDOM_FILE)
        }

    def clear_wisdom(self):
        """Clear all learned wisdom (use with caution)"""
        self.wisdom = self._load_wisdom.__func__(self)  # Reset to default
        self._save_wisdom()
        self.trade_memory = []
        self._save_trade_memory()
        logger.warning("All wisdom cleared!")
