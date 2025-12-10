"""
GPT Market Analyst - Intelligent Market Analysis and Signal Generation

This module uses GPT to analyze market conditions and generate trading signals
with deep reasoning that goes beyond traditional technical indicators.

Key capabilities:
- Holistic market analysis (technicals + sentiment + context)
- Pattern recognition that indicators miss
- News and macro interpretation
- Opportunity identification
- Contrarian signal detection
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from src.llm.gpt_client import GPTClient

logger = logging.getLogger(__name__)


class GPTMarketAnalyst:
    """
    GPT-powered market analyst that can:
    1. Analyze market conditions holistically
    2. Generate trading signals based on reasoning
    3. Validate/enhance signals from other systems
    4. Identify opportunities the system might miss
    """

    ANALYST_SYSTEM_PROMPT = """Eres un trader profesional de criptomonedas con 15+ años de experiencia.
Tu trabajo es analizar el mercado y encontrar las MEJORES oportunidades de trading.

Tienes acceso a:
- Indicadores técnicos (RSI, MACD, EMA, Bollinger Bands, ATR, ADX)
- Datos de sentiment (Fear & Greed Index, noticias, social buzz)
- Order book data (presión de compra/venta, imbalances)
- Market regime (tendencia, volatilidad)
- Multi-timeframe data (1h, 4h, 1d)

Tu análisis debe ser:
1. CONTRARIAN cuando sea apropiado (comprar en miedo extremo, vender en euforia)
2. CONTEXTUAL (considera el momento del mercado, no solo indicadores)
3. PRECISO (da recomendaciones específicas con precios)
4. HONESTO (di cuando NO hay oportunidades claras)

IMPORTANTE:
- Solo recomienda trades con alta probabilidad (>65% confianza)
- Prefiere calidad sobre cantidad
- Un "NO TRADE" es mejor que un mal trade

Responde siempre en español."""

    SIGNAL_SYSTEM_PROMPT = """Eres un sistema experto de generación de señales de trading.
Tu trabajo es analizar datos de mercado y decidir si hay una oportunidad de trade.

CRITERIOS PARA SEÑAL:
1. Confluencia de al menos 3 factores (técnicos + sentiment + orderbook)
2. Risk/Reward mínimo de 1:2
3. Confianza mínima del 65%
4. Contexto de mercado favorable

TIPOS DE SEÑALES:
- STRONG_BUY: Alta confianza, múltiples confirmaciones
- BUY: Buena oportunidad, riesgo controlado
- WEAK_BUY: Oportunidad menor, tamaño reducido
- HOLD: No hay oportunidad clara
- WEAK_SELL: Señal de venta menor
- SELL: Buena oportunidad de short
- STRONG_SELL: Alta confianza para short

Siempre responde en JSON estructurado. Siempre en español."""

    def __init__(self, gpt_client: GPTClient, config: Optional[Any] = None):
        """
        Initialize GPT Market Analyst

        Args:
            gpt_client: Initialized GPT client
            config: Configuration object
        """
        self.gpt = gpt_client
        self.config = config
        self.analysis_cache: Dict[str, Dict] = {}
        self.signals_generated: List[Dict] = []
        self.last_full_scan: Optional[datetime] = None
        self.opportunities_found = 0
        self.signals_approved = 0

    async def analyze_market(
        self,
        pair: str,
        indicators: Dict,
        sentiment_data: Optional[Dict] = None,
        orderbook_data: Optional[Dict] = None,
        regime_data: Optional[Dict] = None,
        mtf_data: Optional[Dict] = None,
        news_summary: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform deep market analysis for a trading pair

        Args:
            pair: Trading pair (e.g., 'BTC/USDT')
            indicators: Technical indicators
            sentiment_data: Sentiment analysis data
            orderbook_data: Order book analysis
            regime_data: Market regime data
            mtf_data: Multi-timeframe data
            news_summary: Recent news summary

        Returns:
            Comprehensive market analysis
        """
        prompt = self._build_analysis_prompt(
            pair, indicators, sentiment_data, orderbook_data,
            regime_data, mtf_data, news_summary
        )

        try:
            response = await self.gpt.analyze(
                system_prompt=self.ANALYST_SYSTEM_PROMPT,
                user_prompt=prompt,
                temperature=0.6,
                max_tokens=1500,
                json_response=True
            )

            analysis = response["data"]

            # Cache the analysis
            self.analysis_cache[pair] = {
                "analysis": analysis,
                "timestamp": datetime.now(),
                "cost": response["cost"]
            }

            logger.info(f"GPT Market Analysis for {pair}: {analysis.get('outlook', 'N/A')}")

            return {
                "success": True,
                "pair": pair,
                "analysis": analysis,
                "cost": response["cost"]
            }

        except Exception as e:
            logger.error(f"Market analysis failed for {pair}: {e}")
            return {
                "success": False,
                "pair": pair,
                "error": str(e)
            }

    def _build_analysis_prompt(
        self,
        pair: str,
        indicators: Dict,
        sentiment_data: Optional[Dict],
        orderbook_data: Optional[Dict],
        regime_data: Optional[Dict],
        mtf_data: Optional[Dict],
        news_summary: Optional[str]
    ) -> str:
        """Build comprehensive analysis prompt"""

        prompt = f"""
ANALIZA EL MERCADO DE {pair} Y ENCUENTRA OPORTUNIDADES.

== INDICADORES TÉCNICOS ==
- Precio actual: ${indicators.get('current_price', 0):,.2f}
- RSI (14): {indicators.get('rsi', 50):.1f}
- MACD: {indicators.get('macd', 0):.4f}
- MACD Signal: {indicators.get('macd_signal', 0):.4f}
- MACD Histogram: {indicators.get('macd_histogram', 0):.4f}
- EMA 9: ${indicators.get('ema_9', 0):,.2f}
- EMA 21: ${indicators.get('ema_21', 0):,.2f}
- EMA 50: ${indicators.get('ema_50', 0):,.2f}
- Bollinger Upper: ${indicators.get('bb_upper', 0):,.2f}
- Bollinger Lower: ${indicators.get('bb_lower', 0):,.2f}
- ATR: {indicators.get('atr', 0):.4f}
- ADX: {indicators.get('adx', 0):.1f}
- Volume Ratio: {indicators.get('volume_ratio', 1):.2f}x
"""

        if sentiment_data:
            prompt += f"""
== SENTIMENT ==
- Fear & Greed Index: {sentiment_data.get('fear_greed_index', 50) * 100:.0f}/100
- Fear & Greed Label: {sentiment_data.get('fear_greed_label', 'Neutral')}
- News Sentiment: {sentiment_data.get('news_sentiment_overall', 0.5):.2f}
- Social Buzz: {sentiment_data.get('social_buzz', 0):.1f}
- High Impact News: {sentiment_data.get('high_impact_news_count', 0)}
"""

        if orderbook_data:
            prompt += f"""
== ORDER BOOK ==
- Presión de Mercado: {orderbook_data.get('market_pressure', 'NEUTRAL')}
- Imbalance: {orderbook_data.get('imbalance', 0):+.2f}
- Bid/Ask Spread: {orderbook_data.get('spread_pct', 0):.3f}%
- Depth Score: {orderbook_data.get('depth_score', 0):.1f}
"""

        if regime_data:
            prompt += f"""
== MARKET REGIME ==
- Régimen: {regime_data.get('regime', 'SIDEWAYS')}
- Fuerza: {regime_data.get('regime_strength', 'MEDIUM')}
- Confianza: {regime_data.get('confidence', 50):.0f}%
- Volatilidad: {regime_data.get('volatility', 'NORMAL')}
"""

        if mtf_data:
            prompt += f"""
== MULTI-TIMEFRAME ==
- Tendencia 1h: {mtf_data.get('1h_trend', 'neutral')}
- Tendencia 4h: {mtf_data.get('4h_trend', 'neutral')}
- Tendencia 1d: {mtf_data.get('1d_trend', 'neutral')}
- Alineación: {mtf_data.get('alignment', 0):.0f}%
"""

        if news_summary:
            prompt += f"""
== NOTICIAS RECIENTES ==
{news_summary}
"""

        prompt += """
== INSTRUCCIONES ==
Analiza todos los datos y responde en JSON:

{
    "outlook": "BULLISH/BEARISH/NEUTRAL",
    "outlook_strength": "STRONG/MODERATE/WEAK",
    "confidence": 75,

    "key_observations": [
        "Observación 1: qué ves en los datos",
        "Observación 2: patrones importantes",
        "Observación 3: confluencias o divergencias"
    ],

    "opportunity": {
        "exists": true,
        "type": "LONG/SHORT/NONE",
        "reasoning": "Por qué hay o no hay oportunidad",
        "entry_zone": [precio_bajo, precio_alto],
        "stop_loss": precio,
        "take_profit": precio,
        "risk_reward": 2.5,
        "timeframe": "4h-1d"
    },

    "risks": [
        "Riesgo 1",
        "Riesgo 2"
    ],

    "what_to_watch": [
        "Nivel clave a observar",
        "Evento o indicador importante"
    ],

    "contrarian_view": "Si la mayoría piensa X, considera Y porque...",

    "summary": "Resumen de 2-3 oraciones sobre la situación actual"
}

IMPORTANTE: Si no hay oportunidad clara, di "opportunity.exists": false.
Un análisis honesto es más valioso que forzar una señal.
"""
        return prompt

    async def generate_signal(
        self,
        pair: str,
        indicators: Dict,
        sentiment_data: Optional[Dict] = None,
        orderbook_data: Optional[Dict] = None,
        regime_data: Optional[Dict] = None,
        existing_signal: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Generate a trading signal based on GPT analysis

        Args:
            pair: Trading pair
            indicators: Technical indicators
            sentiment_data: Sentiment data
            orderbook_data: Order book data
            regime_data: Market regime
            existing_signal: Signal from traditional system (to validate/enhance)

        Returns:
            GPT-generated or validated signal
        """
        prompt = self._build_signal_prompt(
            pair, indicators, sentiment_data, orderbook_data,
            regime_data, existing_signal
        )

        try:
            response = await self.gpt.analyze(
                system_prompt=self.SIGNAL_SYSTEM_PROMPT,
                user_prompt=prompt,
                temperature=0.4,  # Lower temperature for more consistent signals
                max_tokens=1000,
                json_response=True
            )

            signal_data = response["data"]

            # Track signals
            self.signals_generated.append({
                "pair": pair,
                "signal": signal_data,
                "timestamp": datetime.now().isoformat(),
                "cost": response["cost"]
            })

            if signal_data.get("action") not in ["HOLD", "NO_TRADE"]:
                self.opportunities_found += 1
                if signal_data.get("confidence", 0) >= 70:
                    self.signals_approved += 1

            logger.info(
                f"GPT Signal for {pair}: {signal_data.get('action', 'HOLD')} "
                f"(confidence={signal_data.get('confidence', 0)}%)"
            )

            return {
                "success": True,
                "pair": pair,
                "signal": signal_data,
                "cost": response["cost"],
                "source": "GPT_ANALYST"
            }

        except Exception as e:
            logger.error(f"Signal generation failed for {pair}: {e}")
            return {
                "success": False,
                "pair": pair,
                "error": str(e),
                "signal": {"action": "HOLD", "reason": "GPT analysis unavailable"}
            }

    def _build_signal_prompt(
        self,
        pair: str,
        indicators: Dict,
        sentiment_data: Optional[Dict],
        orderbook_data: Optional[Dict],
        regime_data: Optional[Dict],
        existing_signal: Optional[Dict]
    ) -> str:
        """Build signal generation prompt"""

        current_price = indicators.get('current_price', 0)
        rsi = indicators.get('rsi', 50)

        prompt = f"""
GENERA UNA SEÑAL DE TRADING PARA {pair}

== DATOS DE MERCADO ==
Precio: ${current_price:,.2f}
RSI: {rsi:.1f}
MACD: {indicators.get('macd', 0):.4f} (Signal: {indicators.get('macd_signal', 0):.4f})
EMA 9/21/50: ${indicators.get('ema_9', 0):,.2f} / ${indicators.get('ema_21', 0):,.2f} / ${indicators.get('ema_50', 0):,.2f}
Volume Ratio: {indicators.get('volume_ratio', 1):.2f}x
ATR: {indicators.get('atr', 0):.4f}
"""

        if sentiment_data:
            fg = sentiment_data.get('fear_greed_index', 0.5) * 100
            prompt += f"""
Fear & Greed: {fg:.0f}/100 ({sentiment_data.get('fear_greed_label', 'Neutral')})
News Sentiment: {sentiment_data.get('news_sentiment_overall', 0.5):.2f}
"""

        if orderbook_data:
            prompt += f"""
Order Book: {orderbook_data.get('market_pressure', 'NEUTRAL')} (imbalance: {orderbook_data.get('imbalance', 0):+.2f})
"""

        if regime_data:
            prompt += f"""
Régimen: {regime_data.get('regime', 'SIDEWAYS')} ({regime_data.get('regime_strength', 'MEDIUM')})
"""

        if existing_signal:
            prompt += f"""
== SEÑAL EXISTENTE DEL SISTEMA ==
El sistema tradicional generó: {existing_signal.get('action', 'HOLD')}
Score: {existing_signal.get('score', 0):.1f}/10
Razones: {', '.join(existing_signal.get('reasons', [])[:3])}
"""

        prompt += """
== GENERA SEÑAL ==
Responde en JSON:

{
    "action": "STRONG_BUY/BUY/WEAK_BUY/HOLD/WEAK_SELL/SELL/STRONG_SELL",
    "confidence": 75,

    "reasoning": {
        "main_factor": "Factor principal que determina la señal",
        "supporting_factors": ["Factor 2", "Factor 3"],
        "concerns": ["Preocupación que reduce confianza"]
    },

    "trade_setup": {
        "entry_price": precio_sugerido,
        "stop_loss": precio_stop,
        "take_profit": precio_tp,
        "position_size_recommendation": "FULL/HALF/QUARTER",
        "risk_reward": 2.0
    },

    "timing": {
        "urgency": "IMMEDIATE/SOON/WAIT",
        "valid_for_hours": 4,
        "wait_for": "Condición para entrar si no es inmediato"
    },

    "validation_of_existing": {
        "agrees_with_system": true,
        "enhancement": "Cómo mejora o contradice la señal existente"
    },

    "summary": "Resumen de 1-2 oraciones de por qué esta señal"
}

REGLAS:
- Solo STRONG_BUY/STRONG_SELL si confianza >= 80%
- Solo BUY/SELL si confianza >= 65%
- HOLD si confianza < 65% o no hay setup claro
- Risk/Reward mínimo 1:1.5
"""
        return prompt

    async def scan_opportunities(
        self,
        pairs_data: List[Dict],
        top_n: int = 5
    ) -> List[Dict]:
        """
        Scan multiple pairs for the best opportunities

        Args:
            pairs_data: List of dicts with pair data (pair, indicators, etc.)
            top_n: Number of top opportunities to return

        Returns:
            List of best opportunities sorted by score
        """
        prompt = self._build_scan_prompt(pairs_data)

        try:
            response = await self.gpt.analyze(
                system_prompt=self.ANALYST_SYSTEM_PROMPT,
                user_prompt=prompt,
                temperature=0.5,
                max_tokens=2000,
                json_response=True
            )

            scan_results = response["data"]
            self.last_full_scan = datetime.now()

            opportunities = scan_results.get("opportunities", [])

            logger.info(
                f"GPT Scan complete: Found {len(opportunities)} opportunities, "
                f"cost: ${response['cost']:.4f}"
            )

            return {
                "success": True,
                "opportunities": opportunities[:top_n],
                "market_summary": scan_results.get("market_summary", ""),
                "best_opportunity": opportunities[0] if opportunities else None,
                "avoid_pairs": scan_results.get("avoid_pairs", []),
                "cost": response["cost"]
            }

        except Exception as e:
            logger.error(f"Opportunity scan failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "opportunities": []
            }

    def _build_scan_prompt(self, pairs_data: List[Dict]) -> str:
        """Build multi-pair scan prompt"""

        pairs_summary = []
        for data in pairs_data[:20]:  # Limit to 20 pairs
            pair = data.get('pair', 'UNKNOWN')
            indicators = data.get('indicators', {})
            sentiment = data.get('sentiment', {})

            pairs_summary.append(f"""
{pair}:
  Price: ${indicators.get('current_price', 0):,.2f}
  RSI: {indicators.get('rsi', 50):.1f}
  MACD: {'Bullish' if indicators.get('macd', 0) > indicators.get('macd_signal', 0) else 'Bearish'}
  Volume: {indicators.get('volume_ratio', 1):.1f}x
  F&G: {sentiment.get('fear_greed_index', 0.5) * 100:.0f}
""")

        prompt = f"""
ESCANEA ESTOS PARES Y ENCUENTRA LAS MEJORES OPORTUNIDADES:

{chr(10).join(pairs_summary)}

== INSTRUCCIONES ==
Analiza todos los pares y encuentra las mejores oportunidades.
Responde en JSON:

{{
    "market_summary": "Resumen del estado general del mercado en 2-3 oraciones",

    "opportunities": [
        {{
            "rank": 1,
            "pair": "BTC/USDT",
            "action": "BUY/SELL",
            "score": 85,
            "reason": "Por qué es la mejor oportunidad",
            "entry": precio,
            "stop_loss": precio,
            "take_profit": precio,
            "urgency": "HIGH/MEDIUM/LOW"
        }}
    ],

    "avoid_pairs": [
        {{
            "pair": "XXX/USDT",
            "reason": "Por qué evitar este par"
        }}
    ],

    "market_conditions": {{
        "overall_sentiment": "BULLISH/BEARISH/NEUTRAL",
        "volatility": "HIGH/MEDIUM/LOW",
        "trend_strength": "STRONG/MODERATE/WEAK"
    }}
}}

Solo incluye oportunidades con score >= 70.
Si no hay buenas oportunidades, devuelve lista vacía con explicación en market_summary.
"""
        return prompt

    async def validate_signal(
        self,
        pair: str,
        signal: Dict,
        market_context: Dict
    ) -> Dict[str, Any]:
        """
        Validate a signal from the traditional system

        Args:
            pair: Trading pair
            signal: Signal to validate
            market_context: Current market context

        Returns:
            Validation result with approve/reject and modifications
        """
        prompt = f"""
VALIDA ESTA SEÑAL DE TRADING:

Par: {pair}
Señal: {signal.get('action', 'HOLD')}
Score: {signal.get('score', 0):.1f}/10
Confianza: {signal.get('confidence', 0)}%
Razones: {', '.join(signal.get('reasons', []))}

Contexto de mercado:
{json.dumps(market_context, indent=2)}

Responde en JSON:
{{
    "approved": true,
    "confidence_adjustment": 0,
    "position_size_modifier": 1.0,
    "modifications": {{
        "adjust_entry": null,
        "adjust_stop_loss": null,
        "adjust_take_profit": null
    }},
    "warnings": [],
    "reasoning": "Por qué aprobar o rechazar"
}}

REGLAS:
- approved=false si detectas riesgo no considerado
- approved=false si el contexto contradice la señal
- Modifica position_size_modifier (0.5-1.5) según confianza
"""

        try:
            response = await self.gpt.analyze(
                system_prompt="Eres un validador de señales de trading. Sé conservador.",
                user_prompt=prompt,
                temperature=0.3,
                max_tokens=2000,  # Must be enough for reasoning + response
                json_response=True
            )

            return {
                "success": True,
                "validation": response["data"],
                "cost": response["cost"]
            }

        except Exception as e:
            logger.error(f"Signal validation failed: {e}")
            return {
                "success": False,
                "validation": {"approved": True, "reasoning": "Validation unavailable"},
                "error": str(e)
            }

    async def get_market_commentary(
        self,
        pair: str,
        timeframe: str = "4h"
    ) -> str:
        """
        Get natural language market commentary

        Args:
            pair: Trading pair
            timeframe: Analysis timeframe

        Returns:
            Market commentary string
        """
        # Check cache
        cache_key = f"{pair}_{timeframe}"
        if cache_key in self.analysis_cache:
            cached = self.analysis_cache[cache_key]
            if datetime.now() - cached["timestamp"] < timedelta(minutes=15):
                analysis = cached["analysis"]
                return analysis.get("summary", "Análisis no disponible")

        return "Ejecuta /gpt_scan para obtener análisis actualizado del mercado."

    def get_stats(self) -> Dict[str, Any]:
        """Get analyst statistics"""
        return {
            "signals_generated": len(self.signals_generated),
            "opportunities_found": self.opportunities_found,
            "signals_approved": self.signals_approved,
            "approval_rate": (
                self.signals_approved / max(1, self.opportunities_found) * 100
            ),
            "last_full_scan": (
                self.last_full_scan.isoformat() if self.last_full_scan else None
            ),
            "cached_analyses": len(self.analysis_cache)
        }

    def clear_cache(self):
        """Clear analysis cache"""
        self.analysis_cache.clear()
