"""
LLM Integration Module - GPT-Powered Trading Intelligence (ABSOLUTE CONTROL)

This module provides advanced AI reasoning capabilities using OpenAI's GPT models
with ABSOLUTE CONTROL over all trading decisions.

Components:
- GPTClient: Base client for OpenAI API
- GPTMetaReasoner: Analyzes performance and suggests improvements
- GPTDecisionExplainer: Explains trading decisions in natural language
- GPTRiskAssessor: Evaluates risk before trades
- GPTStrategyAdvisor: Modifies parameters based on reasoning
- GPTMarketAnalyst: Analyzes market and generates trading signals
- GPTExperienceLearner: Persistent learning from trade history
- GPTTradeController: ABSOLUTE CONTROL over trading decisions
- GPTDataProvider: Comprehensive data integration (ALL sources)
- GPTBrain: Central orchestrator for all GPT components
"""

from src.llm.gpt_client import GPTClient
from src.llm.gpt_meta_reasoner import GPTMetaReasoner
from src.llm.gpt_decision_explainer import GPTDecisionExplainer
from src.llm.gpt_risk_assessor import GPTRiskAssessor
from src.llm.gpt_strategy_advisor import GPTStrategyAdvisor
from src.llm.gpt_market_analyst import GPTMarketAnalyst
from src.llm.gpt_experience_learner import GPTExperienceLearner
from src.llm.gpt_trade_controller import GPTTradeController
from src.llm.gpt_data_provider import GPTDataProvider
from src.llm.gpt_brain import GPTBrain

__all__ = [
    'GPTClient',
    'GPTMetaReasoner',
    'GPTDecisionExplainer',
    'GPTRiskAssessor',
    'GPTStrategyAdvisor',
    'GPTMarketAnalyst',
    'GPTExperienceLearner',
    'GPTTradeController',
    'GPTDataProvider',
    'GPTBrain'
]
