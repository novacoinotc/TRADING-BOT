"""
LLM Integration Module - GPT-Powered Trading Intelligence

This module provides advanced AI reasoning capabilities using OpenAI's GPT models
to enhance trading decisions, analyze performance, and optimize strategies.

Components:
- GPTClient: Base client for OpenAI API
- GPTMetaReasoner: Analyzes performance and suggests improvements
- GPTDecisionExplainer: Explains trading decisions in natural language
- GPTRiskAssessor: Evaluates risk before trades
- GPTStrategyAdvisor: Modifies parameters based on reasoning
- GPTBrain: Central orchestrator for all GPT components
"""

from src.llm.gpt_client import GPTClient
from src.llm.gpt_meta_reasoner import GPTMetaReasoner
from src.llm.gpt_decision_explainer import GPTDecisionExplainer
from src.llm.gpt_risk_assessor import GPTRiskAssessor
from src.llm.gpt_strategy_advisor import GPTStrategyAdvisor
from src.llm.gpt_brain import GPTBrain

__all__ = [
    'GPTClient',
    'GPTMetaReasoner',
    'GPTDecisionExplainer',
    'GPTRiskAssessor',
    'GPTStrategyAdvisor',
    'GPTBrain'
]
