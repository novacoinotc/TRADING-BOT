"""
Sentiment Analyzer - Analiza sentiment de noticias con NLP
Usa análisis de palabras clave y scoring para determinar sentiment
"""
import logging
import re
from typing import Dict, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Analiza sentiment de noticias cripto

    Métodos:
    - Keyword-based sentiment (positivo, negativo, neutral)
    - Impact classification (HIGH, MEDIUM, LOW)
    - Category detection (REGULATION, ADOPTION, HACK, etc)
    """

    def __init__(self):
        # Keywords positivos
        self.positive_keywords = {
            # Adoption
            'adoption', 'adopt', 'approved', 'approval', 'partnership', 'integrate',
            'institutional', 'mainstream', 'acceptance', 'legalize',
            # Price positive
            'surge', 'rally', 'bullish', 'bull', 'moon', 'pump', 'soar', 'skyrocket',
            'breakthrough', 'all-time high', 'ath', 'record high', 'breakout',
            # Tech positive
            'upgrade', 'improvement', 'innovation', 'launch', 'release', 'milestone',
            'success', 'successful', 'achievement', 'breakthrough',
            # Investment
            'invest', 'investment', 'fund', 'capital', 'inflow', 'buying', 'accumulate',
            'etf approved', 'institutional buying', 'whale buying',
            # General positive
            'positive', 'optimistic', 'confidence', 'growth', 'recovery', 'strong'
        }

        # Keywords negativos
        self.negative_keywords = {
            # Regulation negative
            'ban', 'banned', 'crackdown', 'lawsuit', 'sue', 'investigation',
            'illegal', 'prohibit', 'restriction', 'sanction', 'fraud',
            # Security
            'hack', 'hacked', 'exploit', 'vulnerability', 'breach', 'stolen',
            'scam', 'rug pull', 'collapse', 'insolvent',
            # Price negative
            'crash', 'dump', 'plunge', 'bearish', 'bear', 'decline', 'drop',
            'fall', 'down', 'lose', 'losses', 'sell-off', 'panic',
            # Market negative
            'delisting', 'suspended', 'halt', 'freeze', 'concern', 'worry',
            'fear', 'uncertainty', 'risk', 'warning', 'alert',
            # General negative
            'negative', 'pessimistic', 'failure', 'failed', 'problem', 'issue'
        }

        # High impact events
        self.high_impact_keywords = {
            'etf', 'sec', 'regulation', 'federal reserve', 'fed', 'government',
            'institutional', 'blackrock', 'fidelity', 'jp morgan', 'goldman sachs',
            'hack', 'exploit', 'billion', 'trillion', 'ban', 'lawsuit',
            'hard fork', 'upgrade', 'merge', 'halving', 'adoption'
        }

        # Categories
        self.category_keywords = {
            'REGULATION': ['sec', 'regulation', 'regulatory', 'government', 'ban', 'lawsuit', 'legal'],
            'ADOPTION': ['adoption', 'adopt', 'mainstream', 'partnership', 'integrate', 'accept'],
            'INSTITUTIONAL': ['institutional', 'blackrock', 'fidelity', 'etf', 'fund', 'investment'],
            'SECURITY': ['hack', 'exploit', 'vulnerability', 'breach', 'stolen', 'scam'],
            'DEVELOPMENT': ['upgrade', 'update', 'development', 'launch', 'release', 'fork'],
            'MACRO': ['fed', 'federal reserve', 'inflation', 'interest rate', 'gdp', 'recession'],
            'MARKET': ['price', 'trading', 'volume', 'liquidity', 'exchange', 'listing']
        }

    def analyze_news(self, news: Dict) -> Dict:
        """
        Analiza sentiment de una noticia individual

        Args:
            news: Dict con título y contenido de noticia

        Returns:
            Dict con sentiment score y metadata
        """
        title = news.get('title', '').lower()
        text = title  # Por ahora solo título, se puede expandir

        # Sentiment score
        sentiment_score = self._calculate_sentiment(text)

        # Impact level
        impact = self._calculate_impact(text)

        # Category
        category = self._detect_category(text)

        # Relevance por moneda
        currencies = news.get('currencies', [])

        return {
            'sentiment_score': sentiment_score,
            'sentiment_label': self._get_sentiment_label(sentiment_score),
            'impact': impact,
            'category': category,
            'currencies': currencies,
            'title': news.get('title'),
            'published_at': news.get('published_at'),
            'source': news.get('source')
        }

    def analyze_bulk(self, news_list: List[Dict]) -> Dict:
        """
        Analiza múltiples noticias y retorna agregado

        Args:
            news_list: Lista de noticias

        Returns:
            Dict con sentiment agregado
        """
        if not news_list:
            return {
                'overall_sentiment': 0.5,
                'sentiment_label': 'NEUTRAL',
                'total_news': 0,
                'positive_news': 0,
                'negative_news': 0,
                'neutral_news': 0,
                'high_impact_news': 0
            }

        analyzed = [self.analyze_news(news) for news in news_list]

        # Calcular promedios
        sentiments = [a['sentiment_score'] for a in analyzed]
        avg_sentiment = sum(sentiments) / len(sentiments)

        # Contar por categoría
        positive = sum(1 for s in sentiments if s > 0.6)
        negative = sum(1 for s in sentiments if s < 0.4)
        neutral = len(sentiments) - positive - negative

        # High impact
        high_impact = sum(1 for a in analyzed if a['impact'] == 'HIGH')

        return {
            'overall_sentiment': round(avg_sentiment, 3),
            'sentiment_label': self._get_sentiment_label(avg_sentiment),
            'total_news': len(news_list),
            'positive_news': positive,
            'negative_news': negative,
            'neutral_news': neutral,
            'high_impact_news': high_impact,
            'analyzed_news': analyzed
        }

    def analyze_for_pair(self, pair: str, news_list: List[Dict], hours: int = 24) -> Dict:
        """
        Analiza sentiment específico para un par

        Args:
            pair: Par de trading (ej: 'BTC/USDT')
            news_list: Lista de noticias
            hours: Ventana de tiempo

        Returns:
            Dict con sentiment para ese par
        """
        base = pair.split('/')[0]

        # Filtrar noticias relevantes
        cutoff = datetime.now() - timedelta(hours=hours)
        relevant_news = []

        for news in news_list:
            currencies = news.get('currencies', [])
            if base in currencies:
                try:
                    pub_date = datetime.fromisoformat(news['published_at'].replace('Z', '+00:00'))
                    if pub_date >= cutoff:
                        relevant_news.append(news)
                except:
                    pass

        # Analizar
        analysis = self.analyze_bulk(relevant_news)
        analysis['pair'] = pair
        analysis['base_currency'] = base
        analysis['hours_analyzed'] = hours

        return analysis

    def _calculate_sentiment(self, text: str) -> float:
        """
        Calcula sentiment score (0-1)

        0 = muy negativo
        0.5 = neutral
        1 = muy positivo
        """
        text = text.lower()

        positive_count = sum(1 for keyword in self.positive_keywords if keyword in text)
        negative_count = sum(1 for keyword in self.negative_keywords if keyword in text)

        total = positive_count + negative_count

        if total == 0:
            return 0.5  # Neutral

        # Calcular score
        score = (positive_count - negative_count) / (total * 2) + 0.5

        # Clamp entre 0 y 1
        return max(0.0, min(1.0, score))

    def _calculate_impact(self, text: str) -> str:
        """
        Calcula nivel de impacto: HIGH, MEDIUM, LOW
        """
        text = text.lower()

        high_impact_count = sum(1 for keyword in self.high_impact_keywords if keyword in text)

        if high_impact_count >= 2:
            return 'HIGH'
        elif high_impact_count == 1:
            return 'MEDIUM'
        else:
            return 'LOW'

    def _detect_category(self, text: str) -> str:
        """
        Detecta categoría de noticia
        """
        text = text.lower()

        category_scores = {}

        for category, keywords in self.category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                category_scores[category] = score

        if not category_scores:
            return 'GENERAL'

        # Retornar categoría con mayor score
        return max(category_scores, key=category_scores.get)

    def _get_sentiment_label(self, score: float) -> str:
        """
        Convierte score numérico a label
        """
        if score >= 0.7:
            return 'VERY_POSITIVE'
        elif score >= 0.6:
            return 'POSITIVE'
        elif score > 0.4:
            return 'NEUTRAL'
        elif score > 0.3:
            return 'NEGATIVE'
        else:
            return 'VERY_NEGATIVE'

    def get_recent_highlights(self, analyzed_news: List[Dict], limit: int = 5) -> List[Dict]:
        """
        Obtiene las noticias más relevantes (high impact + extreme sentiment)

        Args:
            analyzed_news: Lista de noticias ya analizadas
            limit: Número máximo de highlights

        Returns:
            Lista de noticias destacadas
        """
        # Filtrar high impact
        highlights = [
            news for news in analyzed_news
            if news['impact'] == 'HIGH' and
            (news['sentiment_score'] > 0.7 or news['sentiment_score'] < 0.3)
        ]

        # Ordenar por extremos de sentiment
        highlights.sort(key=lambda x: abs(x['sentiment_score'] - 0.5), reverse=True)

        return highlights[:limit]

    def calculate_sentiment_momentum(
        self,
        current_sentiment: float,
        historical_sentiments: List[float]
    ) -> Dict:
        """
        Calcula momentum de sentiment (está mejorando o empeorando?)

        Args:
            current_sentiment: Sentiment actual
            historical_sentiments: Lista de sentiments históricos

        Returns:
            Dict con momentum y trend
        """
        if not historical_sentiments or len(historical_sentiments) < 2:
            return {
                'momentum': 0,
                'trend': 'NEUTRAL',
                'change': 0
            }

        # Calcular promedio histórico
        avg_historical = sum(historical_sentiments) / len(historical_sentiments)

        # Momentum = diferencia con histórico
        momentum = current_sentiment - avg_historical

        # Change reciente (vs último valor)
        recent_change = current_sentiment - historical_sentiments[-1]

        # Trend
        if momentum > 0.1:
            trend = 'IMPROVING'
        elif momentum < -0.1:
            trend = 'DETERIORATING'
        else:
            trend = 'STABLE'

        return {
            'momentum': round(momentum, 3),
            'trend': trend,
            'change': round(recent_change, 3),
            'avg_historical': round(avg_historical, 3)
        }
