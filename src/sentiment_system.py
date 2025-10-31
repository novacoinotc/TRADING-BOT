"""
Sentiment Analysis System
Integrates news and social sentiment from CryptoPanic API
"""
import requests
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import defaultdict
import time

logger = logging.getLogger(__name__)


class SentimentSystem:
    """
    Analyzes market sentiment using CryptoPanic news and social data
    """

    def __init__(self, api_key: str):
        """
        Initialize sentiment system

        Args:
            api_key: CryptoPanic API key
        """
        self.api_key = api_key
        self.base_url = "https://cryptopanic.com/api/v1"
        self.cache = {}
        self.cache_duration = 900  # 15 minutes cache
        self.last_update = {}

    def _get_currency_code(self, pair: str) -> str:
        """
        Extract currency code from trading pair

        Args:
            pair: Trading pair (e.g., 'BTC/USD')

        Returns:
            Currency code (e.g., 'BTC')
        """
        return pair.split('/')[0]

    def _fetch_news(self, currency: Optional[str] = None) -> List[Dict]:
        """
        Fetch news from CryptoPanic API

        Args:
            currency: Specific currency to filter (optional)

        Returns:
            List of news items
        """
        try:
            params = {
                'auth_token': self.api_key,
                'public': 'true',
                'kind': 'news'  # Only news articles (not social media)
            }

            if currency:
                params['currencies'] = currency.upper()

            response = requests.get(
                f"{self.base_url}/posts/",
                params=params,
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                return data.get('results', [])
            else:
                logger.error(f"CryptoPanic API error: {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return []

    def _calculate_sentiment_score(self, news_items: List[Dict]) -> Dict:
        """
        Calculate sentiment score from news items

        Args:
            news_items: List of news items

        Returns:
            Sentiment analysis dict
        """
        if not news_items:
            return {
                'score': 0,  # Neutral
                'positive': 0,
                'negative': 0,
                'neutral': 0,
                'total_news': 0,
                'keywords': []
            }

        positive_count = 0
        negative_count = 0
        neutral_count = 0
        keywords = defaultdict(int)

        # Analyze each news item
        for item in news_items:
            # Check for positive/negative votes
            votes = item.get('votes', {})
            positive = votes.get('positive', 0)
            negative = votes.get('negative', 0)

            # Classify sentiment based on votes
            if positive > negative:
                positive_count += 1
            elif negative > positive:
                negative_count += 1
            else:
                neutral_count += 1

            # Extract keywords from title
            title = item.get('title', '').lower()
            for word in ['bullish', 'bull', 'surge', 'rally', 'gain', 'up']:
                if word in title:
                    keywords[word] += 1

            for word in ['bearish', 'bear', 'crash', 'drop', 'fall', 'down']:
                if word in title:
                    keywords[word] += 1

        total = len(news_items)

        # Calculate normalized score (-1 to 1)
        if total > 0:
            score = (positive_count - negative_count) / total
        else:
            score = 0

        # Get top keywords
        top_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            'score': round(score, 2),
            'positive': positive_count,
            'negative': negative_count,
            'neutral': neutral_count,
            'total_news': total,
            'keywords': [k for k, v in top_keywords],
            'timestamp': datetime.now().isoformat()
        }

    def update(self, pair: Optional[str] = None):
        """
        Update sentiment data for a specific pair or all pairs

        Args:
            pair: Trading pair (optional, if None updates general market)
        """
        cache_key = pair if pair else 'market'

        # Check if cache is still valid
        if cache_key in self.last_update:
            time_since_update = time.time() - self.last_update[cache_key]
            if time_since_update < self.cache_duration:
                logger.debug(f"Using cached sentiment for {cache_key}")
                return

        # Fetch fresh data
        currency = self._get_currency_code(pair) if pair else None
        news_items = self._fetch_news(currency)

        # Calculate sentiment
        sentiment = self._calculate_sentiment_score(news_items)

        # Update cache
        self.cache[cache_key] = sentiment
        self.last_update[cache_key] = time.time()

        logger.info(
            f"Updated sentiment for {cache_key}: "
            f"score={sentiment['score']} ({sentiment['total_news']} news)"
        )

    def get_sentiment(self, pair: str) -> Dict:
        """
        Get sentiment data for a specific pair

        Args:
            pair: Trading pair

        Returns:
            Sentiment data dict
        """
        # Try to get specific pair sentiment
        if pair in self.cache:
            return self.cache[pair]

        # Fall back to general market sentiment
        if 'market' in self.cache:
            return self.cache['market']

        # No data available, return neutral
        return {
            'score': 0,
            'positive': 0,
            'negative': 0,
            'neutral': 0,
            'total_news': 0,
            'keywords': [],
            'timestamp': None
        }

    def get_sentiment_features(self, pair: str) -> Dict:
        """
        Get sentiment features for ML integration

        Args:
            pair: Trading pair

        Returns:
            Feature dict suitable for ML models
        """
        sentiment = self.get_sentiment(pair)

        return {
            'sentiment_score': sentiment['score'],
            'sentiment_positive_ratio': (
                sentiment['positive'] / sentiment['total_news']
                if sentiment['total_news'] > 0 else 0
            ),
            'sentiment_negative_ratio': (
                sentiment['negative'] / sentiment['total_news']
                if sentiment['total_news'] > 0 else 0
            ),
            'news_volume': sentiment['total_news']
        }

    def get_sentiment_signal(self, pair: str) -> Dict:
        """
        Get actionable sentiment signal

        Args:
            pair: Trading pair

        Returns:
            Signal dict with action and strength
        """
        sentiment = self.get_sentiment(pair)
        score = sentiment['score']
        total_news = sentiment['total_news']

        # Determine action based on score
        if score >= 0.3 and total_news >= 3:
            action = 'bullish'
            strength = min(3, int(abs(score) * 3))
        elif score <= -0.3 and total_news >= 3:
            action = 'bearish'
            strength = min(3, int(abs(score) * 3))
        else:
            action = 'neutral'
            strength = 0

        return {
            'action': action,
            'strength': strength,
            'score': score,
            'confidence': min(100, total_news * 10),  # More news = more confidence
            'total_news': total_news,
            'keywords': sentiment['keywords']
        }
