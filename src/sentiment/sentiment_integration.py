"""
Sentiment Integration - Integra análisis de sentiment con ML y trading
Coordina News Collector, Sentiment Analyzer y Fear & Greed
"""
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import json

from src.sentiment.news_collector import NewsCollector
from src.sentiment.sentiment_analyzer import SentimentAnalyzer

logger = logging.getLogger(__name__)


class SentimentIntegration:
    """
    Sistema completo de sentiment analysis integrado con trading

    Features:
    - Recolecta noticias cada 15 minutos
    - Analiza sentiment en tiempo real
    - Genera features ML adicionales
    - Ajusta confianza de señales según sentiment
    - Alerta de noticias críticas
    """

    def __init__(
        self,
        cryptopanic_api_key: Optional[str] = None,
        update_interval_minutes: int = 15,
        enable_blocking: bool = True
    ):
        self.news_collector = NewsCollector(cryptopanic_api_key=cryptopanic_api_key)
        self.sentiment_analyzer = SentimentAnalyzer()

        self.update_interval = update_interval_minutes
        self.enable_blocking = enable_blocking  # Bloquear trades en sentiment extremo negativo

        # Cache
        self.cache_dir = Path('data/sentiment')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sentiment_history_file = self.cache_dir / 'sentiment_history.json'

        # Estado
        self.current_sentiment = {}
        self.sentiment_history = {}
        self.last_critical_news = []

        # Load historical data
        self._load_history()

    def update(self, currencies: List[str] = None) -> Dict:
        """
        Actualiza sentiment de mercado

        Args:
            currencies: Lista de monedas a monitorear

        Returns:
            Dict con sentiment actualizado
        """
        # Verificar si es tiempo de actualizar
        if not self.news_collector.should_update(self.update_interval):
            logger.debug("Sentiment cache is fresh, skipping update")
            return self.current_sentiment

        logger.info("📰 Actualizando sentiment analysis...")

        # Recolectar noticias
        all_news = self.news_collector.collect_all_news(currencies=currencies)

        # Analizar sentiment general
        general_analysis = self.sentiment_analyzer.analyze_bulk(
            all_news.get('cryptopanic', [])
        )

        # Fear & Greed
        fear_greed = all_news.get('fear_greed', {})

        # Sentiment por par
        pair_sentiments = {}
        if currencies:
            for currency in currencies:
                pair = f"{currency}/USDT"
                pair_analysis = self.sentiment_analyzer.analyze_for_pair(
                    pair,
                    all_news.get('cryptopanic', []),
                    hours=24
                )
                pair_sentiments[pair] = pair_analysis

        # High impact news
        high_impact = self.news_collector.get_high_impact_news(hours=24, min_votes=10)
        analyzed_high_impact = [
            self.sentiment_analyzer.analyze_news(news) for news in high_impact
        ]

        # Critical news (very positive or very negative + high impact)
        critical_news = [
            news for news in analyzed_high_impact
            if (news['sentiment_score'] > 0.75 or news['sentiment_score'] < 0.25)
        ]

        # Update state
        self.current_sentiment = {
            'timestamp': datetime.now().isoformat(),
            'general': general_analysis,
            'fear_greed': fear_greed,
            'pair_sentiments': pair_sentiments,
            'high_impact_news': analyzed_high_impact,
            'critical_news': critical_news
        }

        # Save to history
        self._save_to_history()

        # Check for new critical news
        self._check_critical_news(critical_news)

        logger.info(f"✅ Sentiment actualizado:")
        logger.info(f"   Overall: {general_analysis['sentiment_label']}")
        logger.info(f"   Fear & Greed: {fear_greed.get('value', 50)} ({fear_greed.get('value_classification', 'N/A')})")
        logger.info(f"   High Impact News: {len(analyzed_high_impact)}")
        logger.info(f"   Critical News: {len(critical_news)}")

        return self.current_sentiment

    def get_sentiment_features(self, pair: str) -> Dict:
        """
        Genera features de sentiment para ML

        Args:
            pair: Par de trading (ej: 'BTC/USDT')

        Returns:
            Dict con features de sentiment
        """
        if not self.current_sentiment:
            return self._get_neutral_features()

        general = self.current_sentiment.get('general', {})
        fear_greed = self.current_sentiment.get('fear_greed', {})
        pair_sentiment = self.current_sentiment.get('pair_sentiments', {}).get(pair, {})

        # Features básicas
        features = {
            # General sentiment
            'news_sentiment_overall': general.get('overall_sentiment', 0.5),
            'news_count_24h': general.get('total_news', 0),
            'news_positive_ratio': general.get('positive_news', 0) / max(general.get('total_news', 1), 1),
            'news_negative_ratio': general.get('negative_news', 0) / max(general.get('total_news', 1), 1),

            # Fear & Greed
            'fear_greed_index': fear_greed.get('value', 50) / 100.0,  # Normalizar a 0-1
            'fear_greed_extreme_fear': 1 if fear_greed.get('value', 50) < 25 else 0,
            'fear_greed_extreme_greed': 1 if fear_greed.get('value', 50) > 75 else 0,

            # Pair-specific
            'pair_sentiment': pair_sentiment.get('overall_sentiment', 0.5),
            'pair_news_count': pair_sentiment.get('total_news', 0),

            # High impact
            'high_impact_news_count': len(self.current_sentiment.get('high_impact_news', [])),
            'critical_news_count': len(self.current_sentiment.get('critical_news', [])),

            # Momentum (requiere histórico)
            'sentiment_momentum': 0,
            'sentiment_trend_improving': 0
        }

        # Calcular momentum si hay histórico
        if pair in self.sentiment_history:
            history = self.sentiment_history[pair][-7:]  # Últimos 7 registros
            if history:
                historical_sentiments = [h['sentiment'] for h in history]
                momentum_data = self.sentiment_analyzer.calculate_sentiment_momentum(
                    current_sentiment=features['pair_sentiment'],
                    historical_sentiments=historical_sentiments
                )
                features['sentiment_momentum'] = momentum_data['momentum']
                features['sentiment_trend_improving'] = 1 if momentum_data['trend'] == 'IMPROVING' else 0

        return features

    def should_block_trade(self, pair: str, signal: Dict) -> bool:
        """
        Determina si debe bloquear un trade por sentiment negativo

        Args:
            pair: Par de trading
            signal: Señal de trading

        Returns:
            True si debe bloquear
        """
        if not self.enable_blocking:
            return False

        if not self.current_sentiment:
            return False

        # Fear & Greed extremo
        fear_greed = self.current_sentiment.get('fear_greed', {})
        fg_value = fear_greed.get('value', 50)

        # Bloquear en extreme fear (< 20) para señales BUY
        if signal.get('action') == 'BUY' and fg_value < 20:
            logger.warning(f"⛔ Trade bloqueado por Extreme Fear (FG={fg_value})")
            return True

        # Bloquear en extreme greed (> 80) para señales BUY agresivas
        if signal.get('action') == 'BUY' and fg_value > 80 and signal.get('score', 0) < 8:
            logger.warning(f"⛔ Trade bloqueado por Extreme Greed (FG={fg_value})")
            return True

        # Critical news muy negativa
        critical_news = self.current_sentiment.get('critical_news', [])
        very_negative_critical = [
            news for news in critical_news
            if news['sentiment_score'] < 0.25
        ]

        if len(very_negative_critical) >= 2:  # 2+ noticias críticas negativas
            logger.warning(f"⛔ Trade bloqueado por {len(very_negative_critical)} noticias críticas negativas")
            return False

        return False

    def adjust_signal_confidence(self, pair: str, signal: Dict) -> Dict:
        """
        Ajusta confianza de señal basándose en sentiment

        Args:
            pair: Par de trading
            signal: Señal original

        Returns:
            Señal con confianza ajustada
        """
        if not self.current_sentiment:
            return signal

        adjusted_signal = signal.copy()
        original_confidence = signal.get('confidence', 50)

        # Get sentiment features
        features = self.get_sentiment_features(pair)

        # Ajustes
        adjustment = 0

        # Fear & Greed
        fg = features['fear_greed_index']
        if signal.get('action') == 'BUY':
            if fg < 0.3:  # Fear -> más confianza en BUY
                adjustment += 5
            elif fg > 0.7:  # Greed -> menos confianza en BUY
                adjustment -= 5
        elif signal.get('action') == 'SELL':
            if fg > 0.7:  # Greed -> más confianza en SELL
                adjustment += 5
            elif fg < 0.3:  # Fear -> menos confianza en SELL
                adjustment -= 5

        # News sentiment
        news_sentiment = features['news_sentiment_overall']
        if signal.get('action') == 'BUY':
            if news_sentiment > 0.7:  # Very positive
                adjustment += 8
            elif news_sentiment < 0.3:  # Very negative
                adjustment -= 8
        elif signal.get('action') == 'SELL':
            if news_sentiment < 0.3:
                adjustment += 8
            elif news_sentiment > 0.7:
                adjustment -= 8

        # High impact news
        high_impact_count = features['high_impact_news_count']
        if high_impact_count > 3:
            # Mucha volatilidad esperada, reducir confianza
            adjustment -= 3

        # Apply adjustment
        adjusted_confidence = max(0, min(100, original_confidence + adjustment))

        adjusted_signal['confidence'] = adjusted_confidence
        adjusted_signal['sentiment_adjustment'] = adjustment
        adjusted_signal['sentiment_features'] = features

        if adjustment != 0:
            logger.info(f"📊 Confidence ajustada: {original_confidence}% → {adjusted_confidence}% ({adjustment:+d}%)")

        return adjusted_signal

    def get_critical_news_alerts(self) -> List[Dict]:
        """Retorna noticias críticas para alertar en Telegram"""
        return self.current_sentiment.get('critical_news', [])

    def _check_critical_news(self, critical_news: List[Dict]):
        """Verifica si hay noticias críticas nuevas"""
        # Comparar con anteriores
        new_critical = []

        for news in critical_news:
            news_id = news.get('title', '')
            if news_id not in [n.get('title', '') for n in self.last_critical_news]:
                new_critical.append(news)

        if new_critical:
            logger.warning(f"🚨 {len(new_critical)} noticias críticas NUEVAS detectadas:")
            for news in new_critical:
                logger.warning(f"   - {news.get('title')} ({news.get('sentiment_label')})")

        self.last_critical_news = critical_news

    def _save_to_history(self):
        """Guarda sentiment actual al histórico"""
        timestamp = datetime.now().isoformat()

        # Guardar por par
        pair_sentiments = self.current_sentiment.get('pair_sentiments', {})
        for pair, sentiment in pair_sentiments.items():
            if pair not in self.sentiment_history:
                self.sentiment_history[pair] = []

            self.sentiment_history[pair].append({
                'timestamp': timestamp,
                'sentiment': sentiment.get('overall_sentiment', 0.5),
                'fear_greed': self.current_sentiment.get('fear_greed', {}).get('value', 50)
            })

            # Mantener solo últimos 30 registros
            if len(self.sentiment_history[pair]) > 30:
                self.sentiment_history[pair] = self.sentiment_history[pair][-30:]

        # Guardar a disco
        try:
            with open(self.sentiment_history_file, 'w') as f:
                json.dump(self.sentiment_history, f, indent=2)
        except Exception as e:
            logger.error(f"Error guardando historial de sentiment: {e}")

    def _load_history(self):
        """Carga histórico de sentiment"""
        if not self.sentiment_history_file.exists():
            return

        try:
            with open(self.sentiment_history_file, 'r') as f:
                self.sentiment_history = json.load(f)

            logger.debug(f"Historial de sentiment cargado: {len(self.sentiment_history)} pares")
        except Exception as e:
            logger.error(f"Error cargando historial: {e}")

    def _get_neutral_features(self) -> Dict:
        """Retorna features neutrales cuando no hay datos"""
        return {
            'news_sentiment_overall': 0.5,
            'news_count_24h': 0,
            'news_positive_ratio': 0.5,
            'news_negative_ratio': 0.5,
            'fear_greed_index': 0.5,
            'fear_greed_extreme_fear': 0,
            'fear_greed_extreme_greed': 0,
            'pair_sentiment': 0.5,
            'pair_news_count': 0,
            'high_impact_news_count': 0,
            'critical_news_count': 0,
            'sentiment_momentum': 0,
            'sentiment_trend_improving': 0
        }

    def get_summary(self) -> str:
        """Genera resumen de sentiment para logging/Telegram"""
        if not self.current_sentiment:
            return "Sentiment analysis no disponible"

        general = self.current_sentiment.get('general', {})
        fear_greed = self.current_sentiment.get('fear_greed', {})

        summary = f"""
📰 SENTIMENT ANALYSIS

Overall Sentiment: {general.get('sentiment_label', 'N/A')} ({general.get('overall_sentiment', 0):.2f})
Fear & Greed Index: {fear_greed.get('value', 50)} ({fear_greed.get('value_classification', 'N/A')})

News 24h: {general.get('total_news', 0)} total
  ✅ Positive: {general.get('positive_news', 0)}
  ❌ Negative: {general.get('negative_news', 0)}
  ⚪ Neutral: {general.get('neutral_news', 0)}

High Impact: {len(self.current_sentiment.get('high_impact_news', []))} news
Critical: {len(self.current_sentiment.get('critical_news', []))} news
"""
        return summary.strip()
