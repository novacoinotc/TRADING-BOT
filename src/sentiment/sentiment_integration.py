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
from src.sentiment.news_trigger import NewsTrigger

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
        self.news_trigger = NewsTrigger()  # GROWTH API - News-triggered trading

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

        # Recolectar noticias (1 request general para todas las monedas - más eficiente)
        all_news = self.news_collector.collect_all_news(currencies=None)

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

        # Update state (incluir raw news para GROWTH features)
        self.current_sentiment = {
            'timestamp': datetime.now().isoformat(),
            'general': general_analysis,
            'fear_greed': fear_greed,
            'pair_sentiments': pair_sentiments,
            'high_impact_news': analyzed_high_impact,
            'critical_news': critical_news,
            'raw_news': all_news.get('cryptopanic', [])  # GROWTH API - noticias completas
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

        # GROWTH API - Calcular features avanzados desde noticias raw
        all_news = self.current_sentiment.get('raw_news', [])

        # Calcular engagement, importance, source types
        total_engagement = 0
        total_importance_votes = 0
        total_votes = 0
        twitter_count = 0
        reddit_count = 0
        top10_mentions = 0
        total_market_cap_mentioned = 0

        for news in all_news:
            total_engagement += news.get('engagement_score', 0)
            total_importance_votes += news.get('votes_important', 0)
            total_votes += news.get('votes_positive', 0) + news.get('votes_negative', 0)

            if news.get('source_type') == 'twitter':
                twitter_count += 1
            elif news.get('source_type') == 'reddit':
                reddit_count += 1

            if news.get('mentions_top10', False):
                top10_mentions += 1

            total_market_cap_mentioned += news.get('total_market_cap', 0)

        # Features básicas + GROWTH exclusivas
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

            # GROWTH API - Engagement Features
            'avg_engagement_score': total_engagement / max(len(all_news), 1),
            'total_engagement': total_engagement,

            # GROWTH API - Importance Features
            'importance_votes_ratio': total_importance_votes / max(total_votes, 1),
            'high_importance_news': 1 if total_importance_votes > 50 else 0,

            # GROWTH API - Source Type Features
            'twitter_news_count': twitter_count,
            'reddit_news_count': reddit_count,
            'twitter_ratio': twitter_count / max(len(all_news), 1),
            'reddit_ratio': reddit_count / max(len(all_news), 1),
            'social_buzz': 1 if (twitter_count + reddit_count) > 10 else 0,

            # GROWTH API - Market Data Features
            'top10_mentions_count': top10_mentions,
            'top10_mentions_ratio': top10_mentions / max(len(all_news), 1),
            'total_market_cap_mentioned': total_market_cap_mentioned / 1e12,  # Normalizar a trillones
            'high_market_cap_news': 1 if total_market_cap_mentioned > 1e12 else 0,

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

        # MULTI-LAYER CONFIDENCE SYSTEM (5 capas)
        adjustment = 0
        adjustment_breakdown = {}

        # LAYER 1: Fear & Greed (básico)
        fg = features['fear_greed_index']
        fg_adjust = 0
        if signal.get('action') == 'BUY':
            if fg < 0.3:  # Fear -> más confianza en BUY
                fg_adjust = 5
            elif fg > 0.7:  # Greed -> menos confianza en BUY
                fg_adjust = -5
        elif signal.get('action') == 'SELL':
            if fg > 0.7:  # Greed -> más confianza en SELL
                fg_adjust = 5
            elif fg < 0.3:  # Fear -> menos confianza en SELL
                fg_adjust = -5

        adjustment += fg_adjust
        adjustment_breakdown['fear_greed'] = fg_adjust

        # LAYER 2: News Sentiment (básico)
        news_sentiment = features['news_sentiment_overall']
        news_adjust = 0
        if signal.get('action') == 'BUY':
            if news_sentiment > 0.7:  # Very positive
                news_adjust = 8
            elif news_sentiment < 0.3:  # Very negative
                news_adjust = -8
        elif signal.get('action') == 'SELL':
            if news_sentiment < 0.3:
                news_adjust = 8
            elif news_sentiment > 0.7:
                news_adjust = -8

        adjustment += news_adjust
        adjustment_breakdown['news_sentiment'] = news_adjust

        # LAYER 3: GROWTH - News Importance (NUEVO)
        importance_adjust = 0
        importance_ratio = features.get('importance_votes_ratio', 0)
        high_importance = features.get('high_importance_news', 0)

        if importance_ratio > 0.4 or high_importance == 1:
            # Noticias importantes detectadas
            if signal.get('action') == 'BUY':
                importance_adjust = 10  # Aumentar confianza
            elif signal.get('action') == 'SELL':
                importance_adjust = 5
        elif importance_ratio > 0.25:
            importance_adjust = 5

        adjustment += importance_adjust
        adjustment_breakdown['importance'] = importance_adjust

        # LAYER 4: GROWTH - Social Buzz (NUEVO)
        social_adjust = 0
        social_buzz = features.get('social_buzz', 0)
        twitter_ratio = features.get('twitter_ratio', 0)
        reddit_ratio = features.get('reddit_ratio', 0)

        if social_buzz == 1:
            # Alto buzz en redes sociales
            if signal.get('action') == 'BUY':
                social_adjust = 8  # Buzz generalmente precede pump
            elif signal.get('action') == 'SELL':
                social_adjust = 3
        elif twitter_ratio > 0.15 or reddit_ratio > 0.1:
            # Buzz moderado
            social_adjust = 4

        adjustment += social_adjust
        adjustment_breakdown['social_buzz'] = social_adjust

        # LAYER 5: GROWTH - Market Cap Filter (NUEVO)
        market_cap_adjust = 0
        top10_ratio = features.get('top10_mentions_ratio', 0)
        high_market_cap = features.get('high_market_cap_news', 0)

        if top10_ratio > 0.5 or high_market_cap == 1:
            # Noticias sobre grandes caps (más seguras)
            market_cap_adjust = 5
        elif top10_ratio > 0.3:
            market_cap_adjust = 3

        adjustment += market_cap_adjust
        adjustment_breakdown['market_cap'] = market_cap_adjust

        # LAYER 6: High impact news (volatilidad)
        volatility_adjust = 0
        high_impact_count = features['high_impact_news_count']
        if high_impact_count > 5:
            # Mucha volatilidad esperada, reducir confianza ligeramente
            volatility_adjust = -3
        elif high_impact_count > 10:
            volatility_adjust = -5

        adjustment += volatility_adjust
        adjustment_breakdown['volatility'] = volatility_adjust

        # Apply adjustment
        adjusted_confidence = max(0, min(100, original_confidence + adjustment))

        adjusted_signal['confidence'] = adjusted_confidence
        adjusted_signal['sentiment_adjustment'] = adjustment
        adjusted_signal['sentiment_adjustment_breakdown'] = adjustment_breakdown
        adjusted_signal['sentiment_features'] = features

        if adjustment != 0:
            # Log detallado con breakdown
            breakdown_str = ", ".join([f"{k}: {v:+d}" for k, v in adjustment_breakdown.items() if v != 0])
            logger.info(
                f"📊 Confidence ajustada: {original_confidence}% → {adjusted_confidence}% "
                f"({adjustment:+d}%) [{breakdown_str}]"
            )

        return adjusted_signal

    def get_news_triggered_signals(
        self,
        current_pairs: List[str],
        current_price_changes: Dict[str, float] = None
    ) -> List[Dict]:
        """
        Genera señales de trading basadas en noticias críticas (GROWTH API)

        Args:
            current_pairs: Lista de pares disponibles para trading
            current_price_changes: Dict con cambios de precio 1h {pair: %change}

        Returns:
            Lista de señales urgentes generadas por noticias
        """
        if not self.current_sentiment:
            return []

        all_news = self.current_sentiment.get('raw_news', [])

        if not all_news:
            return []

        # Detectar noticias críticas
        critical_news = self.news_trigger.detect_critical_news(all_news)

        # Detectar buzz social
        social_buzz = self.news_trigger.detect_social_buzz(all_news)

        # Detectar patrones pre-pump
        pre_pump = []
        if current_price_changes:
            pre_pump = self.news_trigger.detect_pre_pump(all_news, current_price_changes)

        # Generar señales
        signals = self.news_trigger.generate_news_signal(
            critical_news=critical_news,
            social_buzz=social_buzz,
            pre_pump=pre_pump,
            current_pairs=current_pairs
        )

        return signals

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
        """Retorna features neutrales cuando no hay datos (incluye GROWTH features)"""
        return {
            # Basic features
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
            'sentiment_trend_improving': 0,

            # GROWTH API features (neutral defaults)
            'avg_engagement_score': 0,
            'total_engagement': 0,
            'importance_votes_ratio': 0,
            'high_importance_news': 0,
            'twitter_news_count': 0,
            'reddit_news_count': 0,
            'twitter_ratio': 0,
            'reddit_ratio': 0,
            'social_buzz': 0,
            'top10_mentions_count': 0,
            'top10_mentions_ratio': 0,
            'total_market_cap_mentioned': 0,
            'high_market_cap_news': 0,
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
