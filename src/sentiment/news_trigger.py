"""
News-Triggered Trading - Detecta noticias críticas ANTES del mercado
Aprovecha GROWTH API para entrar en trades con ventaja temporal de 5-30 min
"""
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class NewsTrigger:
    """
    Sistema de trading activado por noticias críticas

    Detecta:
    1. Noticias importantes (high importance votes)
    2. Buzz en Twitter/Reddit
    3. Pre-pump patterns
    4. Sentiment shifts
    """

    def __init__(self):
        # Thresholds para detectar noticias críticas
        self.importance_threshold = 0.4  # >40% votes importantes
        self.engagement_threshold = 30    # >30 saves + comments
        self.social_buzz_threshold = 10   # >10 noticias sociales

        # Time window para noticias recientes (minutos)
        self.recent_news_window = 5

        # Tracking de noticias ya procesadas
        self.processed_news_ids = set()

        logger.info("📰 News-Triggered Trading inicializado")
        logger.info(f"   Importance threshold: {self.importance_threshold}")
        logger.info(f"   Engagement threshold: {self.engagement_threshold}")
        logger.info(f"   Recent window: {self.recent_news_window} min")

    def detect_critical_news(self, news_list: List[Dict]) -> List[Dict]:
        """
        Detecta noticias CRÍTICAS que pueden mover el mercado

        Args:
            news_list: Lista de noticias de CryptoPanic

        Returns:
            Lista de noticias críticas con metadatos de trading
        """
        critical_news = []
        now = datetime.now()

        for news in news_list:
            # Skip si ya procesamos esta noticia
            news_id = news.get('id')
            if news_id in self.processed_news_ids:
                continue

            # Verificar si es reciente (últimos N minutos)
            try:
                published_at = datetime.fromisoformat(
                    news.get('published_at', '').replace('Z', '+00:00')
                )
                minutes_ago = (now - published_at).total_seconds() / 60

                if minutes_ago > self.recent_news_window:
                    continue  # Demasiado vieja

            except:
                continue

            # Criterios de criticidad
            importance_ratio = news.get('importance_ratio', 0)
            engagement = news.get('engagement_score', 0)
            mentions_top10 = news.get('mentions_top10', False)

            # CRITICAL = Alta importance + Alto engagement + Top 10 coin
            is_critical = (
                importance_ratio >= self.importance_threshold and
                engagement >= self.engagement_threshold and
                mentions_top10
            )

            if is_critical:
                # Calcular criticality score (0-100)
                criticality_score = min(100, int(
                    (importance_ratio * 50) +          # Max 50 pts
                    (min(engagement, 100) * 0.3) +     # Max 30 pts
                    (20 if mentions_top10 else 0)      # Max 20 pts
                ))

                critical_item = {
                    'news': news,
                    'criticality_score': criticality_score,
                    'minutes_ago': minutes_ago,
                    'currencies': news.get('currencies', []),
                    'importance_ratio': importance_ratio,
                    'engagement': engagement,
                    'source_type': news.get('source_type', 'unknown')
                }

                critical_news.append(critical_item)
                self.processed_news_ids.add(news_id)

                logger.warning(
                    f"🚨 CRITICAL NEWS DETECTED: {news.get('title', 'Unknown')[:50]}... "
                    f"(Score: {criticality_score}, Coins: {news.get('currencies', [])})"
                )

        # Limpiar processed_ids viejos (mantener solo últimos 1000)
        if len(self.processed_news_ids) > 1000:
            self.processed_news_ids = set(list(self.processed_news_ids)[-1000:])

        return critical_news

    def detect_social_buzz(self, news_list: List[Dict]) -> Optional[Dict]:
        """
        Detecta explosión de buzz en Twitter/Reddit

        Returns:
            Dict con análisis de buzz o None
        """
        twitter_count = 0
        reddit_count = 0
        total_engagement = 0
        affected_currencies = set()

        # Analizar últimos 15 minutos
        now = datetime.now()
        cutoff = now - timedelta(minutes=15)

        for news in news_list:
            try:
                published_at = datetime.fromisoformat(
                    news.get('published_at', '').replace('Z', '+00:00')
                )
                if published_at < cutoff:
                    continue
            except:
                continue

            source_type = news.get('source_type', '')

            if source_type == 'twitter':
                twitter_count += 1
            elif source_type == 'reddit':
                reddit_count += 1

            total_engagement += news.get('engagement_score', 0)

            # Agregar monedas mencionadas
            for currency in news.get('currencies', []):
                affected_currencies.add(currency)

        social_count = twitter_count + reddit_count

        # Detectar buzz
        if social_count >= self.social_buzz_threshold:
            buzz_data = {
                'twitter_count': twitter_count,
                'reddit_count': reddit_count,
                'total_social': social_count,
                'avg_engagement': total_engagement / max(social_count, 1),
                'affected_currencies': list(affected_currencies),
                'buzz_level': 'HIGH' if social_count > 20 else 'MEDIUM'
            }

            logger.warning(
                f"📱 SOCIAL BUZZ DETECTED: {social_count} posts "
                f"(Twitter: {twitter_count}, Reddit: {reddit_count}) "
                f"Coins: {list(affected_currencies)}"
            )

            return buzz_data

        return None

    def detect_pre_pump(
        self,
        news_list: List[Dict],
        current_price_changes: Dict[str, float]
    ) -> List[Dict]:
        """
        Detecta patrones PRE-PUMP (antes de que suba el precio)

        Pattern:
        - Alto buzz social
        - Alta importance
        - Top 10-20 coin
        - Precio aún no subió (lateral o bajando)

        Args:
            news_list: Noticias recientes
            current_price_changes: Dict {pair: %change_1h}

        Returns:
            Lista de pares con alta probabilidad de pump
        """
        pre_pump_candidates = []

        # Agrupar por moneda
        currency_data = {}

        for news in news_list:
            # Solo últimos 30 min
            try:
                published_at = datetime.fromisoformat(
                    news.get('published_at', '').replace('Z', '+00:00')
                )
                minutes_ago = (datetime.now() - published_at).total_seconds() / 60
                if minutes_ago > 30:
                    continue
            except:
                continue

            for currency in news.get('currencies', []):
                if currency not in currency_data:
                    currency_data[currency] = {
                        'news_count': 0,
                        'total_importance': 0,
                        'total_engagement': 0,
                        'social_count': 0,
                        'min_rank': 999
                    }

                data = currency_data[currency]
                data['news_count'] += 1
                data['total_importance'] += news.get('importance_ratio', 0)
                data['total_engagement'] += news.get('engagement_score', 0)

                if news.get('source_type') in ['twitter', 'reddit']:
                    data['social_count'] += 1

                # Market rank
                rank = news.get('min_market_rank', 999)
                if rank < data['min_rank']:
                    data['min_rank'] = rank

        # Analizar cada moneda
        for currency, data in currency_data.items():
            if data['news_count'] < 3:
                continue  # Muy pocas noticias

            avg_importance = data['total_importance'] / data['news_count']
            avg_engagement = data['total_engagement'] / data['news_count']

            # Verificar patrón pre-pump
            has_social_buzz = data['social_count'] >= 3
            high_importance = avg_importance > 0.3
            is_top_coin = data['min_rank'] <= 20

            # Verificar precio (debe estar lateral o bajando)
            pair = f"{currency}/USDT"
            price_change = current_price_changes.get(pair, 0)
            price_not_pumped = price_change < 2.0  # Subida < 2%

            # PRE-PUMP = buzz + importance + top coin + precio no subió aún
            if has_social_buzz and high_importance and is_top_coin and price_not_pumped:
                pre_pump_score = min(100, int(
                    (avg_importance * 40) +
                    (min(avg_engagement, 50) * 0.6) +
                    (data['social_count'] * 2) +
                    (30 if is_top_coin else 0)
                ))

                pre_pump_candidates.append({
                    'currency': currency,
                    'pair': pair,
                    'pre_pump_score': pre_pump_score,
                    'news_count': data['news_count'],
                    'social_count': data['social_count'],
                    'avg_importance': avg_importance,
                    'avg_engagement': avg_engagement,
                    'market_rank': data['min_rank'],
                    'current_price_change': price_change
                })

                logger.warning(
                    f"🎯 PRE-PUMP DETECTED: {pair} "
                    f"(Score: {pre_pump_score}, News: {data['news_count']}, "
                    f"Social: {data['social_count']}, Rank: {data['min_rank']})"
                )

        # Ordenar por score (mayor primero)
        pre_pump_candidates.sort(key=lambda x: x['pre_pump_score'], reverse=True)

        return pre_pump_candidates

    def generate_news_signal(
        self,
        critical_news: List[Dict],
        social_buzz: Optional[Dict],
        pre_pump: List[Dict],
        current_pairs: List[str]
    ) -> List[Dict]:
        """
        Genera señales de trading basadas en noticias

        Returns:
            Lista de señales con pair, action, confidence, reason
        """
        signals = []

        # Señales de PRE-PUMP (prioridad alta)
        for pump in pre_pump[:3]:  # Top 3 candidatos
            pair = pump['pair']

            if pair not in current_pairs:
                continue  # No trading este par

            signals.append({
                'pair': pair,
                'action': 'BUY',
                'confidence': min(95, 70 + (pump['pre_pump_score'] // 4)),
                'reason': f"PRE-PUMP: {pump['news_count']} news, {pump['social_count']} social",
                'source': 'news_trigger_pre_pump',
                'metadata': {
                    'pre_pump_score': pump['pre_pump_score'],
                    'urgency': 'HIGH',
                    'expected_move': '+0.5-2%',
                    'timeframe': '5-30min'
                }
            })

        # Señales de CRITICAL NEWS
        for critical in critical_news[:5]:  # Top 5 noticias
            news = critical['news']
            currencies = critical['currencies']

            for currency in currencies:
                pair = f"{currency}/USDT"

                if pair not in current_pairs:
                    continue

                # Determinar acción basándose en sentiment
                vote_sentiment = news.get('vote_sentiment', 0)
                action = 'BUY' if vote_sentiment > 0 else 'SELL'

                confidence = min(90, 60 + (critical['criticality_score'] // 3))

                signals.append({
                    'pair': pair,
                    'action': action,
                    'confidence': confidence,
                    'reason': f"CRITICAL NEWS: {news.get('title', '')[:40]}...",
                    'source': 'news_trigger_critical',
                    'metadata': {
                        'criticality_score': critical['criticality_score'],
                        'importance_ratio': critical['importance_ratio'],
                        'engagement': critical['engagement'],
                        'minutes_ago': critical['minutes_ago'],
                        'urgency': 'HIGH'
                    }
                })

        # Señales de SOCIAL BUZZ
        if social_buzz and social_buzz['buzz_level'] == 'HIGH':
            for currency in social_buzz['affected_currencies'][:3]:
                pair = f"{currency}/USDT"

                if pair not in current_pairs:
                    continue

                confidence = min(85, 65 + social_buzz['total_social'])

                signals.append({
                    'pair': pair,
                    'action': 'BUY',
                    'confidence': confidence,
                    'reason': f"SOCIAL BUZZ: {social_buzz['total_social']} posts",
                    'source': 'news_trigger_social',
                    'metadata': {
                        'twitter_count': social_buzz['twitter_count'],
                        'reddit_count': social_buzz['reddit_count'],
                        'buzz_level': social_buzz['buzz_level'],
                        'urgency': 'MEDIUM'
                    }
                })

        if signals:
            logger.info(f"📊 Generated {len(signals)} news-triggered signals")

        return signals
