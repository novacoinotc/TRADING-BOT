"""
News Collector - Recolecta noticias de múltiples fuentes
Fuentes: CryptoPanic, Fear & Greed Index, NewsAPI, CoinGecko
"""
import requests
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import json
import time

logger = logging.getLogger(__name__)


class NewsCollector:
    """
    Recolecta noticias cripto de múltiples fuentes

    Fuentes gratuitas:
    - CryptoPanic API (crypto news agregadas)
    - Fear & Greed Index (sentiment general)
    - CoinGecko (eventos importantes)
    """

    def __init__(
        self,
        cryptopanic_api_key: Optional[str] = None,
        cache_dir: str = 'data/sentiment'
    ):
        self.cryptopanic_api_key = cryptopanic_api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # URLs - GROWTH API v2 (real-time data)
        self.cryptopanic_url = 'https://cryptopanic.com/api/growth/v2/posts/'
        self.fear_greed_url = 'https://api.alternative.me/fng/'

        # Cache (30 segundos recomendado por CryptoPanic - server-side cache)
        self.news_cache_file = self.cache_dir / 'news_cache.json'
        self.cache_duration_seconds = 30  # Official recommendation
        self.last_update = None
        self.cached_news = []

        # Load cache
        self._load_cache()

    def collect_all_news(self, currencies: List[str] = None) -> Dict:
        """
        Recolecta noticias de todas las fuentes

        Args:
            currencies: Lista de monedas a monitorear (ej: ['BTC', 'ETH'])

        Returns:
            Dict con noticias agregadas
        """
        logger.info("📰 Recolectando noticias de todas las fuentes...")

        all_news = {
            'cryptopanic': [],
            'fear_greed': {},
            'timestamp': datetime.now().isoformat()
        }

        # CryptoPanic
        if self.cryptopanic_api_key:
            try:
                # Request general (no filtrar por moneda) - más eficiente con quota
                cp_news = self.get_cryptopanic_news(currencies=None, limit=100)
                all_news['cryptopanic'] = cp_news
                logger.info(f"   ✅ CryptoPanic: {len(cp_news)} noticias (todas las monedas)")
            except Exception as e:
                logger.error(f"   ❌ Error en CryptoPanic: {e}")
        else:
            logger.warning("   ⚠️ CryptoPanic API key no configurada")

        # Fear & Greed
        try:
            fg_data = self.get_fear_greed_index()
            all_news['fear_greed'] = fg_data
            logger.info(f"   ✅ Fear & Greed: {fg_data.get('value')} ({fg_data.get('value_classification')})")
        except Exception as e:
            logger.error(f"   ❌ Error en Fear & Greed: {e}")

        # Cache results
        self._save_cache(all_news)

        return all_news

    def get_cryptopanic_news(
        self,
        currencies: List[str] = None,
        limit: int = 50,
        filter_type: str = 'all'  # 'all', 'hot', 'rising', 'bullish', 'bearish'
    ) -> List[Dict]:
        """
        Obtiene noticias de CryptoPanic

        Args:
            currencies: Lista de monedas (ej: ['BTC', 'ETH'])
            limit: Número máximo de noticias
            filter_type: Tipo de filtro

        Returns:
            Lista de noticias
        """
        if not self.cryptopanic_api_key:
            return []

        params = {
            'auth_token': self.cryptopanic_api_key,
            'public': 'true',
            'filter': filter_type
        }

        if currencies:
            params['currencies'] = ','.join(currencies)

        # Retry logic for transient errors (502, 503, 504)
        max_retries = 3
        retry_delay = 2  # seconds

        for attempt in range(max_retries):
            try:
                response = requests.get(self.cryptopanic_url, params=params, timeout=10)

                # Handle transient server errors with retry
                if response.status_code in [502, 503, 504]:
                    if attempt < max_retries - 1:
                        logger.warning(f"CryptoPanic {response.status_code} error, retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries})")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    else:
                        logger.warning(f"CryptoPanic {response.status_code} error after {max_retries} attempts - using cached data")
                        return self.cached_news if self.cached_news else []

                response.raise_for_status()
                break  # Success, exit retry loop

            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    logger.warning(f"CryptoPanic timeout, retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    logger.warning(f"CryptoPanic timeout after {max_retries} attempts - using cached data")
                    return self.cached_news if self.cached_news else []

            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    logger.warning(f"CryptoPanic request error: {e}, retrying in {retry_delay}s")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    logger.error(f"CryptoPanic failed after {max_retries} attempts: {e}")
                    return self.cached_news if self.cached_news else []

        try:
            data = response.json()

            if 'results' not in data:
                return []

            news = []
            for item in data['results'][:limit]:
                # GROWTH API v2 - Extraer TODOS los campos disponibles
                votes = item.get('votes', {})
                instruments = item.get('instruments', [])
                source = item.get('source', {})

                news_item = {
                    # Basic fields
                    'id': item.get('id'),
                    'slug': item.get('slug', ''),
                    'title': item.get('title'),
                    'url': item.get('url'),
                    'original_url': item.get('original_url', ''),
                    'published_at': item.get('published_at'),
                    'created_at': item.get('created_at'),
                    'kind': item.get('kind', 'news'),  # news, media, blog, twitter, reddit

                    # Source metadata (GROWTH exclusive)
                    'source_title': source.get('title', 'Unknown'),
                    'source_domain': source.get('domain', ''),
                    'source_type': source.get('type', 'feed'),  # feed, blog, twitter, media, reddit
                    'source_region': source.get('region', 'en'),

                    # Author (GROWTH)
                    'author': item.get('author', ''),

                    # Instruments with market data (GROWTH exclusive)
                    'instruments': instruments,
                    'currencies': [inst.get('code') for inst in instruments],

                    # Votes detallados (GROWTH enhanced)
                    'votes': votes,
                    'votes_negative': votes.get('negative', 0),
                    'votes_positive': votes.get('positive', 0),
                    'votes_important': votes.get('important', 0),
                    'votes_liked': votes.get('liked', 0),
                    'votes_disliked': votes.get('disliked', 0),
                    'votes_lol': votes.get('lol', 0),
                    'votes_toxic': votes.get('toxic', 0),
                    'votes_saved': votes.get('saved', 0),
                    'votes_comments': votes.get('comments', 0),
                }

                # Calcular sentiment mejorado basado en votes
                positive = votes.get('positive', 0)
                negative = votes.get('negative', 0)
                important = votes.get('important', 0)
                saved = votes.get('saved', 0)
                comments = votes.get('comments', 0)

                total_votes = positive + negative
                if total_votes > 0:
                    news_item['vote_sentiment'] = (positive - negative) / total_votes
                else:
                    news_item['vote_sentiment'] = 0

                # Engagement score (saved + comments = cuánto interesa)
                news_item['engagement_score'] = saved + comments

                # Importance score (important votes / total)
                if total_votes > 0:
                    news_item['importance_ratio'] = important / total_votes
                else:
                    news_item['importance_ratio'] = 0

                # Extractar market data de instruments
                if instruments:
                    # Market caps de todas las monedas mencionadas
                    news_item['total_market_cap'] = sum(
                        inst.get('market_cap_usd', 0) for inst in instruments
                    )

                    # Menor market rank = más importante (BTC=1, ETH=2, etc)
                    market_ranks = [inst.get('market_rank', 999) for inst in instruments if inst.get('market_rank')]
                    news_item['min_market_rank'] = min(market_ranks) if market_ranks else 999

                    # Detectar si menciona top 10
                    news_item['mentions_top10'] = any(
                        inst.get('market_rank', 999) <= 10 for inst in instruments
                    )
                else:
                    news_item['total_market_cap'] = 0
                    news_item['min_market_rank'] = 999
                    news_item['mentions_top10'] = False

                news.append(news_item)

            return news

        except Exception as e:
            logger.error(f"Error obteniendo noticias de CryptoPanic: {e}")
            return []

    def get_fear_greed_index(self, days: int = 1) -> Dict:
        """
        Obtiene Fear & Greed Index

        Args:
            days: Número de días de histórico (1-365)

        Returns:
            Dict con valor actual y clasificación
        """
        params = {'limit': days}

        try:
            response = requests.get(self.fear_greed_url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            if 'data' not in data or len(data['data']) == 0:
                return {}

            current = data['data'][0]

            result = {
                'value': int(current.get('value', 50)),
                'value_classification': current.get('value_classification', 'Neutral'),
                'timestamp': current.get('timestamp'),
                'time_until_update': current.get('time_until_update')
            }

            # Agregar histórico si hay más de 1 día
            if days > 1:
                result['history'] = [
                    {
                        'value': int(item.get('value', 50)),
                        'classification': item.get('value_classification'),
                        'timestamp': item.get('timestamp')
                    }
                    for item in data['data']
                ]

            return result

        except Exception as e:
            logger.error(f"Error obteniendo Fear & Greed Index: {e}")
            return {}

    def get_trending_news(self, hours: int = 24) -> List[Dict]:
        """
        Obtiene noticias trending de las últimas N horas

        Args:
            hours: Horas hacia atrás

        Returns:
            Lista de noticias trending
        """
        all_news = self.cached_news if self.cached_news else []

        cutoff_time = datetime.now() - timedelta(hours=hours)

        trending = []
        for news in all_news:
            try:
                pub_date = datetime.fromisoformat(news['published_at'].replace('Z', '+00:00'))
                if pub_date >= cutoff_time:
                    trending.append(news)
            except:
                pass

        # Ordenar por votes (más relevantes primero)
        trending.sort(
            key=lambda x: x.get('votes', {}).get('positive', 0) - x.get('votes', {}).get('negative', 0),
            reverse=True
        )

        return trending[:10]

    def get_news_for_pair(self, pair: str, hours: int = 24) -> List[Dict]:
        """
        Obtiene noticias relevantes para un par específico

        Args:
            pair: Par de trading (ej: 'BTC/USDT')
            hours: Horas hacia atrás

        Returns:
            Lista de noticias relevantes
        """
        # Extraer base currency (ej: BTC de BTC/USDT)
        base = pair.split('/')[0]

        all_news = self.cached_news if self.cached_news else []

        cutoff_time = datetime.now() - timedelta(hours=hours)

        relevant = []
        for news in all_news:
            try:
                # Verificar si es relevante para esta moneda
                if base in news.get('currencies', []):
                    pub_date = datetime.fromisoformat(news['published_at'].replace('Z', '+00:00'))
                    if pub_date >= cutoff_time:
                        relevant.append(news)
            except:
                pass

        return relevant

    def get_high_impact_news(self, hours: int = 24, min_votes: int = 10) -> List[Dict]:
        """
        Obtiene noticias de alto impacto (muchos votes)

        Args:
            hours: Horas hacia atrás
            min_votes: Mínimo de votes totales

        Returns:
            Lista de noticias de alto impacto
        """
        trending = self.get_trending_news(hours=hours)

        high_impact = []
        for news in trending:
            votes = news.get('votes', {})
            total_votes = votes.get('positive', 0) + votes.get('negative', 0)

            if total_votes >= min_votes:
                news['total_votes'] = total_votes
                high_impact.append(news)

        return high_impact

    def _save_cache(self, news_data: Dict):
        """Guarda noticias en cache"""
        try:
            with open(self.news_cache_file, 'w') as f:
                json.dump(news_data, f, indent=2)

            self.cached_news = news_data.get('cryptopanic', [])
            self.last_update = datetime.now()

        except Exception as e:
            logger.error(f"Error guardando cache de noticias: {e}")

    def _load_cache(self):
        """Carga noticias desde cache"""
        if not self.news_cache_file.exists():
            return

        try:
            with open(self.news_cache_file, 'r') as f:
                data = json.load(f)

            self.cached_news = data.get('cryptopanic', [])

            # Verificar si cache es reciente (< 15 minutos)
            timestamp = data.get('timestamp')
            if timestamp:
                cache_time = datetime.fromisoformat(timestamp)
                if datetime.now() - cache_time < timedelta(minutes=15):
                    self.last_update = cache_time
                    logger.debug(f"Cache de noticias cargado ({len(self.cached_news)} items)")

        except Exception as e:
            logger.error(f"Error cargando cache de noticias: {e}")

    def should_update(self, interval_minutes: int = 15) -> bool:
        """
        Verifica si es tiempo de actualizar noticias

        Usa cache de 30s (oficial) + intervalo configurable (default 15 min)
        """
        if not self.last_update:
            return True

        time_since_update = (datetime.now() - self.last_update).total_seconds()

        # Si pasó menos de 30s, usar cache (recomendación CryptoPanic)
        if time_since_update < self.cache_duration_seconds:
            return False

        # Si pasó más de 30s pero menos del intervalo configurado, usar cache
        if time_since_update < (interval_minutes * 60):
            return False

        return True

    def get_summary(self) -> Dict:
        """Retorna resumen de noticias actuales"""
        fear_greed = self.get_fear_greed_index()
        trending = self.get_trending_news(hours=24)
        high_impact = self.get_high_impact_news(hours=24)

        return {
            'fear_greed_index': fear_greed.get('value', 50),
            'fear_greed_classification': fear_greed.get('value_classification', 'Neutral'),
            'total_news_24h': len(trending),
            'high_impact_news_24h': len(high_impact),
            'last_update': self.last_update.isoformat() if self.last_update else None
        }
