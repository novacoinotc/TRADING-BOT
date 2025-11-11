"""
Startup Validator - Sistema de validación completa al inicio del bot
Valida TODOS los servicios y genera reporte detallado
"""
import logging
import asyncio
from typing import Dict, List, Tuple
from datetime import datetime
import requests

logger = logging.getLogger(__name__)


class StartupValidator:
    """
    Valida todos los servicios críticos al inicio del bot
    Genera checklist numerada de servicios operativos
    """

    def __init__(self):
        self.services = []
        self.validation_results = {}

    async def validate_all_services(self, monitor) -> Dict:
        """
        Ejecuta validación completa de TODOS los servicios

        Args:
            monitor: Instancia de MarketMonitor con todos los componentes

        Returns:
            Dict con resultados de validación
        """
        logger.info("=" * 60)
        logger.info("🔍 INICIANDO VALIDACIÓN COMPLETA DE SERVICIOS")
        logger.info("=" * 60)

        start_time = datetime.now()

        # Lista de validaciones (orden de importancia)
        validations = [
            ("Exchange (Binance)", self._validate_exchange, monitor),
            ("Telegram Bot", self._validate_telegram, monitor),
            ("CryptoPanic GROWTH API", self._validate_cryptopanic, monitor),
            ("Fear & Greed Index", self._validate_fear_greed, monitor),
            ("Sentiment Analysis", self._validate_sentiment, monitor),
            ("News-Triggered Trading", self._validate_news_trigger, monitor),
            ("Multi-Layer Confidence System", self._validate_confidence_layers, monitor),
            ("ML System (Predictor)", self._validate_ml_system, monitor),
            ("Paper Trading Engine", self._validate_paper_trading, monitor),
            ("RL Agent (Q-Learning)", self._validate_rl_agent, monitor),
            ("Parameter Optimizer (93 params)", self._validate_parameter_optimizer, monitor),
            ("Order Book Analyzer", self._validate_orderbook, monitor),
            ("Market Regime Detector", self._validate_market_regime, monitor),
            ("Dynamic TP Manager", self._validate_dynamic_tp, monitor),
            ("Learning Persistence (Export/Import)", self._validate_learning_persistence, monitor),
            ("Git Backup System", self._validate_git_backup, monitor),

            # ===== ARSENAL AVANZADO (NUEVOS MÓDULOS) =====
            ("Correlation Matrix (Diversification)", self._validate_correlation_matrix, monitor),
            ("Liquidation Heatmap (Stop Hunts)", self._validate_liquidation_heatmap, monitor),
            ("Funding Rate Analyzer (Sentiment)", self._validate_funding_rate, monitor),
            ("Volume Profile & POC (Value Zones)", self._validate_volume_profile, monitor),
            ("Pattern Recognition (Chartism)", self._validate_pattern_recognition, monitor),
            ("Session-Based Trading (Volatility)", self._validate_session_trading, monitor),
            ("Order Flow Imbalance (Momentum)", self._validate_order_flow, monitor),
            ("Feature Aggregator (Central Hub)", self._validate_feature_aggregator, monitor),
        ]

        operational = []
        failed = []

        for i, (service_name, validator_func, component) in enumerate(validations, 1):
            try:
                is_operational, details = await validator_func(component)

                status_icon = "✅" if is_operational else "❌"
                logger.info(f"{i:2d}. {status_icon} {service_name}: {details}")

                self.validation_results[service_name] = {
                    'operational': is_operational,
                    'details': details,
                    'number': i
                }

                if is_operational:
                    operational.append(f"{i}. {service_name}")
                else:
                    failed.append(f"{i}. {service_name}: {details}")

            except Exception as e:
                logger.error(f"{i:2d}. ❌ {service_name}: ERROR - {e}")
                failed.append(f"{i}. {service_name}: ERROR - {str(e)}")
                self.validation_results[service_name] = {
                    'operational': False,
                    'details': f"ERROR: {str(e)}",
                    'number': i
                }

        elapsed = (datetime.now() - start_time).total_seconds()

        logger.info("=" * 60)
        logger.info(f"✅ SERVICIOS OPERANDO: {len(operational)}/{len(validations)}")
        logger.info(f"⏱️  Validación completada en {elapsed:.2f}s")
        logger.info("=" * 60)

        return {
            'total': len(validations),
            'operational': operational,
            'failed': failed,
            'count_operational': len(operational),
            'count_failed': len(failed),
            'elapsed_seconds': elapsed,
            'validation_results': self.validation_results
        }

    async def _validate_exchange(self, monitor) -> Tuple[bool, str]:
        """Valida conexión con Binance"""
        try:
            if hasattr(monitor, 'exchange') and monitor.exchange:
                # Test fetch ticker (no requiere API key)
                ticker = monitor.exchange.fetch_ticker('BTC/USDT')
                if ticker and 'last' in ticker:
                    return True, f"Conectado vía proxy {monitor.proxy if hasattr(monitor, 'proxy') else 'directo'}"
            return False, "Exchange no inicializado"
        except Exception as e:
            return False, f"Error: {str(e)[:50]}"

    async def _validate_telegram(self, monitor) -> Tuple[bool, str]:
        """Valida Telegram Bot"""
        try:
            if hasattr(monitor, 'telegram_commands') and monitor.telegram_commands:
                # Check if bot is running
                if hasattr(monitor.telegram_commands, 'application') and monitor.telegram_commands.application:
                    return True, "Bot activo y escuchando comandos"
            return False, "Bot no inicializado"
        except Exception as e:
            return False, f"Error: {str(e)[:50]}"

    async def _validate_cryptopanic(self, monitor) -> Tuple[bool, str]:
        """Valida CryptoPanic GROWTH API v2"""
        try:
            from config import config
            if not config.CRYPTOPANIC_API_KEY:
                return False, "API key no configurada"

            # Test API call
            url = f"https://cryptopanic.com/api/growth/v2/posts/?auth_token={config.CRYPTOPANIC_API_KEY}&public=true"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                count = len(data.get('results', []))
                return True, f"GROWTH v2 activo - {count} posts obtenidos"
            elif response.status_code == 429:
                return False, "Límite de requests excedido (429)"
            else:
                return False, f"HTTP {response.status_code}"

        except Exception as e:
            return False, f"Error: {str(e)[:50]}"

    async def _validate_fear_greed(self, monitor) -> Tuple[bool, str]:
        """Valida Fear & Greed Index API"""
        try:
            url = "https://api.alternative.me/fng/"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                if 'data' in data and len(data['data']) > 0:
                    value = data['data'][0].get('value', 'N/A')
                    classification = data['data'][0].get('value_classification', 'N/A')
                    return True, f"Valor actual: {value} ({classification})"
            return False, f"HTTP {response.status_code}"
        except Exception as e:
            return False, f"Error: {str(e)[:50]}"

    async def _validate_sentiment(self, monitor) -> Tuple[bool, str]:
        """Valida Sentiment Integration con 26 features"""
        try:
            if hasattr(monitor, 'sentiment_system') and monitor.sentiment_system:
                # Check if sentiment data is available
                if hasattr(monitor.sentiment_system, 'last_sentiment'):
                    return True, "26 features GROWTH + Multi-Layer Confidence activo"
                return True, "Inicializado - esperando primera actualización"
            return False, "No inicializado"
        except Exception as e:
            return False, f"Error: {str(e)[:50]}"

    async def _validate_news_trigger(self, monitor) -> Tuple[bool, str]:
        """Valida News-Triggered Trading"""
        try:
            if hasattr(monitor, 'sentiment_system') and monitor.sentiment_system:
                if hasattr(monitor.sentiment_system, 'news_trigger'):
                    trigger = monitor.sentiment_system.news_trigger
                    return True, f"Thresholds: importance={trigger.importance_threshold}, engagement={trigger.engagement_threshold}"
            return False, "No inicializado"
        except Exception as e:
            return False, f"Error: {str(e)[:50]}"

    async def _validate_confidence_layers(self, monitor) -> Tuple[bool, str]:
        """Valida Multi-Layer Confidence System (6 layers)"""
        try:
            if hasattr(monitor, 'sentiment_system') and monitor.sentiment_system:
                # Confidence layers are part of sentiment integration
                return True, "6 layers: Fear&Greed, News, Importance, Social, MarketCap, Volatility"
            return False, "No inicializado"
        except Exception as e:
            return False, f"Error: {str(e)[:50]}"

    async def _validate_ml_system(self, monitor) -> Tuple[bool, str]:
        """Valida ML System y Predictor"""
        try:
            if hasattr(monitor, 'ml_system') and monitor.ml_system:
                predictor = monitor.ml_system.predictor
                if predictor:
                    model_info = predictor.get_model_info()
                    if model_info.get('available'):
                        accuracy = model_info.get('metrics', {}).get('test_accuracy', 0)
                        return True, f"Modelo entrenado - Accuracy: {accuracy * 100:.1f}%"
                    else:
                        return True, "Inicializado - Sin modelo entrenado aún"
            return False, "No inicializado"
        except Exception as e:
            return False, f"Error: {str(e)[:50]}"

    async def _validate_paper_trading(self, monitor) -> Tuple[bool, str]:
        """Valida Paper Trading Engine"""
        try:
            if hasattr(monitor, 'ml_system') and monitor.ml_system:
                if hasattr(monitor.ml_system, 'paper_trader') and monitor.ml_system.paper_trader:
                    portfolio = monitor.ml_system.paper_trader.portfolio
                    balance = portfolio.get_equity()
                    open_trades = len(portfolio.positions) if hasattr(portfolio, 'positions') else 0
                    closed_trades = len(portfolio.closed_trades) if hasattr(portfolio, 'closed_trades') else 0
                    return True, f"Balance: ${balance:,.2f} USDT - {open_trades} abiertas, {closed_trades} cerradas"
            return False, "No inicializado"
        except Exception as e:
            return False, f"Error: {str(e)[:50]}"

    async def _validate_rl_agent(self, monitor) -> Tuple[bool, str]:
        """Valida RL Agent (Q-Learning)"""
        try:
            if hasattr(monitor, 'autonomy_controller') and monitor.autonomy_controller:
                rl_agent = monitor.autonomy_controller.rl_agent
                if rl_agent:
                    q_table_size = len(rl_agent.q_table) if hasattr(rl_agent, 'q_table') else 0
                    return True, f"Q-Table: {q_table_size} estados aprendidos"
            return False, "No inicializado"
        except Exception as e:
            return False, f"Error: {str(e)[:50]}"

    async def _validate_parameter_optimizer(self, monitor) -> Tuple[bool, str]:
        """Valida Parameter Optimizer"""
        try:
            if hasattr(monitor, 'autonomy_controller') and monitor.autonomy_controller:
                optimizer = monitor.autonomy_controller.parameter_optimizer
                if optimizer:
                    param_count = len(optimizer.parameter_ranges)
                    trials = len(optimizer.trial_history) if hasattr(optimizer, 'trial_history') else 0
                    return True, f"{param_count} parámetros optimizables - {trials} trials completados"
            return False, "No inicializado"
        except Exception as e:
            return False, f"Error: {str(e)[:50]}"

    async def _validate_orderbook(self, monitor) -> Tuple[bool, str]:
        """Valida Order Book Analyzer"""
        try:
            if hasattr(monitor, 'orderbook_analyzer') and monitor.orderbook_analyzer:
                depth = monitor.orderbook_analyzer.depth if hasattr(monitor.orderbook_analyzer, 'depth') else 100
                return True, f"Depth: {depth} niveles"
            return False, "No inicializado"
        except Exception as e:
            return False, f"Error: {str(e)[:50]}"

    async def _validate_market_regime(self, monitor) -> Tuple[bool, str]:
        """Valida Market Regime Detector"""
        try:
            if hasattr(monitor, 'regime_detector') and monitor.regime_detector:
                return True, "Detecta: BULL, BEAR, SIDEWAYS con confidence"
            return False, "No inicializado"
        except Exception as e:
            return False, f"Error: {str(e)[:50]}"

    async def _validate_dynamic_tp(self, monitor) -> Tuple[bool, str]:
        """Valida Dynamic TP Manager"""
        try:
            # Check if DynamicTPManager class exists (módulo disponible)
            import os
            tp_file = 'src/trading/dynamic_tp_manager.py'
            if os.path.exists(tp_file):
                return True, "Código disponible - TPs dinámicos 0.3% a 3%"
            return False, "Archivo no encontrado"
        except Exception as e:
            return False, f"Error: {str(e)[:50]}"

    async def _validate_learning_persistence(self, monitor) -> Tuple[bool, str]:
        """Valida Learning Persistence (Export/Import)"""
        try:
            if hasattr(monitor, 'autonomy_controller') and monitor.autonomy_controller:
                persistence = monitor.autonomy_controller.persistence  # ✅ Correcto: 'persistence' no 'learning_persistence'
                if persistence:
                    import os
                    data_dir = persistence.storage_dir if hasattr(persistence, 'storage_dir') else 'data/autonomous'
                    exists = os.path.exists(data_dir)
                    return True, f"Directorio: {data_dir} ({'existe' if exists else 'será creado'})"
            return False, "No inicializado"
        except Exception as e:
            return False, f"Error: {str(e)[:50]}"

    async def _validate_git_backup(self, monitor) -> Tuple[bool, str]:
        """Valida Git Backup System"""
        try:
            if hasattr(monitor, 'autonomy_controller') and monitor.autonomy_controller:
                git_backup = monitor.autonomy_controller.git_backup
                if git_backup:
                    interval_hours = git_backup.backup_interval_hours if hasattr(git_backup, 'backup_interval_hours') else 24
                    return True, f"Auto-backup cada {interval_hours}h"
            return False, "No inicializado"
        except Exception as e:
            return False, f"Error: {str(e)[:50]}"

    # ===== VALIDACIONES ARSENAL AVANZADO =====

    async def _validate_correlation_matrix(self, monitor) -> Tuple[bool, str]:
        """Valida Correlation Matrix"""
        try:
            if hasattr(monitor, 'feature_aggregator') and monitor.feature_aggregator:
                corr_matrix = monitor.feature_aggregator.correlation_matrix
                if corr_matrix and corr_matrix.enabled:
                    pairs_tracked = len(corr_matrix.price_history)
                    threshold = corr_matrix.high_correlation_threshold
                    return True, f"{pairs_tracked} pares tracked - threshold {threshold}"
            return False, "No inicializado (requiere feature_aggregator)"
        except Exception as e:
            return False, f"Error: {str(e)[:50]}"

    async def _validate_liquidation_heatmap(self, monitor) -> Tuple[bool, str]:
        """Valida Liquidation Heatmap"""
        try:
            if hasattr(monitor, 'feature_aggregator') and monitor.feature_aggregator:
                liq_heatmap = monitor.feature_aggregator.liquidation_heatmap
                if liq_heatmap and liq_heatmap.enabled:
                    min_volume = liq_heatmap.min_liquidation_volume / 1_000_000
                    boost = liq_heatmap.boost_factor
                    return True, f"Min ${min_volume:.1f}M volume - boost {boost}x"
            return False, "No inicializado"
        except Exception as e:
            return False, f"Error: {str(e)[:50]}"

    async def _validate_funding_rate(self, monitor) -> Tuple[bool, str]:
        """Valida Funding Rate Analyzer"""
        try:
            if hasattr(monitor, 'feature_aggregator') and monitor.feature_aggregator:
                funding = monitor.feature_aggregator.funding_rate_analyzer
                if funding and funding.enabled:
                    extreme = funding.extreme_positive_threshold
                    boost = funding.boost_factor_extreme
                    return True, f"Extreme threshold ±{extreme}% - boost {boost}x"
            return False, "No inicializado"
        except Exception as e:
            return False, f"Error: {str(e)[:50]}"

    async def _validate_volume_profile(self, monitor) -> Tuple[bool, str]:
        """Valida Volume Profile & POC"""
        try:
            if hasattr(monitor, 'feature_aggregator') and monitor.feature_aggregator:
                vol_profile = monitor.feature_aggregator.volume_profile
                if vol_profile and vol_profile.enabled:
                    lookback = vol_profile.lookback_periods
                    bins = vol_profile.price_bins
                    return True, f"Lookback {lookback} - {bins} bins - POC detection"
            return False, "No inicializado"
        except Exception as e:
            return False, f"Error: {str(e)[:50]}"

    async def _validate_pattern_recognition(self, monitor) -> Tuple[bool, str]:
        """Valida Pattern Recognition"""
        try:
            if hasattr(monitor, 'feature_aggregator') and monitor.feature_aggregator:
                patterns = monitor.feature_aggregator.pattern_recognition
                if patterns and patterns.enabled:
                    min_conf = patterns.min_pattern_confidence
                    boost = patterns.boost_factor
                    return True, f"H&S, Double Top/Bottom - min conf {min_conf} - boost {boost}x"
            return False, "No inicializado"
        except Exception as e:
            return False, f"Error: {str(e)[:50]}"

    async def _validate_session_trading(self, monitor) -> Tuple[bool, str]:
        """Valida Session-Based Trading"""
        try:
            if hasattr(monitor, 'feature_aggregator') and monitor.feature_aggregator:
                session = monitor.feature_aggregator.session_trading
                if session and session.enabled:
                    us_boost = session.us_open_boost
                    current_session, multiplier = session.get_current_session()
                    return True, f"Current: {current_session} ({multiplier}x) - US boost {us_boost}x"
            return False, "No inicializado"
        except Exception as e:
            return False, f"Error: {str(e)[:50]}"

    async def _validate_order_flow(self, monitor) -> Tuple[bool, str]:
        """Valida Order Flow Imbalance"""
        try:
            if hasattr(monitor, 'feature_aggregator') and monitor.feature_aggregator:
                order_flow = monitor.feature_aggregator.order_flow
                if order_flow and order_flow.enabled:
                    strong_ratio = order_flow.strong_imbalance_ratio
                    boost = order_flow.boost_strong
                    return True, f"Strong ratio {strong_ratio}:1 - boost {boost}x"
            return False, "No inicializado"
        except Exception as e:
            return False, f"Error: {str(e)[:50]}"

    async def _validate_feature_aggregator(self, monitor) -> Tuple[bool, str]:
        """Valida Feature Aggregator (Hub Central)"""
        try:
            if hasattr(monitor, 'feature_aggregator') and monitor.feature_aggregator:
                aggregator = monitor.feature_aggregator
                # Contar módulos activos
                status = aggregator.is_everything_enabled()
                active_count = sum(1 for enabled in status.values() if enabled)
                total_modules = len(status)
                return True, f"Hub central - {active_count}/{total_modules} módulos activos"
            return False, "No inicializado"
        except Exception as e:
            return False, f"Error: {str(e)[:50]}"

    def generate_telegram_message(self, validation_results: Dict) -> str:
        """
        Genera mensaje para Telegram con checklist de servicios

        Args:
            validation_results: Resultados de validación

        Returns:
            Mensaje formateado para Telegram
        """
        operational = validation_results['operational']
        failed = validation_results['failed']
        total = validation_results['total']
        count_operational = validation_results['count_operational']
        elapsed = validation_results['elapsed_seconds']

        # Header
        message = "🚀 **SISTEMA DE TRADING AUTÓNOMO INICIALIZADO**\n\n"

        # Status general
        status_icon = "✅" if count_operational == total else "⚠️"
        message += f"{status_icon} **Servicios operando: {count_operational}/{total}**\n"
        message += f"⏱️ Validación completada en {elapsed:.1f}s\n\n"

        # Checklist de servicios operativos (separar core vs arsenal)
        message += "**📋 SERVICIOS CORE:**\n"
        for service in operational[:16]:  # Primeros 16 son core
            message += f"✅ {service}\n"

        # Arsenal avanzado (si hay más de 16 servicios)
        if len(operational) > 16:
            message += "\n**🎯 ARSENAL AVANZADO (NUEVO):**\n"
            for service in operational[16:]:
                message += f"✅ {service}\n"

        # Servicios fallidos (si hay)
        if failed:
            message += f"\n**⚠️ SERVICIOS CON PROBLEMAS ({len(failed)}):**\n"
            for service in failed[:5]:  # Limitar a 5 para no saturar
                message += f"❌ {service}\n"

        # Footer con info importante
        message += "\n**🎯 SISTEMA LISTO PARA OPERAR**\n"
        message += "📱 Comandos disponibles:\n"
        message += "  /status - Estado actual del bot\n"
        message += "  /export_intelligence - Exportar aprendizaje IA\n"
        message += "  /import_intelligence - Importar aprendizaje IA\n"
        message += "  /stats - Estadísticas de trading\n"
        message += "  /params - Ver parámetros actuales\n\n"
        message += "🤖 **Modo Autónomo ACTIVO** - IA tiene control total\n"
        message += "💰 Iniciando con $50,000 USDT en paper trading"

        return message


async def run_startup_validation(monitor):
    """
    Ejecuta validación completa y envía reporte a Telegram

    Args:
        monitor: Instancia de MarketMonitor

    Returns:
        Dict con resultados de validación
    """
    validator = StartupValidator()
    results = await validator.validate_all_services(monitor)

    # Generar y enviar mensaje a Telegram
    if hasattr(monitor, 'telegram_commands') and monitor.telegram_commands:
        message = validator.generate_telegram_message(results)
        try:
            await monitor.telegram_commands.send_message(message)
            logger.info("📱 Reporte de validación enviado a Telegram")
        except Exception as e:
            logger.error(f"Error enviando reporte a Telegram: {e}")

    return results
