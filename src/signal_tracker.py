"""
Signal Tracker Module
Tracks trading signals and calculates accuracy
"""
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import logging
from config import config

logger = logging.getLogger(__name__)


def _sanitize_for_json(obj):
    """
    Convierte objetos no serializables (pandas Series, numpy, etc.) a tipos JSON válidos
    """
    import pandas as pd
    import numpy as np

    if isinstance(obj, (pd.Series, pd.DataFrame)):
        return obj.to_dict()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(item) for item in obj]
    else:
        return obj


class SignalTracker:
    """
    Tracks trading signals and their outcomes for accuracy calculation
    """

    def __init__(self):
        self.tracking_file = Path(config.TRACKING_FILE)
        self.profit_threshold = config.PROFIT_THRESHOLD
        self.signals = self._load_signals()

    def _load_signals(self) -> List[Dict]:
        """
        Load signals from tracking file

        Returns:
            List of signal dictionaries
        """
        if not self.tracking_file.exists():
            return []

        try:
            with open(self.tracking_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading signals: {e}")
            return []

    def _save_signals(self):
        """
        Save signals to tracking file
        """
        try:
            # Ensure directory exists
            self.tracking_file.parent.mkdir(parents=True, exist_ok=True)

            # 🔧 FIX: Convertir timestamps en KEYS del diccionario también
            def sanitize_dict_keys(obj):
                """Convierte keys de diccionarios que sean Timestamps a strings"""
                import pandas as pd

                if isinstance(obj, dict):
                    new_dict = {}
                    for k, v in obj.items():
                        # Convertir key si es Timestamp
                        if isinstance(k, pd.Timestamp):
                            k = k.isoformat()
                        # Recursivamente sanitizar el valor
                        new_dict[k] = sanitize_dict_keys(v)
                    return new_dict
                elif isinstance(obj, (list, tuple)):
                    return [sanitize_dict_keys(item) for item in obj]
                else:
                    return obj

            # Sanitizar datos (incluyendo keys)
            sanitized_signals = sanitize_dict_keys(_sanitize_for_json(self.signals))

            with open(self.tracking_file, 'w', encoding='utf-8') as f:
                json.dump(sanitized_signals, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Error saving signals: {e}", exc_info=True)

    def add_signal(self, pair: str, action: str, price: float, indicators: Dict, reasons: List[str]):
        """
        Add a new signal to track

        Args:
            pair: Trading pair
            action: BUY or SELL
            price: Price at signal time
            indicators: Technical indicators
            reasons: Reasons for the signal
        """
        # Sanitizar indicators para evitar pandas Series u objetos no serializables
        sanitized_indicators = _sanitize_for_json(indicators)
        sanitized_reasons = _sanitize_for_json(reasons)

        signal = {
            'id': len(self.signals) + 1,
            'pair': pair,
            'action': action,
            'signal_price': price,
            'signal_time': datetime.now().isoformat(),
            'indicators': sanitized_indicators,
            'reasons': sanitized_reasons,
            'status': 'pending',  # pending, success, failed
            'outcome_price': None,
            'outcome_time': None,
            'profit_percent': None,
            'checked': False
        }

        self.signals.append(signal)
        self._save_signals()
        logger.info(f"Signal tracked: {pair} {action} @ ${price:.2f}")

    def update_signal_outcome(self, signal: Dict, current_price: float) -> bool:
        """
        Update signal outcome based on current price

        Args:
            signal: Signal dictionary
            current_price: Current market price

        Returns:
            True if outcome was updated
        """
        if signal['status'] != 'pending':
            return False

        signal_price = signal['signal_price']
        action = signal['action']

        # Calculate profit percentage
        if action == 'BUY':
            profit_percent = ((current_price - signal_price) / signal_price) * 100
        else:  # SELL
            profit_percent = ((signal_price - current_price) / signal_price) * 100

        signal['profit_percent'] = round(profit_percent, 2)
        signal['outcome_price'] = current_price
        signal['outcome_time'] = datetime.now().isoformat()
        signal['checked'] = True

        # Determine if signal was successful
        if profit_percent >= self.profit_threshold:
            signal['status'] = 'success'
            logger.info(f"Signal SUCCESS: {signal['pair']} {action} - Profit: {profit_percent:.2f}%")
        elif profit_percent <= -self.profit_threshold:
            signal['status'] = 'failed'
            logger.info(f"Signal FAILED: {signal['pair']} {action} - Loss: {profit_percent:.2f}%")
        else:
            # Still pending, not enough movement
            signal['checked'] = False
            signal['outcome_price'] = None
            signal['outcome_time'] = None
            return False

        self._save_signals()
        return True

    def check_pending_signals(self, market_data: Dict[str, float]):
        """
        Check all pending signals against current market data

        Args:
            market_data: Dictionary mapping pairs to current prices
        """
        for signal in self.signals:
            if signal['status'] == 'pending' and signal['pair'] in market_data:
                current_price = market_data[signal['pair']]
                self.update_signal_outcome(signal, current_price)

    def get_today_signals(self) -> List[Dict]:
        """
        Get all signals from today

        Returns:
            List of today's signals
        """
        today = datetime.now().date()
        today_signals = []

        for signal in self.signals:
            signal_date = datetime.fromisoformat(signal['signal_time']).date()
            if signal_date == today:
                today_signals.append(signal)

        return today_signals

    def get_accuracy_stats(self, days: int = 1) -> Dict:
        """
        Calculate accuracy statistics

        Args:
            days: Number of days to look back

        Returns:
            Dictionary with accuracy statistics
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        relevant_signals = []

        for signal in self.signals:
            signal_time = datetime.fromisoformat(signal['signal_time'])
            if signal_time >= cutoff_date and signal['status'] != 'pending':
                relevant_signals.append(signal)

        if not relevant_signals:
            return {
                'total_signals': 0,
                'successful': 0,
                'failed': 0,
                'accuracy': 0.0,
                'avg_profit': 0.0,
                'best_signal': None,
                'worst_signal': None
            }

        successful = [s for s in relevant_signals if s['status'] == 'success']
        failed = [s for s in relevant_signals if s['status'] == 'failed']

        # Calculate average profit
        profits = [s['profit_percent'] for s in relevant_signals if s['profit_percent'] is not None]
        avg_profit = sum(profits) / len(profits) if profits else 0.0

        # Best and worst signals
        if relevant_signals:
            best = max(relevant_signals, key=lambda s: s['profit_percent'] or 0)
            worst = min(relevant_signals, key=lambda s: s['profit_percent'] or 0)
        else:
            best = worst = None

        accuracy = (len(successful) / len(relevant_signals) * 100) if relevant_signals else 0.0

        return {
            'total_signals': len(relevant_signals),
            'successful': len(successful),
            'failed': len(failed),
            'accuracy': round(accuracy, 2),
            'avg_profit': round(avg_profit, 2),
            'best_signal': best,
            'worst_signal': worst,
            'by_pair': self._get_stats_by_pair(relevant_signals),
            'by_action': self._get_stats_by_action(relevant_signals)
        }

    def _get_stats_by_pair(self, signals: List[Dict]) -> Dict:
        """
        Get statistics grouped by trading pair

        Args:
            signals: List of signals

        Returns:
            Dictionary with stats per pair
        """
        pairs = {}
        for signal in signals:
            pair = signal['pair']
            if pair not in pairs:
                pairs[pair] = {'total': 0, 'success': 0, 'failed': 0}

            pairs[pair]['total'] += 1
            if signal['status'] == 'success':
                pairs[pair]['success'] += 1
            elif signal['status'] == 'failed':
                pairs[pair]['failed'] += 1

        # Calculate accuracy for each pair
        for pair in pairs:
            total = pairs[pair]['total']
            success = pairs[pair]['success']
            pairs[pair]['accuracy'] = round((success / total * 100) if total > 0 else 0, 2)

        return pairs

    def _get_stats_by_action(self, signals: List[Dict]) -> Dict:
        """
        Get statistics grouped by action (BUY/SELL)

        Args:
            signals: List of signals

        Returns:
            Dictionary with stats per action
        """
        actions = {'BUY': {'total': 0, 'success': 0, 'failed': 0},
                   'SELL': {'total': 0, 'success': 0, 'failed': 0}}

        for signal in signals:
            action = signal['action']
            actions[action]['total'] += 1
            if signal['status'] == 'success':
                actions[action]['success'] += 1
            elif signal['status'] == 'failed':
                actions[action]['failed'] += 1

        # Calculate accuracy for each action
        for action in actions:
            total = actions[action]['total']
            success = actions[action]['success']
            actions[action]['accuracy'] = round((success / total * 100) if total > 0 else 0, 2)

        return actions

    def clear_old_signals(self, days: int = 30):
        """
        Remove signals older than specified days

        Args:
            days: Number of days to keep
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        self.signals = [
            s for s in self.signals
            if datetime.fromisoformat(s['signal_time']) >= cutoff_date
        ]
        self._save_signals()
        logger.info(f"Cleared signals older than {days} days")
