"""
Backtest Analyzer - Analiza resultados de backtest
Proporciona métricas detalladas y visualizaciones de performance
"""
import pandas as pd
import logging
from typing import Dict, List
from collections import defaultdict

logger = logging.getLogger(__name__)


class BacktestAnalyzer:
    """
    Analiza resultados de backtest y genera métricas detalladas
    """

    def __init__(self, backtest_results: List[Dict]):
        self.results = backtest_results
        self.df = pd.DataFrame(backtest_results) if backtest_results else pd.DataFrame()

    def get_overall_metrics(self) -> Dict:
        """Métricas generales del backtest"""
        if self.df.empty:
            return {}

        total = len(self.df)
        wins = len(self.df[self.df['result'] == 'WIN'])
        losses = len(self.df[self.df['result'] == 'LOSS'])
        win_rate = (wins / total) * 100 if total > 0 else 0

        winning_trades = self.df[self.df['result'] == 'WIN']
        losing_trades = self.df[self.df['result'] == 'LOSS']

        avg_win = winning_trades['pnl_pct'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl_pct'].mean() if len(losing_trades) > 0 else 0

        total_pnl = self.df['pnl_pct'].sum()
        avg_pnl = self.df['pnl_pct'].mean()

        # Profit factor
        total_gains = winning_trades['pnl_pct'].sum() if len(winning_trades) > 0 else 0
        total_losses = abs(losing_trades['pnl_pct'].sum()) if len(losing_trades) > 0 else 0
        profit_factor = total_gains / total_losses if total_losses > 0 else 0

        # Expectancy
        expectancy = (win_rate / 100) * avg_win + ((100 - win_rate) / 100) * avg_loss

        return {
            'total_signals': total,
            'wins': wins,
            'losses': losses,
            'win_rate': round(win_rate, 2),
            'avg_win_pct': round(avg_win, 2),
            'avg_loss_pct': round(avg_loss, 2),
            'total_pnl_pct': round(total_pnl, 2),
            'avg_pnl_pct': round(avg_pnl, 2),
            'profit_factor': round(profit_factor, 2),
            'expectancy': round(expectancy, 2)
        }

    def get_metrics_by_pair(self) -> Dict[str, Dict]:
        """Métricas por par"""
        if self.df.empty:
            return {}

        metrics_by_pair = {}

        for pair in self.df['pair'].unique():
            pair_df = self.df[self.df['pair'] == pair]
            total = len(pair_df)
            wins = len(pair_df[pair_df['result'] == 'WIN'])
            win_rate = (wins / total) * 100 if total > 0 else 0
            total_pnl = pair_df['pnl_pct'].sum()

            metrics_by_pair[pair] = {
                'total_signals': total,
                'wins': wins,
                'losses': total - wins,
                'win_rate': round(win_rate, 2),
                'total_pnl_pct': round(total_pnl, 2)
            }

        return metrics_by_pair

    def get_metrics_by_signal_type(self) -> Dict[str, Dict]:
        """Métricas por tipo de señal (conservative vs flash)"""
        if self.df.empty or 'signal_type' not in self.df.columns:
            return {}

        metrics_by_type = {}

        for signal_type in self.df['signal_type'].unique():
            type_df = self.df[self.df['signal_type'] == signal_type]
            total = len(type_df)
            wins = len(type_df[type_df['result'] == 'WIN'])
            win_rate = (wins / total) * 100 if total > 0 else 0
            total_pnl = type_df['pnl_pct'].sum()
            avg_pnl = type_df['pnl_pct'].mean()

            metrics_by_type[signal_type] = {
                'total_signals': total,
                'wins': wins,
                'losses': total - wins,
                'win_rate': round(win_rate, 2),
                'total_pnl_pct': round(total_pnl, 2),
                'avg_pnl_pct': round(avg_pnl, 2)
            }

        return metrics_by_type

    def get_metrics_by_score_range(self) -> Dict[str, Dict]:
        """Métricas por rango de score"""
        if self.df.empty or 'score' not in self.df.columns:
            return {}

        # Definir rangos de score
        ranges = [
            (0, 5, '0-5'),
            (5, 6, '5-6'),
            (6, 7, '6-7'),
            (7, 8, '7-8'),
            (8, 9, '8-9'),
            (9, 10, '9-10')
        ]

        metrics_by_range = {}

        for min_score, max_score, label in ranges:
            range_df = self.df[(self.df['score'] >= min_score) & (self.df['score'] < max_score)]

            if len(range_df) == 0:
                continue

            total = len(range_df)
            wins = len(range_df[range_df['result'] == 'WIN'])
            win_rate = (wins / total) * 100 if total > 0 else 0
            avg_pnl = range_df['pnl_pct'].mean()

            metrics_by_range[label] = {
                'total_signals': total,
                'wins': wins,
                'win_rate': round(win_rate, 2),
                'avg_pnl_pct': round(avg_pnl, 2)
            }

        return metrics_by_range

    def get_time_analysis(self) -> Dict:
        """Análisis temporal de señales"""
        if self.df.empty or 'timestamp' not in self.df.columns:
            return {}

        df_with_time = self.df.copy()
        df_with_time['timestamp'] = pd.to_datetime(df_with_time['timestamp'])
        df_with_time['month'] = df_with_time['timestamp'].dt.to_period('M')
        df_with_time['hour'] = df_with_time['timestamp'].dt.hour
        df_with_time['day_of_week'] = df_with_time['timestamp'].dt.dayofweek

        # Por mes
        monthly_stats = []
        for month in df_with_time['month'].unique():
            month_df = df_with_time[df_with_time['month'] == month]
            wins = len(month_df[month_df['result'] == 'WIN'])
            total = len(month_df)
            win_rate = (wins / total) * 100 if total > 0 else 0

            monthly_stats.append({
                'month': str(month),
                'total_signals': total,
                'win_rate': round(win_rate, 2)
            })

        # Por día de semana
        days = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
        day_stats = []
        for day_idx, day_name in enumerate(days):
            day_df = df_with_time[df_with_time['day_of_week'] == day_idx]
            if len(day_df) == 0:
                continue

            wins = len(day_df[day_df['result'] == 'WIN'])
            total = len(day_df)
            win_rate = (wins / total) * 100 if total > 0 else 0

            day_stats.append({
                'day': day_name,
                'total_signals': total,
                'win_rate': round(win_rate, 2)
            })

        return {
            'monthly': monthly_stats,
            'by_day_of_week': day_stats
        }

    def get_best_and_worst_pairs(self, top_n: int = 5) -> Dict:
        """Mejores y peores pares"""
        metrics_by_pair = self.get_metrics_by_pair()

        if not metrics_by_pair:
            return {}

        # Ordenar por win rate
        sorted_by_wr = sorted(
            metrics_by_pair.items(),
            key=lambda x: x[1]['win_rate'],
            reverse=True
        )

        # Ordenar por total P&L
        sorted_by_pnl = sorted(
            metrics_by_pair.items(),
            key=lambda x: x[1]['total_pnl_pct'],
            reverse=True
        )

        return {
            'best_by_win_rate': [
                {'pair': pair, **metrics}
                for pair, metrics in sorted_by_wr[:top_n]
            ],
            'worst_by_win_rate': [
                {'pair': pair, **metrics}
                for pair, metrics in sorted_by_wr[-top_n:]
            ],
            'best_by_pnl': [
                {'pair': pair, **metrics}
                for pair, metrics in sorted_by_pnl[:top_n]
            ],
            'worst_by_pnl': [
                {'pair': pair, **metrics}
                for pair, metrics in sorted_by_pnl[-top_n:]
            ]
        }

    def print_summary(self):
        """Imprime resumen completo del backtest"""
        print("\n" + "="*60)
        print("📊 BACKTEST ANALYSIS SUMMARY")
        print("="*60)

        # Overall metrics
        overall = self.get_overall_metrics()
        if overall:
            print("\n📈 OVERALL PERFORMANCE")
            print(f"Total Signals: {overall['total_signals']}")
            print(f"Wins: {overall['wins']} | Losses: {overall['losses']}")
            print(f"Win Rate: {overall['win_rate']:.2f}%")
            print(f"Avg Win: {overall['avg_win_pct']:.2f}% | Avg Loss: {overall['avg_loss_pct']:.2f}%")
            print(f"Total P&L: {overall['total_pnl_pct']:.2f}%")
            print(f"Avg P&L per Trade: {overall['avg_pnl_pct']:.2f}%")
            print(f"Profit Factor: {overall['profit_factor']:.2f}")
            print(f"Expectancy: {overall['expectancy']:.2f}%")

        # By signal type
        by_type = self.get_metrics_by_signal_type()
        if by_type:
            print("\n📊 BY SIGNAL TYPE")
            for sig_type, metrics in by_type.items():
                print(f"\n{sig_type.upper()}:")
                print(f"  Signals: {metrics['total_signals']}")
                print(f"  Win Rate: {metrics['win_rate']:.2f}%")
                print(f"  Avg P&L: {metrics['avg_pnl_pct']:.2f}%")

        # By score range
        by_score = self.get_metrics_by_score_range()
        if by_score:
            print("\n📊 BY SCORE RANGE")
            for range_label, metrics in by_score.items():
                print(f"Score {range_label}: WR {metrics['win_rate']:.1f}% | "
                      f"Signals: {metrics['total_signals']} | "
                      f"Avg P&L: {metrics['avg_pnl_pct']:.2f}%")

        # Best/Worst pairs
        best_worst = self.get_best_and_worst_pairs(top_n=5)
        if best_worst:
            print("\n🏆 TOP 5 PAIRS (by win rate)")
            for item in best_worst['best_by_win_rate']:
                print(f"  {item['pair']}: {item['win_rate']:.1f}% WR | "
                      f"{item['total_signals']} signals | P&L: {item['total_pnl_pct']:.2f}%")

            print("\n⚠️ BOTTOM 5 PAIRS (by win rate)")
            for item in best_worst['worst_by_win_rate']:
                print(f"  {item['pair']}: {item['win_rate']:.1f}% WR | "
                      f"{item['total_signals']} signals | P&L: {item['total_pnl_pct']:.2f}%")

        print("\n" + "="*60 + "\n")
