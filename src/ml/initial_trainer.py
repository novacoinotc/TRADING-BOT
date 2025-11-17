"""
Initial Trainer - Pre-entrena modelo con datos históricos
Incluye protecciones anti-overfitting para que aprenda patrones generales
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from pathlib import Path

from src.ml.model_trainer import ModelTrainer
from src.ml.backtest_analyzer import BacktestAnalyzer

logger = logging.getLogger(__name__)


class InitialTrainer:
    """
    Pre-entrena modelo ML con datos históricos usando técnicas anti-overfitting

    Estrategias anti-overfitting:
    1. Walk-forward validation (entrenar con pasado, validar con reciente)
    2. Out-of-sample testing (reservar últimos meses)
    3. Regularización en XGBoost (max_depth, min_child_weight)
    4. Temporal weighting (más peso a datos recientes)
    5. Feature importance analysis (eliminar features ruidosas)
    """

    def __init__(
        self,
        backtest_results: List[Dict],
        temporal_weight_recent: float = 2.0,  # Peso 2x para datos recientes
        oos_months: int = 2  # Meses para out-of-sample testing
    ):
        self.backtest_results = backtest_results
        self.temporal_weight_recent = temporal_weight_recent
        self.oos_months = oos_months
        self.model_trainer = ModelTrainer(min_samples_for_training=100)

    def train_with_validation(self) -> Dict:
        """
        Entrena modelo con validación robusta

        Returns:
            Dict con métricas y modelo entrenado
        """
        if len(self.backtest_results) < 200:
            logger.error(f"Insuficientes señales para entrenar: {len(self.backtest_results)} (mínimo 200)")
            return {'success': False, 'reason': 'insufficient_data'}

        logger.info("🧠 Iniciando entrenamiento con datos históricos...")
        logger.info(f"   Total señales: {len(self.backtest_results)}")

        # 1. Convertir a DataFrame
        df = pd.DataFrame(self.backtest_results)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')

        logger.info(f"   Periodo: {df['timestamp'].min().date()} a {df['timestamp'].max().date()}")

        # 2. Walk-forward validation
        logger.info("\n📊 Walk-forward validation...")
        wf_results = self._walk_forward_validation(df)

        if not wf_results['success']:
            return wf_results

        # 3. Preparar datos completos con pesos temporales
        logger.info("\n🎯 Entrenando modelo final...")
        X, y, sample_weights = self._prepare_training_data_with_weights(df)

        # 4. Split out-of-sample (últimos 2 meses para testing final)
        oos_date = df['timestamp'].max() - timedelta(days=self.oos_months * 30)
        is_oos = df['timestamp'] >= oos_date

        X_train = X[~is_oos]
        y_train = y[~is_oos]
        weights_train = sample_weights[~is_oos]

        X_test = X[is_oos]
        y_test = y[is_oos]

        logger.info(f"   In-sample: {len(X_train)} señales (entrenamiento)")
        logger.info(f"   Out-of-sample: {len(X_test)} señales (validación final)")

        # 5. Entrenar modelo final con regularización
        self.model_trainer.model_config = {
            'n_estimators': 100,
            'max_depth': 5,  # Limitado para evitar overfitting
            'learning_rate': 0.05,  # Más lento = más robusto
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 5,  # Requiere más samples por hoja
            'gamma': 0.1,  # Regularización
            'reg_alpha': 0.1,  # L1 regularization
            'reg_lambda': 1.0  # L2 regularization
        }

        metrics = self.model_trainer.train(
            X=pd.concat([X_train, X_test]),
            y=pd.concat([y_train, y_test]),
            sample_weights=np.concatenate([weights_train, np.ones(len(y_test))]),
            test_size=len(X_test) / len(X)  # Usar out-of-sample como test
        )

        if not metrics:
            return {'success': False, 'reason': 'training_failed'}

        # 6. Feature importance
        feature_importance = self.model_trainer.get_feature_importance(top_n=15)

        logger.info("\n🎯 Modelo entrenado!")
        logger.info(f"   In-sample Accuracy: {wf_results['avg_accuracy']:.3f}")
        logger.info(f"   Out-of-sample Accuracy: {metrics['test_accuracy']:.3f}")
        logger.info(f"   Out-of-sample Precision: {metrics['test_precision']:.3f}")
        logger.info(f"   Out-of-sample Recall: {metrics['test_recall']:.3f}")
        logger.info(f"   Out-of-sample F1: {metrics['test_f1']:.3f}")

        logger.info("\n🔝 Top 10 Features más importantes:")
        for i, (feature, importance) in enumerate(feature_importance[:10], 1):
            logger.info(f"   {i}. {feature}: {importance:.4f}")

        # 7. Analizar resultados
        analyzer = BacktestAnalyzer(self.backtest_results)
        backtest_stats = analyzer.get_overall_metrics()

        return {
            'success': True,
            'model': self.model_trainer.model,
            'metrics': metrics,
            'walk_forward_results': wf_results,
            'backtest_stats': backtest_stats,
            'feature_importance': feature_importance,
            'total_samples': len(self.backtest_results),
            'oos_samples': len(X_test),
            'training_date': datetime.now().isoformat()
        }

    def _walk_forward_validation(self, df: pd.DataFrame) -> Dict:
        """
        Walk-forward validation: entrenar con pasado, predecir futuro

        Divide datos en 5 periodos:
        - Entrena con periodo 1-3, predice 4
        - Entrena con periodo 2-4, predice 5
        Etc.
        """
        # Dividir en 5 periodos
        df_sorted = df.sort_values('timestamp')
        n_folds = 5
        fold_size = len(df_sorted) // n_folds

        accuracies = []
        precisions = []

        for i in range(2, n_folds):  # Empezar en fold 2 para tener suficiente training data
            # Training: todos los folds hasta i
            train_end_idx = (i + 1) * fold_size
            train_df = df_sorted.iloc[:train_end_idx]

            # Testing: siguiente fold
            test_start_idx = train_end_idx
            test_end_idx = min(test_start_idx + fold_size, len(df_sorted))
            test_df = df_sorted.iloc[test_start_idx:test_end_idx]

            if len(train_df) < 100 or len(test_df) < 20:
                continue

            # Preparar datos
            X_train, y_train, _ = self._prepare_training_data_with_weights(train_df)
            X_test, y_test, _ = self._prepare_training_data_with_weights(test_df)

            # Entrenar modelo temporal
            temp_trainer = ModelTrainer()
            temp_trainer.model_config = {
                'n_estimators': 50,
                'max_depth': 4,
                'learning_rate': 0.1
            }

            from xgboost import XGBClassifier
            temp_model = XGBClassifier(**temp_trainer.model_config)
            temp_model.fit(X_train, y_train)

            # Predecir en test
            y_pred = temp_model.predict(X_test)

            # Calcular métricas
            from sklearn.metrics import accuracy_score, precision_score

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)

            accuracies.append(accuracy)
            precisions.append(precision)

            train_period = f"{train_df['timestamp'].min().date()} - {train_df['timestamp'].max().date()}"
            test_period = f"{test_df['timestamp'].min().date()} - {test_df['timestamp'].max().date()}"

            logger.info(f"   Fold {i}: Train [{train_period}] → Test [{test_period}]")
            logger.info(f"           Accuracy: {accuracy:.3f} | Precision: {precision:.3f}")

        if not accuracies:
            return {'success': False, 'reason': 'walk_forward_failed'}

        avg_accuracy = np.mean(accuracies)
        avg_precision = np.mean(precisions)
        std_accuracy = np.std(accuracies)

        logger.info(f"\n   Walk-Forward Results:")
        logger.info(f"   Avg Accuracy: {avg_accuracy:.3f} (±{std_accuracy:.3f})")
        logger.info(f"   Avg Precision: {avg_precision:.3f}")

        # Verificar que no hay overfitting severo
        if std_accuracy > 0.15:
            logger.warning(f"   ⚠️ Alta varianza en accuracy ({std_accuracy:.3f}) - posible overfitting")

        return {
            'success': True,
            'avg_accuracy': avg_accuracy,
            'avg_precision': avg_precision,
            'std_accuracy': std_accuracy,
            'n_folds': len(accuracies)
        }

    def _prepare_training_data_with_weights(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
        """
        Prepara datos con pesos temporales (más peso a datos recientes)

        Args:
            df: DataFrame con backtest results

        Returns:
            Tuple de (X, y, sample_weights)
        """
        # Extraer features y target
        features_list = []
        targets = []

        for _, row in df.iterrows():
            features_list.append(row['features'])
            targets.append(1 if row['result'] == 'WIN' else 0)

        X = pd.DataFrame(features_list)
        y = pd.Series(targets)

        # CONVERSIÓN CRÍTICA: Convertir columnas categóricas a numéricos
        # Usar el método del ModelTrainer para mantener consistencia
        X = self.model_trainer._convert_categorical_to_numeric(X)

        # Calcular pesos temporales
        timestamps = pd.to_datetime(df['timestamp'])
        days_ago = (timestamps.max() - timestamps).dt.days

        # Peso exponencial: datos recientes pesan más
        # Últimos 6 meses: peso 2x
        # Más de 6 meses: peso decrece exponencialmente
        sample_weights = np.where(
            days_ago <= 180,  # 6 meses
            self.temporal_weight_recent,
            np.exp(-days_ago / 365)  # Decae exponencialmente
        )

        # Normalizar pesos
        sample_weights = sample_weights / sample_weights.sum() * len(sample_weights)

        return X, y, sample_weights

    def save_model(self, filepath: str = 'data/models/xgboost_pretrained.pkl'):
        """Guarda modelo pre-entrenado"""
        self.model_trainer.save_model(filepath)
        logger.info(f"💾 Modelo pre-entrenado guardado: {filepath}")

    def get_training_report(self, training_result: Dict) -> str:
        """Genera reporte de entrenamiento"""
        if not training_result.get('success'):
            return f"❌ Entrenamiento falló: {training_result.get('reason', 'unknown')}"

        metrics = training_result['metrics']
        wf = training_result['walk_forward_results']
        backtest = training_result['backtest_stats']

        report = f"""
🧠 **HISTORICAL TRAINING REPORT**

📊 **Backtest Statistics**
Total Signals: {backtest['total_signals']}
Win Rate: {backtest['win_rate']:.2f}%
Total P&L: {backtest['total_pnl_pct']:.2f}%
Profit Factor: {backtest['profit_factor']:.2f}

🎯 **Walk-Forward Validation** (robustez)
Avg Accuracy: {wf['avg_accuracy']:.3f} (±{wf['std_accuracy']:.3f})
Avg Precision: {wf['avg_precision']:.3f}
Folds: {wf['n_folds']}

📈 **Out-of-Sample Testing** (últimos {self.oos_months} meses)
Test Samples: {training_result['oos_samples']}
Test Accuracy: {metrics['test_accuracy']:.3f}
Test Precision: {metrics['test_precision']:.3f}
Test Recall: {metrics['test_recall']:.3f}
Test F1: {metrics['test_f1']:.3f}

✅ **Modelo Listo para Producción**
Total Samples: {training_result['total_samples']}
Training Date: {training_result['training_date']}

⚠️ **Anti-Overfitting Protections Applied**
✅ Walk-forward validation
✅ Out-of-sample testing
✅ XGBoost regularization (max_depth=5, min_child_weight=5)
✅ Temporal weighting (recent data: {self.temporal_weight_recent}x weight)
✅ Feature importance analysis
"""

        return report.strip()
