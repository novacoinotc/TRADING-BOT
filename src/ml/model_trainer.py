"""
Model Trainer - Entrena modelos ML (XGBoost) para predecir señales ganadoras
"""
import logging
import pickle
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost no instalado. Instalar con: pip install xgboost scikit-learn")


class ModelTrainer:
    """
    Entrena modelos ML para predecir si una señal será ganadora
    Usa XGBoost para clasificación binaria (WIN/LOSS)
    """

    def __init__(self, min_samples_for_training: int = 50):
        """
        Args:
            min_samples_for_training: Mínimo de trades para entrenar modelo
        """
        self.min_samples = min_samples_for_training
        self.model = None
        self.model_version = "v1.0"
        self.feature_names = []

        # Directorios
        self.models_dir = Path('data/models')
        self.training_data_dir = Path('data/training')
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.training_data_dir.mkdir(parents=True, exist_ok=True)

        # Archivos
        self.model_file = self.models_dir / 'xgboost_model.pkl'
        self.metadata_file = self.models_dir / 'model_metadata.json'

        # Cargar modelo si existe
        self._load_model()

    def prepare_training_data(self, trades_history: List[Dict],
                            features_list: List[Dict]) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepara datos para entrenamiento

        Args:
            trades_history: Lista de trades cerrados
            features_list: Lista de features correspondientes

        Returns:
            (X, y) donde X son features y y son labels (WIN=1, LOSS=0)
        """
        if len(trades_history) != len(features_list):
            raise ValueError("trades_history y features_list deben tener el mismo tamaño")

        # Crear labels: 1 si ganó, 0 si perdió
        labels = []
        valid_features = []

        for trade, features in zip(trades_history, features_list):
            pnl = trade.get('pnl', 0)

            # Clasificar como ganador/perdedor
            if pnl > 0:
                labels.append(1)  # WIN
                valid_features.append(features)
            elif pnl < 0:
                labels.append(0)  # LOSS
                valid_features.append(features)
            # Si pnl == 0, ignorar (empate)

        # Convertir a DataFrames
        X = pd.DataFrame(valid_features)
        y = pd.Series(labels)

        logger.info(f"📊 Datos preparados: {len(X)} samples | WIN: {sum(labels)} | LOSS: {len(labels) - sum(labels)}")

        return X, y

    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2,
              sample_weights: Optional[np.ndarray] = None) -> Dict:
        """
        Entrena modelo XGBoost

        Args:
            X: Features (DataFrame)
            y: Labels (Series) - 1=WIN, 0=LOSS
            test_size: Porcentaje de datos para test
            sample_weights: Pesos para cada muestra (opcional, para temporal weighting)

        Returns:
            Dict con métricas de entrenamiento
        """
        if not XGBOOST_AVAILABLE:
            logger.error("XGBoost no disponible. No se puede entrenar modelo.")
            return {}

        if len(X) < self.min_samples:
            logger.warning(f"Insuficientes muestras para entrenar: {len(X)} < {self.min_samples}")
            return {}

        # Split train/test (también dividir sample_weights si se proporcionan)
        if sample_weights is not None:
            X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
                X, y, sample_weights, test_size=test_size, random_state=42, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            weights_train = None
            weights_test = None

        logger.info(f"🧠 Entrenando modelo ML...")
        logger.info(f"   Train: {len(X_train)} samples | Test: {len(X_test)} samples")

        # Configurar XGBoost
        # Si model_config ya está definido (por initial_trainer), usarlo
        if hasattr(self, 'model_config') and self.model_config:
            self.model = XGBClassifier(**self.model_config)
        else:
            # Configuración por defecto
            self.model = XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )

        # Entrenar (con sample_weights si están disponibles)
        if weights_train is not None:
            self.model.fit(X_train, y_train, sample_weight=weights_train)
        else:
            self.model.fit(X_train, y_train)

        # Evaluar
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)

        metrics = {
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'train_precision': precision_score(y_train, y_pred_train, zero_division=0),
            'test_precision': precision_score(y_test, y_pred_test, zero_division=0),
            'train_recall': recall_score(y_train, y_pred_train, zero_division=0),
            'test_recall': recall_score(y_test, y_pred_test, zero_division=0),
            'train_f1': f1_score(y_train, y_pred_train, zero_division=0),
            'test_f1': f1_score(y_test, y_pred_test, zero_division=0),
            'samples_total': len(X),
            'samples_train': len(X_train),
            'samples_test': len(X_test),
            'trained_at': datetime.now().isoformat()
        }

        # Guardar nombres de features
        self.feature_names = list(X.columns)

        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance_dict = dict(zip(self.feature_names, self.model.feature_importances_))
            # Top 10 features más importantes
            top_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
            metrics['top_features'] = [
                {'name': name, 'importance': float(importance)} for name, importance in top_features
            ]

        logger.info(f"✅ Modelo entrenado exitosamente!")
        logger.info(f"   Train Accuracy: {metrics['train_accuracy']:.3f} | Test Accuracy: {metrics['test_accuracy']:.3f}")
        logger.info(f"   Test Precision: {metrics['test_precision']:.3f} | Test Recall: {metrics['test_recall']:.3f}")

        # Guardar modelo
        self._save_model(metrics)

        return metrics

    def predict(self, features: Dict) -> Dict:
        """
        Predice si una señal será ganadora

        Args:
            features: Dict con features

        Returns:
            Dict con predicción y probabilidad
        """
        if not self.model or not XGBOOST_AVAILABLE:
            # Si no hay modelo, retornar neutral
            return {
                'prediction': 'UNKNOWN',
                'win_probability': 0.5,
                'confidence': 0.0,
                'model_available': False
            }

        try:
            # Convertir features a DataFrame
            X = pd.DataFrame([features])[self.feature_names]

            # Predecir
            prediction = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]

            win_probability = probabilities[1]  # Probabilidad de WIN
            loss_probability = probabilities[0]  # Probabilidad de LOSS

            confidence = max(win_probability, loss_probability) * 100

            return {
                'prediction': 'WIN' if prediction == 1 else 'LOSS',
                'win_probability': float(win_probability),
                'loss_probability': float(loss_probability),
                'confidence': float(confidence),
                'model_available': True
            }

        except Exception as e:
            logger.error(f"Error en predicción ML: {e}")
            return {
                'prediction': 'ERROR',
                'win_probability': 0.5,
                'confidence': 0.0,
                'model_available': False
            }

    def should_retrain(self, new_trades_count: int, last_trained_samples: int) -> bool:
        """
        Determina si se debe reentrenar el modelo

        Args:
            new_trades_count: Número de trades nuevos
            last_trained_samples: Trades con los que se entrenó última vez

        Returns:
            True si se debe reentrenar
        """
        # Reentrenar si:
        # 1. Hay 20+ trades nuevos
        if new_trades_count - last_trained_samples >= 20:
            return True

        # 2. Han pasado muchos trades (50% más)
        if last_trained_samples > 0 and new_trades_count >= last_trained_samples * 1.5:
            return True

        return False

    def _save_model(self, metrics: Dict):
        """Guarda modelo y metadata"""
        if not self.model:
            return

        try:
            # Guardar modelo
            with open(self.model_file, 'wb') as f:
                pickle.dump(self.model, f)

            # Guardar metadata
            metadata = {
                'version': self.model_version,
                'feature_names': self.feature_names,
                'metrics': metrics,
                'saved_at': datetime.now().isoformat()
            }

            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"💾 Modelo guardado: {self.model_file}")

        except Exception as e:
            logger.error(f"Error guardando modelo: {e}")

    def _load_model(self):
        """Carga modelo si existe"""
        if not self.model_file.exists() or not XGBOOST_AVAILABLE:
            return

        try:
            # Cargar modelo
            with open(self.model_file, 'rb') as f:
                self.model = pickle.load(f)

            # Cargar metadata
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)

                self.feature_names = metadata.get('feature_names', [])
                metrics = metadata.get('metrics', {})

                logger.info(f"🧠 Modelo ML cargado")
                logger.info(f"   Test Accuracy: {metrics.get('test_accuracy', 0):.3f}")
                logger.info(f"   Samples: {metrics.get('samples_total', 0)}")

        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            self.model = None

    def get_model_info(self) -> Dict:
        """Retorna información del modelo actual"""
        if not self.model or not self.metadata_file.exists():
            return {
                'available': False,
                'message': 'No trained model available'
            }

        try:
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)

            return {
                'available': True,
                **metadata
            }

        except:
            return {
                'available': False,
                'message': 'Error loading model metadata'
            }
