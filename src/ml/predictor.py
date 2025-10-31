"""
ML Predictor - Servicio de predicciones en tiempo real
Usa modelos entrenados para predecir probabilidad de éxito de señales
"""
import logging
from typing import Dict, Optional
from src.ml.model_trainer import ModelTrainer
from src.ml.feature_engineer import FeatureEngineer

logger = logging.getLogger(__name__)


class MLPredictor:
    """
    Predictor de señales ganadoras usando ML
    - Convierte indicadores a features
    - Usa modelo XGBoost para predecir WIN/LOSS
    - Retorna probabilidad de ganancia
    """

    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer(min_samples_for_training=50)
        self.enabled = True

        # Verificar si hay modelo disponible
        model_info = self.model_trainer.get_model_info()
        if model_info.get('available'):
            logger.info("🧠 ML Predictor inicializado con modelo entrenado")
            logger.info(f"   Test Accuracy: {model_info.get('metrics', {}).get('test_accuracy', 0):.3f}")
        else:
            logger.info("🧠 ML Predictor inicializado (sin modelo entrenado aún)")

    def predict(self, indicators: Dict, signals: Dict, mtf_indicators: Dict = None) -> Dict:
        """
        Predice probabilidad de éxito de una señal

        Args:
            indicators: Dict con indicadores técnicos
            signals: Dict con señales generadas
            mtf_indicators: Indicadores multi-timeframe (opcional)

        Returns:
            Dict con predicción y probabilidad
        """
        if not self.enabled:
            return self._neutral_prediction("Predictor deshabilitado")

        try:
            # Crear features
            features = self.feature_engineer.create_features(
                indicators=indicators,
                signals=signals,
                mtf_indicators=mtf_indicators
            )

            # Predecir
            prediction_result = self.model_trainer.predict(features)

            # Enriquecer resultado
            prediction_result['features_count'] = len(features)

            if prediction_result.get('model_available'):
                logger.debug(
                    f"ML Prediction: {prediction_result['prediction']} | "
                    f"Win Prob: {prediction_result['win_probability']:.2%} | "
                    f"Confidence: {prediction_result['confidence']:.1f}%"
                )

            return prediction_result

        except Exception as e:
            logger.error(f"Error en predicción ML: {e}")
            return self._neutral_prediction(f"Error: {str(e)}")

    def enhance_signal(self, signal: Dict, indicators: Dict, mtf_indicators: Dict = None) -> Dict:
        """
        Mejora señal con predicción ML

        Args:
            signal: Señal original
            indicators: Indicadores técnicos
            mtf_indicators: Indicadores MTF

        Returns:
            Señal mejorada con datos ML
        """
        # Hacer predicción
        ml_prediction = self.predict(indicators, signal, mtf_indicators)

        # Agregar datos ML a la señal
        enhanced_signal = signal.copy()
        enhanced_signal['ml'] = {
            'prediction': ml_prediction.get('prediction', 'UNKNOWN'),
            'win_probability': ml_prediction.get('win_probability', 0.5),
            'loss_probability': ml_prediction.get('loss_probability', 0.5),
            'ml_confidence': ml_prediction.get('confidence', 0.0),
            'model_available': ml_prediction.get('model_available', False)
        }

        # Ajustar confianza de señal usando ML si está disponible
        if ml_prediction.get('model_available'):
            original_confidence = signal.get('confidence', 50)
            ml_confidence = ml_prediction.get('confidence', 50)

            # Combinar confianza original con ML (70% original, 30% ML)
            combined_confidence = (original_confidence * 0.7) + (ml_confidence * 0.3)
            enhanced_signal['ml_adjusted_confidence'] = round(combined_confidence, 1)

            # Si ML predice LOSS con alta confianza, reducir score
            if ml_prediction['prediction'] == 'LOSS' and ml_confidence > 70:
                enhanced_signal['ml_warning'] = True
                logger.warning(
                    f"⚠️ ML Warning: Modelo predice LOSS con {ml_confidence:.1f}% confianza"
                )

        return enhanced_signal

    def should_trade(self, enhanced_signal: Dict, min_ml_confidence: float = 60.0) -> bool:
        """
        Determina si se debe ejecutar trade basado en predicción ML

        Args:
            enhanced_signal: Señal con datos ML
            min_ml_confidence: Confianza mínima requerida (%)

        Returns:
            True si se debe tradear
        """
        ml_data = enhanced_signal.get('ml', {})

        # Si no hay modelo, permitir trade (usa señales tradicionales)
        if not ml_data.get('model_available'):
            return True

        # Si ML predice WIN con alta confianza, permitir
        if ml_data['prediction'] == 'WIN' and ml_data['ml_confidence'] >= min_ml_confidence:
            return True

        # Si ML predice LOSS con alta confianza, bloquear
        if ml_data['prediction'] == 'LOSS' and ml_data['ml_confidence'] >= 70:
            logger.info(f"🚫 Trade bloqueado por ML (LOSS prediction con {ml_data['ml_confidence']:.1f}%)")
            return False

        # En otros casos, permitir (confianza intermedia o predicción incierta)
        return True

    def get_features(self, indicators: Dict, signals: Dict, mtf_indicators: Dict = None) -> Dict:
        """
        Retorna features creadas para debugging/análisis

        Args:
            indicators: Indicadores técnicos
            signals: Señales
            mtf_indicators: Indicadores MTF

        Returns:
            Dict con features
        """
        return self.feature_engineer.create_features(
            indicators=indicators,
            signals=signals,
            mtf_indicators=mtf_indicators
        )

    def _neutral_prediction(self, reason: str) -> Dict:
        """Retorna predicción neutral cuando no hay modelo"""
        return {
            'prediction': 'UNKNOWN',
            'win_probability': 0.5,
            'loss_probability': 0.5,
            'confidence': 0.0,
            'model_available': False,
            'reason': reason
        }

    def enable(self):
        """Habilita predictor ML"""
        self.enabled = True
        logger.info("✅ ML Predictor habilitado")

    def disable(self):
        """Deshabilita predictor ML"""
        self.enabled = False
        logger.info("❌ ML Predictor deshabilitado")

    def get_model_info(self) -> Dict:
        """Retorna información del modelo actual"""
        return self.model_trainer.get_model_info()
