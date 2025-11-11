from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class PredictionResponse:
    """Resposta de predição."""

    diagnosis: str  # 'benign' ou 'malignant'
    probability: float  # Probabilidade da classe positiva (maligno)
    confidence_level: str  # 'high', 'medium', 'low'
    recommendation: str  # Recomendação médica
    timestamp: str
    model_version: str = "1.0.0"
    patient_id: Optional[str] = None

    def to_dict(self):
        """Converte para dicionário."""
        return {
            'diagnosis': self.diagnosis,
            'probability': round(self.probability, 4),
            'confidence_level': self.confidence_level,
            'recommendation': self.recommendation,
            'timestamp': self.timestamp,
            'model_version': self.model_version,
            'patient_id': self.patient_id
        }

    @staticmethod
    def get_confidence_level(probability: float):
        """Determina o nível de confiança baseado na probabilidade."""
        from ..api_config import APIConfig

        # Para decisões próximas de 0.5, confiança é baixa
        distance_from_threshold = abs(probability - APIConfig.MALIGNANT_THRESHOLD)

        if distance_from_threshold >= (1 - APIConfig.CONFIDENCE_HIGH):
            return 'high'
        elif distance_from_threshold >= (1 - APIConfig.CONFIDENCE_MEDIUM):
            return 'medium'
        else:
            return 'low'

    @staticmethod
    def get_recommendation(diagnosis: str, confidence_level: str):
        """Gera recomendação baseada no diagnóstico e confiança."""
        if diagnosis == 'malignant':
            if confidence_level == 'high':
                return "URGENTE: Consulte um dermatologista imediatamente para avaliação e biópsia."
            elif confidence_level == 'medium':
                return "IMPORTANTE: Agende consulta com dermatologista o mais breve possível."
            else:
                return "Recomenda-se consulta com dermatologista para avaliação profissional."
        else:  # benign
            if confidence_level == 'high':
                return "Lesão aparenta ser benigna. Monitore alterações e consulte dermatologista em caso de mudanças."
            elif confidence_level == 'medium':
                return "Lesão aparenta ser benigna, mas recomenda-se avaliação dermatológica para confirmação."
            else:
                return "Resultado inconclusivo. Consulte um dermatologista para avaliação adequada."