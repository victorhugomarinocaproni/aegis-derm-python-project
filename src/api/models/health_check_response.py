from dataclasses import dataclass


@dataclass
class HealthCheckResponse:
    """Resposta do health check."""

    status: str  # 'healthy' ou 'unhealthy'
    model_loaded: bool
    timestamp: str
    version: str = "1.0.0"

    def to_dict(self):
        """Converte para dicionário."""
        return {
            'status': self.status,
            'model_loaded': self.model_loaded,
            'timestamp': self.timestamp,
            'version': self.version
        }

