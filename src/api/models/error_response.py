from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class ErrorResponse:
    """Resposta de erro padronizada."""

    error: str
    message: str
    status_code: int
    timestamp: str
    details: Optional[Dict[str, Any]] = None

    def to_dict(self):
        """Converte para dicionário."""
        response = {
            'error': self.error,
            'message': self.message,
            'status_code': self.status_code,
            'timestamp': self.timestamp
        }

        if self.details:
            response['details'] = self.details

        return response

