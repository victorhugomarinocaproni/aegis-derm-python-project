class APIException(Exception):
    """Exceção base para a API."""

    def __init__(self, message: str, status_code: int = 500, details: dict = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class ModelNotLoadedException(APIException):
    """Exceção quando o modelo não está carregado."""

    def __init__(self, message: str = "Modelo não carregado"):
        super().__init__(message, status_code=503)


class InvalidImageException(APIException):
    """Exceção para imagem inválida."""

    def __init__(self, message: str = "Imagem inválida"):
        super().__init__(message, status_code=400)


class PredictionException(APIException):
    """Exceção durante predição."""

    def __init__(self, message: str = "Erro ao realizar predição"):
        super().__init__(message, status_code=500)


class ValidationException(APIException):
    """Exceção de validação."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message, status_code=400, details=details)