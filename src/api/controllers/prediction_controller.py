from datetime import datetime

from flask import request, jsonify

from ..api_config import APIConfig
from ..models.error_response import ErrorResponse
from ..models.health_check_response import HealthCheckResponse
from ..models.prediction_request import PredictionRequest
from ..services.prediction_service import PredictionService
from ..utils.exceptions import APIException
from ..utils.logger import APILogger


class PredictionController:
    """Controller para endpoints de predição."""

    def __init__(self, prediction_service: PredictionService = None):
        """
        Inicializa o controller.

        Args:
            prediction_service: Serviço de predição (opcional para testes)
        """
        self.config = APIConfig()
        self.logger = APILogger.get_logger('PredictionController', self.config.LOGS_DIR)
        self.prediction_service = prediction_service or PredictionService(self.config)

    def predict(self):
        """
        Endpoint principal de predição.

        Request:
            - file: Arquivo de imagem (multipart/form-data)
            - patient_id: ID do paciente (opcional)

        Returns:
            JSON response com a predição e status code HTTP
        """
        try:
            self.logger.info("=" * 60)
            self.logger.info("NOVA REQUISIÇÃO DE PREDIÇÃO")
            self.logger.info("=" * 60)

            if 'file' not in request.files:
                raise APIException(
                    "Nenhum arquivo foi enviado. Use o campo 'file' no form-data.",
                    status_code=400
                )

            file = request.files['file']

            if file.filename == '':
                raise APIException(
                    "Nome do arquivo está vazio.",
                    status_code=400
                )

            patient_id = request.form.get('patient_id', None)

            pred_request = PredictionRequest(
                image_file=file,
                patient_id=patient_id
            )

            is_valid, error_msg = pred_request.validate()
            if not is_valid:
                raise APIException(error_msg, status_code=400)

            response = self.prediction_service.predict(pred_request)

            self.logger.info("Predição concluída com sucesso")
            self.logger.info("=" * 60)

            return jsonify(response.to_dict()), 200

        except APIException as e:
            self.logger.warning(f"Erro de validação: {e.message}")
            error_response = ErrorResponse(
                error=e.__class__.__name__,
                message=e.message,
                status_code=e.status_code,
                timestamp=datetime.utcnow().isoformat() + 'Z',
                details=e.details
            )
            return jsonify(error_response.to_dict()), e.status_code

        except Exception as e:
            self.logger.error(f"Erro interno: {str(e)}", exc_info=True)
            error_response = ErrorResponse(
                error='InternalServerError',
                message='Erro interno do servidor. Por favor, tente novamente mais tarde.',
                status_code=500,
                timestamp=datetime.utcnow().isoformat() + 'Z'
            )
            return jsonify(error_response.to_dict()), 500

    def health(self):
        """
        Endpoint de health check.

        Returns:
            JSON response com o status da API
        """
        try:
            service_health = self.prediction_service.health_check()

            status = 'healthy' if service_health['model_loaded'] else 'unhealthy'

            response = HealthCheckResponse(
                status=status,
                model_loaded=service_health['model_loaded'],
                timestamp=datetime.utcnow().isoformat() + 'Z'
            )

            status_code = 200 if status == 'healthy' else 503

            return jsonify(response.to_dict()), status_code

        except Exception as e:
            self.logger.error(f"Erro no health check: {str(e)}")
            error_response = ErrorResponse(
                error='HealthCheckError',
                message=str(e),
                status_code=500,
                timestamp=datetime.utcnow().isoformat() + 'Z'
            )
            return jsonify(error_response.to_dict()), 500