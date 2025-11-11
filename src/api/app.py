from flask import Flask
from flask_cors import CORS
from flasgger import Swagger, swag_from

from .api_config import APIConfig
from .controllers.prediction_controller import PredictionController
from .utils.logger import APILogger


def create_app(config: APIConfig = None) -> Flask:
    """
    Factory function para criar a aplicação Flask.

    Args:
        config: Configurações da API

    Returns:
        Aplicação Flask configurada
    """
    # Configuração
    if config is None:
        config = APIConfig()
        config.validate_config()

    # Criar app
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = config.MAX_FILE_SIZE

    # CORS
    CORS(app, origins=config.CORS_ORIGINS)

    # Logger
    logger = APILogger.get_logger('FlaskApp', config.LOGS_DIR)

    # Swagger configuration
    swagger_config = {
        "headers": [],
        "specs": [
            {
                "endpoint": 'apispec',
                "route": '/apispec.json',
                "rule_filter": lambda rule: True,
                "model_filter": lambda tag: True,
            }
        ],
        "static_url_path": "/flasgger_static",
        "swagger_ui": True,
        "specs_route": "/docs"
    }

    swagger_template = {
        "info": {
            "title": config.SWAGGER_TITLE,
            "version": config.SWAGGER_VERSION,
            "description": config.SWAGGER_DESCRIPTION,
            "contact": {
                "name": "API Support",
                "email": "support@skincancerapi.com"
            }
        },
        "host": f"{config.HOST}:{config.PORT}",
        "basePath": "/api/v1",
        "schemes": ["http", "https"],
        "consumes": ["multipart/form-data"],
        "produces": ["application/json"],
        "tags": [
            {
                "name": "Prediction",
                "description": "Endpoints para predição de câncer de pele"
            },
            {
                "name": "Health",
                "description": "Endpoints de monitoramento"
            }
        ],
        "definitions": {
            "PredictionResponse": {
                "type": "object",
                "properties": {
                    "diagnosis": {
                        "type": "string",
                        "enum": ["benign", "malignant"],
                        "description": "Diagnóstico da lesão"
                    },
                    "probability": {
                        "type": "number",
                        "format": "float",
                        "description": "Probabilidade de ser maligno (0.0 a 1.0)"
                    },
                    "confidence_level": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                        "description": "Nível de confiança da predição"
                    },
                    "recommendation": {
                        "type": "string",
                        "description": "Recomendação médica"
                    },
                    "timestamp": {
                        "type": "string",
                        "format": "date-time",
                        "description": "Timestamp da predição (ISO 8601)"
                    },
                    "model_version": {
                        "type": "string",
                        "description": "Versão do modelo"
                    },
                    "patient_id": {
                        "type": "string",
                        "description": "ID do paciente (se fornecido)"
                    }
                }
            },
            "ErrorResponse": {
                "type": "object",
                "properties": {
                    "error": {
                        "type": "string",
                        "description": "Tipo do erro"
                    },
                    "message": {
                        "type": "string",
                        "description": "Mensagem de erro"
                    },
                    "status_code": {
                        "type": "integer",
                        "description": "Código de status HTTP"
                    },
                    "timestamp": {
                        "type": "string",
                        "format": "date-time"
                    },
                    "details": {
                        "type": "object",
                        "description": "Detalhes adicionais do erro"
                    }
                }
            },
            "HealthCheckResponse": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["healthy", "unhealthy"]
                    },
                    "model_loaded": {
                        "type": "boolean"
                    },
                    "timestamp": {
                        "type": "string",
                        "format": "date-time"
                    },
                    "version": {
                        "type": "string"
                    }
                }
            }
        }
    }

    swagger = Swagger(app, config=swagger_config, template=swagger_template)

    # Inicializar controller
    controller = PredictionController()

    # ========================================================================
    # ROUTES
    # ========================================================================

    @app.route('/api/v1/predict', methods=['POST'])
    @swag_from({
        'tags': ['Prediction'],
        'summary': 'Realizar predição de câncer de pele',
        'description': '''
        Recebe uma imagem de lesão de pele e retorna o diagnóstico (benigno ou maligno) 
        com probabilidade e nível de confiança.

        **Importante:** Esta é uma ferramenta de auxílio diagnóstico. 
        Sempre consulte um profissional de saúde qualificado.
        ''',
        'consumes': ['multipart/form-data'],
        'produces': ['application/json'],
        'parameters': [
            {
                'name': 'file',
                'in': 'formData',
                'type': 'file',
                'required': True,
                'description': 'Imagem da lesão de pele (JPG, JPEG ou PNG, máx 10MB)'
            },
            {
                'name': 'patient_id',
                'in': 'formData',
                'type': 'string',
                'required': False,
                'description': 'ID do paciente (opcional)'
            }
        ],
        'responses': {
            200: {
                'description': 'Predição realizada com sucesso',
                'schema': {'$ref': '#/definitions/PredictionResponse'},
                'examples': {
                    'application/json': {
                        'diagnosis': 'benign',
                        'probability': 0.2341,
                        'confidence_level': 'high',
                        'recommendation': 'Lesão aparenta ser benigna. Monitore alterações e consulte dermatologista em caso de mudanças.',
                        'timestamp': '2024-01-15T10:30:45.123Z',
                        'model_version': '1.0.0',
                        'patient_id': 'PAT001'
                    }
                }
            },
            400: {
                'description': 'Requisição inválida',
                'schema': {'$ref': '#/definitions/ErrorResponse'}
            },
            500: {
                'description': 'Erro interno do servidor',
                'schema': {'$ref': '#/definitions/ErrorResponse'}
            },
            503: {
                'description': 'Serviço indisponível (modelo não carregado)',
                'schema': {'$ref': '#/definitions/ErrorResponse'}
            }
        }
    })
    def predict():
        """Endpoint de predição."""
        return controller.predict()

    @app.route('/api/v1/health', methods=['GET'])
    @swag_from({
        'tags': ['Health'],
        'summary': 'Verificar status da API',
        'description': 'Retorna o status de saúde da API e se o modelo está carregado.',
        'responses': {
            200: {
                'description': 'API está saudável',
                'schema': {'$ref': '#/definitions/HealthCheckResponse'},
                'examples': {
                    'application/json': {
                        'status': 'healthy',
                        'model_loaded': True,
                        'timestamp': '2024-01-15T10:30:45.123Z',
                        'version': '1.0.0'
                    }
                }
            },
            503: {
                'description': 'API não está saudável (modelo não carregado)',
                'schema': {'$ref': '#/definitions/HealthCheckResponse'}
            }
        }
    })
    def health():
        """Endpoint de health check."""
        return controller.health()

    @app.route('/', methods=['GET'])
    def index():
        """Rota raiz - redireciona para documentação."""
        return {
            'message': 'Skin Cancer Classification API',
            'version': config.SWAGGER_VERSION,
            'documentation': '/docs',
            'endpoints': {
                'predict': '/api/v1/predict',
                'health': '/api/v1/health'
            }
        }, 200

    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        """Handler para 404."""
        from datetime import datetime
        return {
            'error': 'NotFound',
            'message': 'Endpoint não encontrado',
            'status_code': 404,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }, 404

    @app.errorhandler(413)
    def request_entity_too_large(error):
        """Handler para arquivo muito grande."""
        from datetime import datetime
        return {
            'error': 'RequestEntityTooLarge',
            'message': f'Arquivo muito grande. Tamanho máximo: {config.MAX_FILE_SIZE / (1024 * 1024):.0f}MB',
            'status_code': 413,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }, 413

    @app.errorhandler(500)
    def internal_error(error):
        """Handler para erro interno."""
        from datetime import datetime
        logger.error(f"Erro interno: {str(error)}", exc_info=True)
        return {
            'error': 'InternalServerError',
            'message': 'Erro interno do servidor',
            'status_code': 500,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }, 500

    logger.info("=" * 60)
    logger.info("SKIN CANCER CLASSIFICATION API")
    logger.info("=" * 60)
    logger.info(f"Versão: {config.SWAGGER_VERSION}")
    logger.info(f"Documentação: http://{config.HOST}:{config.PORT}/docs")
    logger.info("=" * 60)

    return app


# ============================================================================
# PONTO DE ENTRADA
# ============================================================================

def main():
    """Função principal para executar a API."""
    config = APIConfig()
    config.validate_config()

    app = create_app(config)
    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG
    )


if __name__ == '__main__':
    main()