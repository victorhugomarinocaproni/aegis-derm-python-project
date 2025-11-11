import logging
from datetime import datetime
from pathlib import Path


class APILogger:
    """Configuração de logging para a API."""

    _loggers = {}

    @classmethod
    def get_logger(cls, name: str, log_dir: Path = None) -> logging.Logger:
        """
        Obtém ou cria um logger.

        Args:
            name: Nome do logger
            log_dir: Diretório para salvar logs

        Returns:
            Logger configurado
        """
        if name in cls._loggers:
            return cls._loggers[name]

        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        # Evitar duplicação de handlers
        if logger.handlers:
            return logger

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File Handler
        if log_dir:
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / f'api_{datetime.now().strftime("%Y%m%d")}.log'
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        cls._loggers[name] = logger
        return logger