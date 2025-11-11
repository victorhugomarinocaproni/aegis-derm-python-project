import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.api.app import main

if __name__ == '__main__':
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║         SKIN CANCER CLASSIFICATION API v1.0.0                 ║
    ╚═══════════════════════════════════════════════════════════════╝

    Iniciando servidor...
    Documentação disponível em: http://localhost:5000/docs
    Health check em: http://localhost:5000/api/v1/health

    """)

    main()

