# 🏥 Skin Cancer Classification System

Sistema completo de classificação de câncer de pele utilizando Deep Learning com arquitetura ResNet50 e API REST para integração com aplicações frontend.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange.svg)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Índice

- [Visão Geral](#-visão-geral)
- [Características](#-características)
- [Arquitetura](#-arquitetura)
- [Instalação](#-instalação)
- [Uso](#-uso)
- [API Documentation](#-api-documentation)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Treinamento do Modelo](#-treinamento-do-modelo)
- [Testes](#-testes)
- [Deployment](#-deployment)
- [Contribuindo](#-contribuindo)
- [Licença](#-licença)

---

## 🎯 Visão Geral

Este sistema oferece uma solução end-to-end para classificação de lesões de pele como benignas ou malignas, incluindo:

1. **Pipeline de Treinamento**: Sistema completo de treinamento com:
   - Transfer Learning (ResNet50)
   - Hyperparameter Tuning (Keras Tuner)
   - 5-Fold Cross-Validation
   - Model Ensemble

2. **API REST**: API profissional para servir predições com:
   - Arquitetura Clean Code
   - Documentação Swagger/OpenAPI
   - Tratamento robusto de erros
   - Logging completo

---

## ✨ Características

### Modelo de Machine Learning

- ✅ **Arquitetura**: ResNet50 com Transfer Learning (ImageNet)
- ✅ **Estratégia**: 5-Fold Cross-Validation
- ✅ **Ensemble**: Média de 5 modelos para maior robustez
- ✅ **Otimização**: Bayesian Optimization para hiperparâmetros
- ✅ **Data Augmentation**: Rotação, flip, zoom, shift
- ✅ **Class Balancing**: Técnicas de balanceamento avançadas

### API REST

- ✅ **Framework**: Flask com Clean Architecture
- ✅ **Documentação**: Swagger/Flasgger integrado
- ✅ **Validação**: Validação robusta de entrada
- ✅ **Logging**: Sistema completo de logs
- ✅ **Error Handling**: Tratamento profissional de erros
- ✅ **CORS**: Suporte para múltiplas origens
- ✅ **Health Check**: Endpoint de monitoramento

---

## 🏗️ Arquitetura

### Clean Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  PRESENTATION LAYER                     │
│              (Controllers - Flask Routes)               │
├─────────────────────────────────────────────────────────┤
│                  APPLICATION LAYER                      │
│           (Services - Business Logic)                   │
├─────────────────────────────────────────────────────────┤
│                     DOMAIN LAYER                        │
│              (Models - DTOs/Entities)                   │
├─────────────────────────────────────────────────────────┤
│               INFRASTRUCTURE LAYER                      │
│          (Utils, Logging, Exceptions)                   │
└─────────────────────────────────────────────────────────┘
```

### Fluxo Completo

```
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   Training   │      │    Model     │      │     API      │
│   Pipeline   │────▶ │   Ensemble   │─────▶│   Service   │
└──────────────┘      └──────────────┘      └──────────────┘
       │                      │                      │
       ▼                      ▼                      ▼
   CV Models           final_model.keras       Predictions
```

---

## 🚀 Instalação

### Pré-requisitos

- Python 3.8 ou superior
- pip
- virtualenv (recomendado)
- 4GB RAM mínimo
- 2GB espaço em disco

### Passos

```bash
# 1. Clonar repositório
git clone <repository-url>
cd skin-cancer-classifier

# 2. Criar ambiente virtual
python -m venv venv

# Ativar (Linux/Mac)
source venv/bin/activate

# Ativar (Windows)
venv\Scripts\activate

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Verificar instalação
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
```

---

## 💻 Uso

### 1. Treinar o Modelo

```bash
# Treinar com todos os recursos
python train.py

# O processo inclui:
# - Carregamento e preparação dos dados
# - Balanceamento do dataset
# - Hyperparameter tuning (se necessário)
# - 5-Fold Cross-Validation
# - Criação do modelo ensemble
# - Salvamento do modelo final

# Modelo salvo em: models/final_ensemble_model.keras
```

### 2. Executar a API

```bash
# Desenvolvimento
python run_api.py

# A API estará disponível em:
# - Endpoint: http://localhost:5000/api/v1/predict
# - Documentação: http://localhost:5000/docs
# - Health: http://localhost:5000/api/v1/health
```

### 3. Fazer Predições

#### Via cURL

```bash
curl -X POST http://localhost:5000/api/v1/predict \
  -F "file=@path/to/image.jpg" \
  -F "patient_id=PAT001"
```

#### Via Python

```python
import requests

url = "http://localhost:5000/api/v1/predict"

with open("image.jpg", "rb") as f:
    files = {"file": f}
    data = {"patient_id": "PAT001"}
    
    response = requests.post(url, files=files, data=data)
    result = response.json()
    
    print(f"Diagnóstico: {result['diagnosis']}")
    print(f"Probabilidade: {result['probability']}")
```

#### Via Frontend (JavaScript)

```javascript
const formData = new FormData();
formData.append('file', imageFile);
formData.append('patient_id', 'PAT001');

fetch('http://localhost:5000/api/v1/predict', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => {
  console.log('Diagnóstico:', data.diagnosis);
  console.log('Probabilidade:', data.probability);
  console.log('Recomendação:', data.recommendation);
});
```

---

## 📖 API Documentation

### Endpoints Principais

#### POST /api/v1/predict

Realiza predição de câncer de pele.

**Request**:
- `file`: Imagem (JPG/PNG, máx 10MB)
- `patient_id`: ID do paciente (opcional)

**Response**:
```json
{
  "diagnosis": "benign",
  "probability": 0.2341,
  "confidence_level": "high",
  "recommendation": "Lesão aparenta ser benigna...",
  "timestamp": "2024-01-15T10:30:45.123Z",
  "model_version": "1.0.0",
  "patient_id": "PAT001"
}
```

#### GET /api/v1/health

Verifica status da API.

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2024-01-15T10:30:45.123Z",
  "version": "1.0.0"
}
```

### Documentação Completa

Acesse a documentação interativa Swagger em:
```
http://localhost:5000/docs
```

Ou veja a documentação completa em:
```
docs/api_documentation.md
```

---

## 📁 Estrutura do Projeto

```
skin-cancer-classifier/
│
├── assets/                          # Dados de treinamento
│   ├── images/                      # Imagens
│   └── metadata/
│       └── metadata.csv
│
├── models/                          # Modelos treinados
│   ├── fold_1_best_model.keras      # Modelos de cada fold
│   ├── fold_2_best_model.keras
│   ├── ...
│   ├── final_ensemble_model.keras   # Modelo final para API
│   └── ensemble_info.json
│
├── results/                         # Resultados do treinamento
│   ├── cv_results.json
│   ├── cv_results.png
│   └── best_hyperparameters.json
│
├── logs/                            # Logs
│   ├── training_*.log
│   └── api_*.log
│
├── src/                             # Código fonte
│   ├── training/                    # Módulo de treinamento
│   │   ├── config.py
│   │   ├── data_manager.py
│   │   ├── model_builder.py
│   │   ├── hyperparameter_tuner.py
│   │   ├── cross_validation.py
│   │   ├── model_ensemble.py   
│   │   ├── visualization.py
│   │   └── pipeline.py
│   │
│   └── api/                         # Módulo da API
│       ├── app.py                   # Aplicação Flask
│       ├── config.py
│       ├── controllers/
│       │   └── prediction_controller.py
│       ├── services/
│       │   └── prediction_service.py
│       ├── models/
│       │   ├── prediction_request.py
│       │   └── prediction_response.py
│       └── utils/
│           ├── logger.py
│           └── exceptions.py
│
├── tests/                           # Testes
│   ├── test_prediction_service.py
│   └── test_prediction_controller.py
│
├── docs/                            # Documentação
│   └── api_documentation.md
│
├── requirements.txt                 # Dependências
├── train.py                         # Script de treinamento
├── run_api.py                       # Script para rodar API
└── README.md                        # Este arquivo
```

---

## 🎓 Treinamento do Modelo

### Pipeline de Treinamento

O sistema utiliza um pipeline sofisticado:

1. **Carregamento de Dados**
   - Leitura do CSV com metadata
   - Validação de existência das imagens
   - Preparação dos labels

2. **Balanceamento**
   - Estratégia 2:1 (benign:malignant)
   - Undersampling de benignos
   - Oversampling de malignos

3. **Hyperparameter Tuning** (Opcional)
   - Bayesian Optimization
   - 20 trials, 2 executions each
   - Otimização para recall (sensibilidade)

4. **Cross-Validation**
   - 5-Fold Stratified
   - Transfer Learning (ResNet50)
   - Fine-tuning das últimas 30 camadas
   - Class weights balanceados

5. **Model Ensemble**
   - Carrega os 5 melhores modelos
   - Cria ensemble por averaging
   - Salva modelo final unificado

### Métricas Avaliadas

- Accuracy
- AUC-ROC
- Sensitivity (Recall)
- Specificity
- Precision

### Configurações Importantes

Edite `src/training/config.py`:

```python
class Config:
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 50
    N_FOLDS = 5
    RANDOM_SEED = 42
    # ... outras configurações
```

---

## ⚠️ Avisos Legais

### Disclaimer Médico

**IMPORTANTE**: Este sistema é uma ferramenta de **auxílio diagnóstico** e **NÃO substitui** a avaliação de um profissional de saúde qualificado.

- ⚠️ Sempre consulte um dermatologista para diagnóstico definitivo
- ⚠️ Não tome decisões de tratamento baseadas apenas nesta ferramenta
- ⚠️ Os resultados devem ser interpretados por profissionais médicos
- ⚠️ Esta ferramenta tem limitações e pode apresentar erros

### Limitações Conhecidas

- Performance varia com qualidade da imagem
- Treinado para tipos específicos de lesões
- Requer iluminação adequada
- Não detecta todos os tipos de câncer de pele

---

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

## 🙏 Agradecimentos

- Dataset ISIC (International Skin Imaging Collaboration)
- TensorFlow/Keras team
- Flask e Flasgger developers
- Comunidade de Machine Learning

---