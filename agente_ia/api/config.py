# config.py

import os
from pathlib import Path

# Diretório raiz do projeto
BASE_DIR = Path(__file__).resolve().parent.parent 
MODEL_PATH = BASE_DIR / "model" / "model.h5"

# Configurações da API
API_TITLE = "Agente de IA - Classificador de Imagens"
API_VERSION = "1.0.0"
API_DESCRIPTION = "API para classificação de imagens de cães e gatos usando CNN"

# Configurações de CORS
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
    "http://localhost:5500",  # Live Server do VSCode
    "http://127.0.0.1:5500",
]

# Configurações de upload (mantidas)
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "gif", "bmp"}

# Configurações do modelo
# CORREÇÃO CRÍTICA: Ajustado para 128 (o novo tamanho de treinamento)
IMG_SIZE = 128
CLASSES = {
    0: "Gato 🐱",
    1: "Cachorro 🐕"
}

# Configurações de logging
LOG_LEVEL = "INFO"