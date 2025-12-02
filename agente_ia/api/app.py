import os
import sys
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from config import IMG_SIZE

# Adicionar diretório pai ao path para importar config e utils
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    API_TITLE,
    API_VERSION,
    API_DESCRIPTION,
    ALLOWED_ORIGINS,
    MODEL_PATH,
    CLASSES,
    LOG_LEVEL
)
from utils import (
    preprocess_image,
    validate_image_file,
    format_prediction,
    get_image_info
)

# Configurar logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)

# Criar aplicação FastAPI
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Adicionar CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variável global para armazenar o modelo
model = None

class ClassificationResponse(BaseModel):
    """Modelo de resposta para classificação"""
    success: bool
    class_id: int
    label: str
    confidence: float
    confidence_percentage: float
    is_confident: bool
    image_info: Optional[dict] = None
    message: str

class HealthResponse(BaseModel):
    """Modelo de resposta para health check"""
    status: str
    model_loaded: bool
    version: str

@app.on_event("startup")
async def load_model():
    """
    Carrega o modelo CNN ao iniciar a aplicação.
    
    Verifica se o arquivo do modelo existe e tenta carregá-lo.
    Se falhar, a API continuará funcionando mas retornará erro ao
    tentar fazer predições.
    """
    global model
    
    logger.info("🚀 Iniciando aplicação FastAPI...")
    
    if not MODEL_PATH.exists():
        logger.warning(f"⚠️  Modelo não encontrado em {MODEL_PATH}")
        logger.warning("   Execute 'python model/train.py' para treinar o modelo")
        return
    
    try:
        logger.info(f"📥 Carregando modelo de {MODEL_PATH}...")
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info("✅ Modelo carregado com sucesso!")
        
        # Fazer uma predição dummy para aquecimento
        dummy_input = np.random.rand(1, IMG_SIZE, IMG_SIZE, 3).astype("float32")
        _ = model.predict(dummy_input, verbose=0)
        logger.info("🔥 Modelo aquecido e pronto para predições")
        
    except Exception as e:
        logger.error(f"❌ Erro ao carregar modelo: {str(e)}")
        model = None

@app.get("/", response_model=dict)
async def root():
    """
    Retorna informações sobre a API.
    
    Returns:
        dict: Informações da API
    """
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "description": API_DESCRIPTION,
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/health",
            "classify": "/classify"
        },
        "classes": CLASSES
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Verifica o status da API e se o modelo está carregado.
    
    Returns:
        HealthResponse: Status da API
    """
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        version=API_VERSION
    )

@app.post("/classify", response_model=ClassificationResponse)
async def classify_image(file: UploadFile = File(...)):
    """
    Classifica uma imagem enviada como Gato ou Cachorro.
    
    Etapas:
    1. Valida o arquivo enviado
    2. Lê os bytes da imagem
    3. Pré-processa a imagem
    4. Faz a predição com o modelo
    5. Formata e retorna o resultado
    
    Args:
        file (UploadFile): Arquivo de imagem enviado
        
    Returns:
        ClassificationResponse: Resultado da classificação
        
    Raises:
        HTTPException: Se houver erro na validação ou processamento
    """
    
    # Validar se modelo está carregado
    if model is None:
        logger.error("❌ Modelo não está carregado")
        raise HTTPException(
            status_code=503,
            detail="Modelo não está carregado. Execute 'python model/train.py' para treinar."
        )
    
    # Validar nome do arquivo
    if not validate_image_file(file.filename):
        logger.warning(f"⚠️  Arquivo inválido: {file.filename}")
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de arquivo não permitido. Permitidos: jpg, jpeg, png, gif, bmp"
        )
    
    try:
        # Ler bytes da imagem
        logger.info(f"📥 Processando imagem: {file.filename}")
        image_bytes = await file.read()
        
        # Extrair informações da imagem
        image_info = get_image_info(image_bytes)
        logger.info(f"   Tamanho original: {image_info['size']}")
        
        # Pré-processar imagem
        processed_image = preprocess_image(image_bytes)
        logger.info("✅ Imagem pré-processada")
        
        # Fazer predição
        logger.info("🤖 Fazendo predição...")
        prediction = model.predict(processed_image, verbose=0)[0][0]
        logger.info(f"   Predição bruta: {prediction:.4f}")
        
        # Formatar resultado
        result = format_prediction(prediction)
        logger.info(f"✅ Resultado: {result['label']} (confiança: {result['confidence_percentage']:.2f}%)")
        
        return ClassificationResponse(
            success=True,
            class_id=result["class"],
            label=result["label"],
            confidence=result["confidence"],
            confidence_percentage=result["confidence_percentage"],
            is_confident=result["is_confident"],
            image_info=image_info,
            message=f"Classificado como {result['label']} com {result['confidence_percentage']:.2f}% de confiança"
        )
    
    except ValueError as e:
        logger.error(f"❌ Erro ao processar imagem: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Erro ao processar imagem: {str(e)}"
        )
    
    except Exception as e:
        logger.error(f"❌ Erro inesperado: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao fazer predição: {str(e)}"
        )

@app.get("/info")
async def get_info():
    """
    Retorna informações detalhadas sobre o modelo e a API.
    
    Returns:
        dict: Informações detalhadas
    """
    model_info = None
    
    if model is not None:
        model_info = {
            "input_shape": model.input_shape,
            "output_shape": model.output_shape,
            "total_params": int(model.count_params()),
            "layers": len(model.layers)
        }
    
    return {
        "api": {
            "title": API_TITLE,
            "version": API_VERSION,
            "description": API_DESCRIPTION
        },
        "model": model_info,
        "classes": CLASSES,
        "allowed_extensions": ["jpg", "jpeg", "png", "gif", "bmp"],
        "max_upload_size_mb": 10
    }

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("🚀 AGENTE DE IA - API DE CLASSIFICAÇÃO DE IMAGENS")
    print("=" * 60)
    print()
    print("📡 Iniciando servidor FastAPI...")
    print("   URL: http://localhost:8000")
    print("   Docs: http://localhost:8000/docs")
    print()
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )