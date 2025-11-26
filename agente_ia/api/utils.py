import io
import numpy as np
from PIL import Image
from config import IMG_SIZE

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Pré-processa uma imagem para ser utilizada pelo modelo CNN.
    
    Etapas:
    1. Carrega a imagem do bytes
    2. Converte para RGB (caso seja RGBA ou escala de cinza)
    3. Redimensiona para 32x32 pixels
    4. Normaliza os valores de pixel para [0, 1]
    5. Retorna um array numpy com shape (1, 32, 32, 3)
    
    Args:
        image_bytes (bytes): Bytes da imagem
        
    Returns:
        np.ndarray: Array normalizado com shape (1, 32, 32, 3)
        
    Raises:
        ValueError: Se a imagem não puder ser processada
    """
    try:
        # Carregar imagem
        image = Image.open(io.BytesIO(image_bytes))
        
        # Converter para RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Redimensionar para 32x32
        image = image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
        
        # Converter para array numpy
        image_array = np.array(image, dtype="float32")
        
        # Normalizar para [0, 1]
        image_array = image_array / 255.0
        
        # Adicionar dimensão de batch: (32, 32, 3) -> (1, 32, 32, 3)
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    
    except Exception as e:
        raise ValueError(f"Erro ao processar imagem: {str(e)}")

def validate_image_file(filename: str) -> bool:
    """
    Valida se o arquivo é uma imagem permitida.
    
    Args:
        filename (str): Nome do arquivo
        
    Returns:
        bool: True se o arquivo é válido, False caso contrário
    """
    from config import ALLOWED_EXTENSIONS
    
    if "." not in filename:
        return False
    
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS

def format_prediction(prediction: float, confidence_threshold: float = 0.5) -> dict:
    """
    Formata a predição do modelo em um dicionário legível.
    
    Args:
        prediction (float): Valor de predição (0-1)
        confidence_threshold (float): Limiar de confiança
        
    Returns:
        dict: Dicionário com classe, confiança e rótulo
    """
    from config import CLASSES
    
    confidence = prediction if prediction > 0.5 else 1 - prediction
    class_id = 1 if prediction > 0.5 else 0
    
    return {
        "class": class_id,
        "label": CLASSES[class_id],
        "confidence": float(confidence),
        "confidence_percentage": float(confidence * 100),
        "is_confident": float(confidence) >= confidence_threshold
    }

def get_image_info(image_bytes: bytes) -> dict:
    """
    Extrai informações sobre a imagem.
    
    Args:
        image_bytes (bytes): Bytes da imagem
        
    Returns:
        dict: Dicionário com informações da imagem
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        
        return {
            "format": image.format,
            "size": image.size,
            "mode": image.mode,
            "width": image.width,
            "height": image.height
        }
    except Exception as e:
        raise ValueError(f"Erro ao extrair informações da imagem: {str(e)}")