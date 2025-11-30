# train.py (Versão Otimizada com Transfer Learning e Poucos Dados)

import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from pathlib import Path

# --- Configurações Importadas de config.py (Ou definidas localmente) ---
IMG_SIZE = 128 
BATCH_SIZE = 16 # Reduzido para datasets menores
EPOCHS = 50     # Máximo, EarlyStopping irá parar antes
DATA_DIR = Path(__file__).parent.parent / "data" # Caminho para AGENTE_IA/data
MODEL_PATH = Path(__file__).parent / "model.h5" # Caminho para AGENTE_IA/model/model.h5
# -----------------------------------------------------------------------

def prepare_data_generators(data_dir, img_size):
    """Cria geradores de dados com Data Augmentation e Normalização."""
    if not (data_dir / 'train').is_dir() or not (data_dir / 'validation').is_dir():
        raise FileNotFoundError(f"Estrutura de dados não encontrada em {data_dir}. Verifique se 'train/' e 'validation/' existem.")

    print("🛠️ Preparando Data Generators com Data Augmentation...")
    
    # Gerador para treinamento (Com Augmentation e Normalização)
    train_datagen = ImageDataGenerator(
        rescale=1./255, 
        rotation_range=30,           # Aumenta a rotação
        width_shift_range=0.3,       # Aumenta o deslocamento
        height_shift_range=0.3,      # Aumenta o deslocamento
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Gerador para validação (Apenas Normalização 1/255)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        data_dir / 'train',
        target_size=(img_size, img_size),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=True
    )

    validation_generator = val_datagen.flow_from_directory(
        data_dir / 'validation',
        target_size=(img_size, img_size),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
    
    print("✅ Generators criados.")
    return train_generator, validation_generator

def build_transfer_model(input_shape):
    """Constrói um modelo usando MobileNetV2 pré-treinado."""
    print("🏗️ Construindo modelo (MobileNetV2)...")
    
    # 1. Carregar o MobileNetV2 pré-treinado
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False, 
        weights='imagenet' 
    )
    
    # 2. Congelar o modelo base - CRUCIAL para datasets pequenos
    base_model.trainable = False

    # 3. Construir o classificador no topo
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(), 
        layers.Dense(64, activation='relu'), # Simplificado para evitar overfitting
        layers.Dropout(0.3),                 # Reduzido para 0.3
        layers.Dense(1, activation='sigmoid') 
    ])
    
    # 4. Compilar o modelo
    model.compile(
        # Learning rate muito baixo para ajustes finos nas novas camadas
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("✅ Modelo MobileNetV2 construído!")
    model.summary()
    
    return model

def train_model(model, train_gen, val_gen):
    """Treina o novo modelo."""
    print("🚀 Iniciando treinamento...")
    
    callbacks = [
        # EarlyStopping: Monitora a perda de validação. Se não melhorar por 10 épocas, para o treino.
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True), 
        # ModelCheckpoint: Salva o modelo apenas se a acurácia de validação melhorar
        ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True) 
    ]
    
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        # Calcula steps_per_epoch para garantir que todas as imagens sejam usadas
        steps_per_epoch=train_gen.samples // train_gen.batch_size, 
        validation_steps=val_gen.samples // val_gen.batch_size,
        callbacks=callbacks
    )
    
    print("✅ Treinamento concluído!")
    return history

def plot_training_history(history):
    """Plota o histórico de treinamento (loss e accuracy)."""
    # 
    print("📈 Gerando gráficos...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot Loss
    ax1.plot(history.history["loss"], label="Training Loss")
    ax1.plot(history.history["val_loss"], label="Validation Loss")
    ax1.set_title("Model Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)
    
    # Plot Accuracy
    ax2.plot(history.history["accuracy"], label="Training Accuracy")
    ax2.plot(history.history["val_accuracy"], label="Validation Accuracy")
    ax2.set_title("Model Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(MODEL_PATH.parent / "training_history_transfer.png")
    print("✅ Gráficos salvos em training_history_transfer.png")

def main():
    print("=" * 60)
    print("🤖 AGENTE DE IA - TREINAMENTO DE TRANSFER LEARNING (MobileNetV2)")
    print("=" * 60)
    
    # 1. Criar pasta 'model' se não existir
    os.makedirs(MODEL_PATH.parent, exist_ok=True)
    
    # 2. Preparar geradores
    try:
        train_generator, validation_generator = prepare_data_generators(DATA_DIR, IMG_SIZE)
    except FileNotFoundError as e:
        print(f"🚨 ERRO FATAL: {e}")
        print("Certifique-se de que a pasta 'data/' e suas subpastas 'train/cats', 'train/dogs', 'validation/cats', 'validation/dogs' existem e contêm imagens.")
        return
    
    # 3. Construir modelo
    model = build_transfer_model((IMG_SIZE, IMG_SIZE, 3))
    
    # 4. Treinar modelo
    history = train_model(model, train_generator, validation_generator)
    
    # 5. Plotar histórico
    plot_training_history(history)
    
    print("=" * 60)
    print(f"✅ PROCESSO CONCLUÍDO! O modelo mais preciso foi salvo em: {MODEL_PATH}")
    print("=" * 60)

if __name__ == "__main__":
    main()