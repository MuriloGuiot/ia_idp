import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Configurações
BATCH_SIZE = 32
EPOCHS = 20
IMG_SIZE = 32
MODEL_PATH = "model.h5"

def load_and_prepare_data():
    """
    Carrega o dataset CIFAR-10 e prepara os dados para treinamento.
    
    O CIFAR-10 contém 60.000 imagens de 32x32 pixels em 10 classes.
    Usaremos as classes 3 (gato) e 5 (cachorro).
    
    Returns:
        tuple: (X_train, y_train, X_test, y_test) - Dados preparados
    """
    print("📥 Carregando dataset CIFAR-10...")
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    
    # Classes: 3=gato, 5=cachorro
    cat_class = 3
    dog_class = 5
    
    # Filtrar apenas gatos e cachorros
    train_mask = (y_train == cat_class) | (y_train == dog_class)
    test_mask = (y_test == cat_class) | (y_test == dog_class)
    
    X_train = X_train[train_mask.flatten()]
    y_train = y_train[train_mask.flatten()]
    X_test = X_test[test_mask.flatten()]
    y_test = y_test[test_mask.flatten()]
    
    # Converter classes para 0 (gato) e 1 (cachorro)
    y_train = (y_train == dog_class).astype(int)
    y_test = (y_test == dog_class).astype(int)
    
    # Normalizar os valores de pixel para [0, 1]
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0
    
    print(f"✅ Dataset carregado: {X_train.shape[0]} imagens de treinamento, {X_test.shape[0]} de teste")
    print(f"   Classe 0 (Gato): {np.sum(y_train == 0)}")
    print(f"   Classe 1 (Cachorro): {np.sum(y_train == 1)}")
    
    return X_train, y_train, X_test, y_test

def build_cnn_model(input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    """
    Constrói uma Rede Neural Convolucional (CNN) para classificação de imagens.
    
    Arquitetura:
    - 2 blocos de convolução com pooling
    - Camada de flatten
    - 2 camadas densas com dropout
    - Camada de saída com sigmoid (classificação binária)
    
    Args:
        input_shape (tuple): Forma das imagens de entrada
        
    Returns:
        model: Modelo Keras compilado
    """
    print("🏗️  Construindo modelo CNN...")
    
    model = models.Sequential([
        # Bloco 1: Convolução + Pooling
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        # Bloco 2: Convolução + Pooling
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        
        # Bloco 3: Convolução
        layers.Conv2D(64, (3, 3), activation="relu"),
        
        # Flatten
        layers.Flatten(),
        
        # Camadas Densas com Dropout
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.5),
        
        # Camada de Saída (Classificação Binária)
        layers.Dense(1, activation="sigmoid")
    ])
    
    # Compilar o modelo
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    print("✅ Modelo construído com sucesso!")
    model.summary()
    
    return model

def train_model(model, X_train, y_train, X_test, y_test):
    """
    Treina o modelo CNN com data augmentation.
    
    Args:
        model: Modelo Keras
        X_train: Imagens de treinamento
        y_train: Rótulos de treinamento
        X_test: Imagens de teste
        y_test: Rótulos de teste
        
    Returns:
        history: Histórico de treinamento
    """
    print("🚀 Iniciando treinamento...")
    
    # Data Augmentation para melhorar a generalização
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2
    )
    
    # Treinar o modelo
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        verbose=1
    )
    
    print("✅ Treinamento concluído!")
    
    return history

def evaluate_model(model, X_test, y_test):
    """
    Avalia o desempenho do modelo no conjunto de teste.
    
    Args:
        model: Modelo Keras treinado
        X_test: Imagens de teste
        y_test: Rótulos de teste
    """
    print("📊 Avaliando modelo...")
    
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"✅ Resultados da Avaliação:")
    print(f"   Loss: {loss:.4f}")
    print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

def save_model(model, path=MODEL_PATH):
    """
    Salva o modelo treinado em formato .h5.
    
    Args:
        model: Modelo Keras treinado
        path: Caminho para salvar o modelo
    """
    print(f"💾 Salvando modelo em {path}...")
    model.save(path)
    print(f"✅ Modelo salvo com sucesso!")

def plot_training_history(history):
    """
    Plota o histórico de treinamento (loss e accuracy).
    
    Args:
        history: Histórico retornado pelo fit()
    """
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
    plt.savefig("training_history.png")
    print("✅ Gráficos salvos em training_history.png")

def main():
    """
    Função principal que orquestra todo o processo de treinamento.
    """
    print("=" * 60)
    print("🤖 AGENTE DE IA - TREINAMENTO DE MODELO CNN")
    print("   Classificação de Cães e Gatos")
    print("=" * 60)
    print()
    
    # 1. Carregar e preparar dados
    X_train, y_train, X_test, y_test = load_and_prepare_data()
    print()
    
    # 2. Construir modelo
    model = build_cnn_model()
    print()
    
    # 3. Treinar modelo
    history = train_model(model, X_train, y_train, X_test, y_test)
    print()
    
    # 4. Avaliar modelo
    evaluate_model(model, X_test, y_test)
    print()
    
    # 5. Salvar modelo
    save_model(model)
    print()
    
    # 6. Plotar histórico
    plot_training_history(history)
    print()
    
    print("=" * 60)
    print("✅ PROCESSO CONCLUÍDO COM SUCESSO!")
    print("   O modelo foi treinado e salvo em:", MODEL_PATH)
    print("=" * 60)

if __name__ == "__main__":
    main()