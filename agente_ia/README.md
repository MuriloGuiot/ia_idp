# 🤖 Agente de IA - Classificador de Imagens de Cães e Gatos

Este projeto implementa um agente de Inteligência Artificial usando uma **Rede Neural Convolucional (CNN)**, treinada no dataset CIFAR-10 (filtrando Cães e Gatos), e expõe a funcionalidade através de uma **API REST** construída com FastAPI.

## ⚙️ Pré-requisitos

* Python 3.8+
* pip

## 🚀 Como Executar o Agente

### 1. Configurar o Ambiente

Crie e ative um ambiente virtual (recomendado) e instale as dependências:

```bash
# Crie o ambiente (opcional, mas recomendado)
python -m venv venv
source venv/bin/activate  # No Linux/macOS
# .venv\Scripts\activate  # No Windows

# Instale as dependências
pip install -r requirements.txt