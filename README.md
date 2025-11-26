# CardioIA - Detecção de Cardiomegalia com Visão Computacional (Fase 4)

## Sobre o Projeto
CardioIA é um sistema de apoio ao diagnóstico de cardiomegalia em radiografias de tórax. O projeto utiliza técnicas de Deep Learning e redes neurais convolucionais (CNNs) para auxiliar profissionais de saúde na identificação da patologia, oferecendo uma interface interativa e pipelines reprodutíveis de treinamento.

## Arquitetura
- **Transfer Learning com ResNet50**: backbone pré-treinado no ImageNet com camadas finais customizadas e congelamento controlado para acelerar convergência e reduzir overfitting.
- **Baseline CNN Simples**: rede convolucional enxuta construída do zero para comparação de desempenho e análise de ablação.
- **Dataset**: NIH Chest X-rays (versão 224x224 redimensionada), filtrado para as classes *Cardiomegaly* e *No Finding* com balanceamento amostral e divisão treino/validação.

## Estrutura de Arquivos
- `src/`: scripts Python de pré-processamento, definição dos modelos (ResNet50 e CNN simples), orquestração de treinamento e aplicação Streamlit para inferência.
- `notebooks/`: notebook `treino_colab.ipynb` com pipeline completo para execução no Google Colab (download via Kaggle, ETL, treinamento e exportação de pesos).
- `models/`: diretório destinado aos modelos treinados (`.h5` e arquivos exportados). É populado após a execução do treinamento local ou no Colab.

## Como Executar (Localmente)
1. Crie e ative um ambiente virtual Python (opcional, porém recomendado).
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
3. Inicie a interface Streamlit:
   ```bash
   streamlit run src/app.py
   ```
4. Faça upload de uma radiografia no aplicativo para obter uma predição binária (cardiomegalia vs. normal).

## Como Executar (Google Colab)
1. Abra o notebook no Colab usando o badge a seguir (substitua o link quando o notebook estiver publicado):
   
   [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)

2. No Colab, faça upload do arquivo `kaggle.json` (ou configure `KAGGLE_USERNAME` e `KAGGLE_KEY` nas variáveis de ambiente) para permitir o download do dataset.
3. Execute as células sequencialmente para realizar o download, preparação do dataset, treinamento comparativo (ResNet50 e CNN simples) e exportação dos modelos em `.zip`.

## Equipe
Grupo 30 — Ana, Carlos, Junior.
