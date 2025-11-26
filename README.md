# CardioIA - Fase 4: Visão Computacional no Diagnóstico Cardíaco

## Sobre o Projeto
Este repositório consolida o pipeline desenvolvido para detectar cardiomegalia em radiografias de tórax. O sistema compara duas abordagens de Deep Learning: uma CNN criada do zero como baseline e um modelo de Transfer Learning baseado na ResNet-50, destacando ganhos de desempenho, tempo de treinamento e robustez.

## Dataset
Utilizamos o *NIH Chest X-ray Dataset* (versão otimizada em 224x224 pixels), focado nas classes *Cardiomegaly* e *No Finding*. O notebook realiza o download via Kaggle, aplica o pré-processamento necessário e separa os conjuntos de treino e validação.

## Estrutura do Projeto
- `src/`: código-fonte para pré-processamento, definição das arquiteturas (CNN própria e ResNet-50) e scripts de treinamento/inferência.
- `notebooks/`: notebook `treino_colab.ipynb` preparado para execução em nuvem com todo o pipeline automatizado.
- `app/`: interface Streamlit para inferência (aplicativo principal está em `src/app.py` caso o diretório não esteja presente localmente).

## Como Executar (Google Colab - Recomendado)
1. Abra o notebook `notebooks/treino_colab.ipynb` e clique no badge **Open in Colab** disponível no topo.
2. Execute as células na ordem. O notebook clona este repositório, realiza o setup das dependências e baixa o dataset automaticamente.
3. Necessário fazer upload do arquivo `kaggle.json` quando solicitado para habilitar o download direto via API.

## Como Executar (Localmente)
1. (Opcional) Crie e ative um ambiente virtual Python.
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
3. Inicie a aplicação Streamlit:
   ```bash
   streamlit run src/app.py
   ```

## Equipe
Grupo 30 — Ana (Documentação), Carlos (Frontend/EDA), Junior (Tech Lead/Modelagem).
