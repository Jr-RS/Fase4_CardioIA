

# CardioIA - Assistente Cardiológico Virtual

<p align="center">
<a href="https://www.fiap.com.br/"><img src="assets/logo-fiap.png" alt="FIAP - Faculdade de Informática e Administração Paulista" border="0" width=40% height=40%></a>
</p>

---

## Sumário
1. Visão Geral
2. Objetivos do Projeto
3. Arquitetura do Pipeline
4. Estrutura de Pastas
5. Instruções de Uso
6. Resultados e Métricas
7. Decisões Técnicas
8. Distribuição de Responsabilidades
9. Referências e Links Úteis
10. Licença

---

## 1. Visão Geral

O CardioIA é um protótipo de Assistente Cardiológico Virtual desenvolvido para apoiar a tomada de decisão clínica por meio da análise de radiografias de tórax. Utiliza técnicas modernas de Visão Computacional, como redes neurais convolucionais (CNNs) e Transfer Learning, para identificar padrões associados à cardiomegalia. O sistema é modular, automatizado e reprodutível, integrando scripts Python, experiment tracking e uma interface intuitiva em Streamlit.

---

## 2. Objetivos do Projeto

- Realizar o pré-processamento de imagens médicas simuladas (ex: raios-X do NIH Chest X-ray Dataset).
- Treinar e avaliar modelos de CNN para classificar e identificar padrões em imagens médicas.
- Testar duas abordagens: CNN simples do zero e Transfer Learning (ResNet-50).
- Apresentar resultados de forma acessível em uma aplicação web simples (Streamlit) e notebook Colab.
- Promover trabalho em equipe, colaboração interdisciplinar e documentação clara.

---

## 3. Arquitetura do Pipeline

O pipeline do CardioIA é composto por etapas independentes e rastreáveis:
1. **ETL (`src/data_preprocessing.py`)**: Pré-processamento, redimensionamento, normalização e organização dos dados em conjuntos de treino/validação.
2. **Treinamento (`src/train.py`)**: Treinamento da CNN do zero e do modelo ResNet-50, salvando modelos e métricas.
3. **Experiment Tracking (`experiments/`)**: Armazenamento de gráficos, logs, métricas e artefatos dos experimentos.
4. **Inferência (`src/app.py`)**: Interface Streamlit para diagnóstico em novas imagens.
5. **Orquestração Colab (`notebooks/treino_colab.ipynb`)**: Notebook que automatiza todo o pipeline, do download dos dados à inferência, facilitando reprodutibilidade e compartilhamento.

<p align="center">
<img src="assets/tela_resultado.png" alt="Diagrama do pipeline e resultado" width="60%">
</p>

---

## 4. Estrutura de Pastas

```text
Fase4_CardioIA/
├── assets/                 # Logos e imagens para documentação
│   ├── tela_inicial.png    # Print da tela inicial do app
│   ├── tela_imagem_carregada.png # Print da imagem carregada
│   ├── tela_resultado.png  # Print do resultado da inferência
│   └── logo-fiap.png       # Logo FIAP
├── data/                   # Dados de treino/validação (após ETL)
│   ├── train/
│   └── validation/
├── experiments/            # Métricas, gráficos e artefatos de experimentos
├── models/                 # Modelos treinados (.h5)
├── notebooks/
│   └── treino_colab.ipynb  # Notebook orquestrador (Colab)
├── src/
│   ├── app.py              # Aplicação Streamlit de inferência
│   ├── data_preprocessing.py
│   ├── model_resnet.py
│   ├── model_simple_cnn.py
│   └── train.py            # Script principal de treino
└── README.md
```

---

## 5. Instruções de Uso

### Pré-requisitos
- Python 3.10+
- Instalar dependências: `pip install -r requirements.txt`
- Dados organizados em `data/train` e `data/validation` (após ETL)

### Executando o ETL
```bash
python src/data_preprocessing.py
```

### Treinando os modelos
```bash
python src/train.py
```

### Rodando o app de inferência
```bash
streamlit run src/app.py
```

### Reprodutibilidade e Orquestração no Google Colab
O notebook `notebooks/treino_colab.ipynb` automatiza todo o pipeline, desde o download dos dados, execução do ETL, treinamento dos modelos, até a geração dos resultados e inferência. Basta abrir o notebook no Colab, seguir as instruções e executar as células sequencialmente. Não é necessário configurar nada localmente.

---

## 6. Resultados e Métricas

Os resultados comprovam o impacto do Transfer Learning em tarefas médicas:

- **CNN do zero:**
    - Acurácia: 0.82
    - Loss: 0.41
- **ResNet-50 (Transfer Learning):**
    - Acurácia: 0.89
    - Loss: 0.28

<p align="center">
<img src="experiments/grafico_acuracia.png" alt="Gráfico de acurácia" width="30%">
<img src="experiments/grafico_loss.png" alt="Gráfico de loss" width="30%">
<img src="assets/grafico_comparativo.png" alt="Gráfico comparativo CNN vs ResNet-50" width="30%">
</p>

**Avaliação dos resultados:**
O modelo ResNet-50 apresentou desempenho superior em acurácia e menor perda, evidenciando os benefícios do Transfer Learning em cenários com dados limitados. Os gráficos mostram a evolução do treinamento e a diferença entre as abordagens. Todos os artefatos, logs e gráficos estão disponíveis na pasta `experiments/` para consulta detalhada.

---

## 7. Decisões Técnicas

- Uso de TensorFlow/Keras para modelagem e treinamento.
- Separação clara entre scripts de ETL, treinamento e inferência.
- Experiment tracking via organização de artefatos e métricas.
- Streamlit para interface simples e acessível.
- Reprodutibilidade garantida por scripts e notebook Colab.
- Estrutura modular para facilitar manutenção e expansão.
- **Escolha do ResNet-50:** Optamos pelo ResNet-50 por ser um dos modelos mais consagrados em tarefas de classificação de imagens médicas, devido à sua profundidade, capacidade de generalização e uso eficiente de transfer learning. Isso nos permitiu obter resultados superiores com menos dados e tempo de treinamento, além de facilitar a reprodutibilidade.

---

## 8. Distribuição de Responsabilidades

| Integrante                        | Responsabilidades principais |
|-----------------------------------|-----------------------------|
| Ana Beatriz Duarte Domingues      | ETL, documentação, testes   |
| Junior Rodrigues da Silva         | Modelos, experiment tracking, Streamlit |
| Carlos Emilio Castillo Estrada    | Colab, integração, validação|

---

## 9. Referências e Links Úteis

- [NIH Chest X-ray Dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC)
- [Paper ResNet](https://arxiv.org/abs/1512.03385)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Documentation](https://keras.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

## 10. Licença

<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1"><p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/"><a property="dct:title" rel="cc:attributionURL" href="https://github.com/agodoi/template">MODELO GIT FIAP</a> por <a rel="cc:attributionURL dct:creator" property="cc:attributionName" href="https://fiap.com.br">Fiap</a> está licenciado sobre <a href="http://creativecommons.org/licenses/by/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">Attribution 4.0 International</a>.</p>

<p align="center">
<img src="assets/tela_resultado.png" alt="Diagrama do pipeline e resultado" width="60%">
</p>

---

---

## 3. Estrutura de Pastas

```text
Fase4_CardioIA/
├── assets/                 # Logos e imagens para documentação
│   ├── tela_inicial.png    # Print da tela inicial do app
│   ├── tela_imagem_carregada.png # Print da imagem carregada
│   ├── tela_resultado.png  # Print do resultado da inferência
│   └── logo-fiap.png       # Logo FIAP
├── data/                   # Dados de treino/validação (após ETL)
│   ├── train/
│   └── validation/
├── experiments/            # Métricas, gráficos e artefatos de experimentos
├── models/                 # Modelos treinados (.h5)
├── notebooks/
│   └── treino_colab.ipynb  # Notebook orquestrador (Colab)
├── src/
│   ├── app.py              # Aplicação Streamlit de inferência
│   ├── data_preprocessing.py
│   ├── model_resnet.py
│   ├── model_simple_cnn.py
│   └── train.py            # Script principal de treino
└── README.md
```

---


## 4. Instruções de Uso

### Pré-requisitos
- Python 3.10+
- Instalar dependências: `pip install -r requirements.txt`
- Dados organizados em `data/train` e `data/validation` (após ETL)

### Executando o ETL
O script `src/data_preprocessing.py` realiza o pré-processamento das imagens, incluindo redimensionamento, normalização e organização em pastas de treino/validação. Isso garante dados limpos e prontos para o treinamento dos modelos.
```bash
python src/data_preprocessing.py
```

### Treinando os modelos
O script `src/train.py` permite treinar tanto a CNN do zero quanto o modelo ResNet-50. Os hiperparâmetros principais estão documentados no próprio script.
```bash
python src/train.py
```

### Rodando o app de inferência
O app Streamlit (`src/app.py`) oferece uma interface intuitiva para carregar radiografias e obter o diagnóstico. Prints das telas:

<p align="center">
<img src="assets/tela_imagem_carregada.png" alt="Imagem carregada" width="45%">
<img src="assets/tela_resultado.png" alt="Resultado da inferência" width="45%">
</p>

```bash
streamlit run src/app.py
```

### Reprodutibilidade e Orquestração no Google Colab
O notebook `notebooks/treino_colab.ipynb` automatiza todo o pipeline, desde o download dos dados, execução do ETL, treinamento dos modelos, até a geração dos resultados e inferência. Ele foi projetado para facilitar a reprodutibilidade e compartilhamento do projeto, permitindo que qualquer usuário execute todas as etapas sem necessidade de configuração local. Basta abrir o notebook no Colab, seguir as instruções e executar as células sequencialmente.

---

---

## 5. Histórico de Mudanças

| Versão | Data       | Mudanças principais |
|--------|------------|--------------------|
| 1.0.0  | 01/12/2025 | Entrega final, documentação completa, Streamlit, experiment tracking |
| 0.9.0  | 28/11/2025 | Ajustes finais no pipeline, integração Colab/local |
| 0.8.0  | 25/11/2025 | Implementação do app Streamlit |
| 0.7.0  | 20/11/2025 | Experiment tracking, organização dos artefatos |
| 0.6.0  | 15/11/2025 | Treinamento ResNet-50, comparação com CNN |
| 0.5.0  | 10/11/2025 | ETL robusto, separação dos dados |
| 0.4.0  | 05/11/2025 | Implementação da CNN do zero |
| 0.3.0  | 01/11/2025 | Estrutura inicial do projeto |

---

## 6. Distribuição de Responsabilidades

| Integrante                        | Responsabilidades principais |
|-----------------------------------|-----------------------------|
| Ana Beatriz Duarte Domingues      | ETL, documentação, testes   |
| Junior Rodrigues da Silva         | Modelos, experiment tracking, Streamlit |
| Carlos Emilio Castillo Estrada    | Colab, integração, validação|

---


## 7. Resultados e Métricas

Os resultados obtidos demonstram o impacto do uso de Transfer Learning em tarefas médicas:

- **CNN do zero:**
    - Acurácia: 0.82
    - Loss: 0.41
- **ResNet-50 (Transfer Learning):**
    - Acurácia: 0.89
    - Loss: 0.28

<p align="center">
<img src="experiments/grafico_acuracia.png" alt="Gráfico de acurácia" width="30%">
<img src="experiments/grafico_loss.png" alt="Gráfico de loss" width="30%">
<img src="assets/grafico_comparativo.png" alt="Gráfico comparativo CNN vs ResNet-50" width="30%">
</p>

**Avaliação dos resultados:**
O modelo ResNet-50 apresentou desempenho superior em acurácia e menor perda, evidenciando os benefícios do Transfer Learning em cenários com dados limitados. Os gráficos mostram a evolução do treinamento e a diferença entre as abordagens. Todos os artefatos, logs e gráficos estão disponíveis na pasta `experiments/` para consulta detalhada.

---

---

## 8. Decisões Técnicas

- Uso de TensorFlow/Keras para modelagem e treinamento.
- Separação clara entre scripts de ETL, treinamento e inferência.
- Experiment tracking via organização de artefatos e métricas.
- Streamlit para interface simples e acessível.
- Reprodutibilidade garantida por scripts e notebook Colab.
- Estrutura modular para facilitar manutenção e expansão.
- **Escolha do ResNet-50:** Optamos pelo ResNet-50 por ser um dos modelos mais consagrados em tarefas de classificação de imagens médicas, devido à sua profundidade, capacidade de generalização e uso eficiente de transfer learning. Isso nos permitiu obter resultados superiores com menos dados e tempo de treinamento, além de facilitar a reprodutibilidade.

---



## 10. Referências e Links Úteis

- [NIH Chest X-ray Dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC)
- [Paper ResNet](https://arxiv.org/abs/1512.03385)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Documentation](https://keras.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

## 11. Licença

<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1"><p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/"><a property="dct:title" rel="cc:attributionURL" href="https://github.com/agodoi/template">MODELO GIT FIAP</a> por <a rel="cc:attributionURL dct:creator" property="cc:attributionName" href="https://fiap.com.br">Fiap</a> está licenciado sobre <a href="http://creativecommons.org/licenses/by/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">Attribution 4.0 International</a>.</p>

---

## 2. Arquitetura do Pipeline

```mermaid
graph TD;
        Colab[Launcher (Colab)] --> ETL[src/etl.py];
        ETL --> Train[src/train.py];
        Train --> Experiments[experiments/ + Git Tracking];
        Train --> Model[models/model.h5];
        Model --> App[Streamlit (src/app.py)];
```

---

## 3. Estrutura de Pastas

```text
Fase4_CardioIA/
├── assets/                 # Logos e imagens para documentação
├── data/                   # Dados de treino/validação (após ETL)
│   ├── train/
│   └── validation/
├── experiments/            # Métricas, gráficos e artefatos de experimentos
├── models/                 # Modelos treinados (.h5)
├── notebooks/
│   └── treino_colab.ipynb  # Notebook orquestrador (Colab)
├── src/
│   ├── app.py              # Aplicação Streamlit de inferência
│   ├── data_preprocessing.py
│   ├── model_resnet.py
│   ├── model_simple_cnn.py
│   └── train.py            # Script principal de treino
└── README.md
```

---

## 4. Instruções de Uso

### Pré-requisitos
- Python 3.10+
- Instalar dependências: `pip install -r requirements.txt`
- Dados organizados em `data/train` e `data/validation` (após ETL)

### Executando o ETL
```bash
python src/data_preprocessing.py
```

### Treinando os modelos
```bash
python src/train.py
```

### Rodando o app de inferência
```bash
streamlit run src/app.py
```

### Reprodutibilidade no Colab
Executar o notebook `notebooks/treino_colab.ipynb` para orquestrar todo o pipeline.

---


## 6. Distribuição de Responsabilidades

| Integrante                        | Responsabilidades principais |
|-----------------------------------|-----------------------------|
| Ana Beatriz Duarte Domingues      | ETL, documentação, testes   |
| Junior Rodrigues da Silva         | Modelos, experiment tracking, Streamlit |
| Carlos Emilio Castillo Estrada    | Colab, integração, validação|

---

## 7. Resultados e Métricas

- **CNN do zero:**
    - Acurácia: 0.82
    - Loss: 0.41
- **ResNet-50 (Transfer Learning):**
    - Acurácia: 0.89
    - Loss: 0.28

Gráficos e artefatos disponíveis em `experiments/`.

---

## 8. Decisões Técnicas

- Uso de TensorFlow/Keras para modelagem e treinamento.
- Separação clara entre scripts de ETL, treinamento e inferência.
- Experiment tracking via organização de artefatos e métricas.
- Streamlit para interface simples e acessível.
- Reprodutibilidade garantida por scripts e notebook Colab.
- Estrutura modular para facilitar manutenção e expansão.

---

## 📋 Licença

<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1"><p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/"><a property="dct:title" rel="cc:attributionURL" href="https://github.com/agodoi/template">MODELO GIT FIAP</a> por <a rel="cc:attributionURL dct:creator" property="cc:attributionName" href="https://fiap.com.br">Fiap</a> está licenciado sobre <a href="http://creativecommons.org/licenses/by/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">Attribution 4.0 International</a>.</p>