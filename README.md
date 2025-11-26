# CardioIA - Fase 4: Pipeline de Visão Computacional e MLOps

## Sobre
CardioIA é um sistema de apoio ao diagnóstico de cardiomegalia em radiografias de tórax. O pipeline compara uma CNN baseline construída do zero com um modelo de Transfer Learning baseado na ResNet-50, empregando scripts dedicados de ETL, treinamento parametrizado e rastreamento automático de experimentos versionados no Git.

## Arquitetura do Pipeline
```
Launcher (Colab)
      ↓
ETL Script (src/etl.py)
      ↓
Train Script (src/train.py)
      ↓
Git Tracking (experiments/ + push automatizado)
```
O notebook `notebooks/launcher.ipynb` atua como orquestrador: prepara o ambiente, executa o ETL via Kaggle e dispara o treinamento, que ao final registra métricas, gráficos e metadados diretamente no repositório através do módulo `src/utils_git.py`.

## Instalação e Execução (O Guia de Ouro)

### Opção A — Google Colab (Recomendada)
1. Abra `notebooks/launcher.ipynb` no GitHub e clique no botão **Open in Colab**.
2. No Colab, configure os *Secrets* `KAGGLE_USERNAME`, `KAGGLE_KEY` e `GITHUB_TOKEN` em *Settings → Secrets*.
3. Execute todas as células do notebook para clonar o repositório, instalar dependências, executar o ETL e iniciar o treinamento.

### Opção B — Localmente (VS Code)
1. Configure o arquivo `.env` na raiz utilizando o template disponibilizado.
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
3. Execute o ETL (download e organização dos dados):
   ```bash
   python src/etl.py
   ```
4. Treine o modelo (parâmetros opcionais como `--epochs` e `--batch-size`):
   ```bash
   python src/train.py
   ```
5. Inicie a interface de inferência:
   ```bash
   streamlit run src/app.py
   ```

## Estrutura de Pastas
- `src/`: módulos de autenticação (`auth.py`), ETL (`etl.py`), treinamento (`train.py`), modelos e utilitários de versionamento (`utils_git.py`).
- `notebooks/`: notebooks de apoio; `launcher.ipynb` é o ponto de entrada recomendado para execução em nuvem.
- `experiments/`: histórico versionado de métricas e gráficos gerados a cada execução de treino.
- `models/`: artefatos de modelos `.h5` salvos localmente (não versionados em git).

## Equipe
Grupo 30 — Ana (Documentação), Carlos (Frontend/EDA), Junior (Tech Lead/Modelagem).
