"""Script de ETL para preparar o dataset do CardioIA."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict, Tuple
from zipfile import ZipFile

import pandas as pd
from sklearn.model_selection import train_test_split

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent))
    import auth  # type: ignore
else:  # pragma: no cover
    from . import auth

DATASET_DEFAULT = "khanfashee/nih-chest-x-ray-14-224x224-resized"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def _extrair_arquivos(raiz: Path) -> None:
    """Extrai arquivos ZIP encontrados recursivamente."""

    for arquivo in list(raiz.rglob("*.zip")):
        destino = arquivo.parent
        with ZipFile(arquivo) as zip_ref:
            zip_ref.extractall(destino)
        arquivo.unlink(missing_ok=True)


def _encontrar_csv(base: Path) -> Path:
    """Localiza o arquivo Data_Entry_2017.csv dentro dos downloads."""

    for caminho in base.rglob("Data_Entry_2017.csv"):
        return caminho
    raise FileNotFoundError(
        "Arquivo 'Data_Entry_2017.csv' não encontrado no dataset baixado."
    )


def _amostrar_registros(df: pd.DataFrame, rotulo: str, n_amostras: int) -> pd.DataFrame:
    """Seleciona uma quantidade fixa de amostras para o rótulo informado."""

    if df.empty:
        raise ValueError(f"Nenhum registro encontrado para a classe '{rotulo}'.")

    quantidade = min(len(df), n_amostras)
    if quantidade < n_amostras:
        print(
            f"[etl] Aviso: apenas {quantidade} amostras disponíveis para '{rotulo}'."
        )

    return df.sample(n=quantidade, random_state=42).assign(label=rotulo)


def _indexar_imagens(base: Path) -> Dict[str, Path]:
    """Cria um índice nome do arquivo -> caminho completo para as imagens."""

    indice: Dict[str, Path] = {}
    for item in base.rglob("*"):
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS:
            indice.setdefault(item.name, item)

    if not indice:
        raise FileNotFoundError(
            "Nenhuma imagem foi encontrada no dataset baixado após a extração."
        )

    return indice


def _preparar_splits(downloads_dir: Path, data_dir: Path) -> Tuple[Path, Path]:
    """Filtra, amostra, divide e copia as imagens para treino/validação."""

    csv_path = _encontrar_csv(downloads_dir)
    df = pd.read_csv(csv_path)

    df_cardiomegaly = df[df["Finding Labels"] == "Cardiomegaly"]
    df_normal = df[df["Finding Labels"] == "No Finding"]

    cardio_amostra = _amostrar_registros(df_cardiomegaly, "cardiomegaly", 1000)
    normal_amostra = _amostrar_registros(df_normal, "normal", 1000)

    dataset_balanceado = pd.concat([cardio_amostra, normal_amostra], ignore_index=True)

    treino_df, validacao_df = train_test_split(
        dataset_balanceado,
        test_size=0.2,
        random_state=42,
        stratify=dataset_balanceado["label"],
    )

    if data_dir.exists():
        shutil.rmtree(data_dir)

    train_dir = data_dir / "train"
    validation_dir = data_dir / "validation"

    for base_dir in (train_dir, validation_dir):
        for classe in ("cardiomegaly", "normal"):
            (base_dir / classe).mkdir(parents=True, exist_ok=True)

    indice_imagens = _indexar_imagens(downloads_dir)
    faltantes = 0

    for split_nome, frame in (("train", treino_df), ("validation", validacao_df)):
        for _, linha in frame.iterrows():
            imagem = linha["Image Index"]
            classe = linha["label"]
            origem = indice_imagens.get(imagem)
            if origem is None:
                faltantes += 1
                print(f"[etl] Aviso: imagem não encontrada '{imagem}'.")
                continue

            destino = data_dir / split_nome / classe / imagem
            shutil.copy2(origem, destino)

    if faltantes:
        print(
            f"[etl] Aviso: {faltantes} imagens não foram copiadas por ausência no pacote baixado."
        )

    return train_dir, validation_dir


def _contar_imagens(diretorio: Path) -> Counter[str]:
    """Conta os arquivos de imagem por classe."""

    contagem: Counter[str] = Counter()
    for classe in sorted(diretorio.iterdir()):
        if not classe.is_dir():
            continue
        total = sum(
            1 for arquivo in classe.iterdir() if arquivo.is_file() and arquivo.suffix.lower() in IMAGE_EXTENSIONS
        )
        contagem[classe.name] = total
    return contagem


def _imprimir_estatisticas(data_dir: Path) -> None:
    """Exibe a distribuição de classes para treino e validação."""

    for split in ("train", "validation"):
        split_dir = data_dir / split
        if not split_dir.exists():
            print(f"[etl] Aviso: diretório '{split}' não encontrado.")
            continue

        contagem = _contar_imagens(split_dir)
        total = sum(contagem.values())
        print(f"[etl] {split} -> {total} imagens")
        for classe, quantidade in contagem.items():
            percentual = (quantidade / total * 100) if total else 0.0
            print(f"    - {classe}: {quantidade} ({percentual:.2f}%)")


def executar_etl() -> None:
    """Pipeline completo: autentica, baixa, organiza e reporta estatísticas."""

    credenciais = auth.obter_credenciais()
    auth.configurar_kaggle(credenciais)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependência externa
        raise ImportError(
            "O pacote 'kaggle' é obrigatório para executar o ETL. "
            "Instale-o com 'pip install kaggle'."
        ) from exc

    dataset_alvo = os.environ.get("CARDIOIA_KAGGLE_DATASET", DATASET_DEFAULT)
    print(f"[etl] Baixando dataset: {dataset_alvo}")

    api = KaggleApi()
    api.authenticate()

    temp_dir = Path(tempfile.mkdtemp(prefix="cardioia_"))
    downloads_dir = temp_dir / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)

    try:
        api.dataset_download_files(dataset_alvo, path=str(downloads_dir), unzip=True, quiet=False)
        _extrair_arquivos(downloads_dir)

        repo_root = Path(__file__).resolve().parents[1]
        data_dir = repo_root / "data"

        _preparar_splits(downloads_dir, data_dir)

        _imprimir_estatisticas(data_dir)
        print("[etl] ETL concluído com sucesso.")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    try:
        executar_etl()
    except Exception as exc:  # noqa: BLE001
        print(f"[etl] Falha no ETL: {exc}")
        raise
