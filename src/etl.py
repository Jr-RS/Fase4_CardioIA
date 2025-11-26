"""Script de ETL para preparar o dataset do CardioIA."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Tuple
from zipfile import ZipFile

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


def _localizar_splits(base: Path) -> Tuple[Path, Path]:
    """Encontra diretórios de treino e validação dentro da pasta extraída."""

    val_aliases = {"validation", "val", "valid"}

    for raiz, dirs, _ in os.walk(base):
        nomes_normalizados = {nome.lower(): nome for nome in dirs}
        if "train" in nomes_normalizados:
            for alias in val_aliases:
                if alias in nomes_normalizados:
                    train_dir = Path(raiz) / nomes_normalizados["train"]
                    val_dir = Path(raiz) / nomes_normalizados[alias]
                    return train_dir.resolve(), val_dir.resolve()

    raise FileNotFoundError(
        "Estrutura esperada não encontrada. Certifique-se de que o dataset "
        "contenha pastas 'train' e 'validation'."
    )


def _sincronizar_origem(origem: Path, destino: Path) -> None:
    """Move o conteúdo da origem para o destino, substituindo se necessário."""

    if destino.exists():
        shutil.rmtree(destino)

    destino.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.move(str(origem), str(destino))
    except shutil.Error:
        shutil.copytree(origem, destino)


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

        train_src, val_src = _localizar_splits(downloads_dir)

        repo_root = Path(__file__).resolve().parents[1]
        data_dir = repo_root / "data"

        _sincronizar_origem(train_src, data_dir / "train")
        _sincronizar_origem(val_src, data_dir / "validation")

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
