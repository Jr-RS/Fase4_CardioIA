"""Ferramentas de pré-processamento de dados para o projeto CardioIA."""

from pathlib import Path
from typing import Tuple

from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import DirectoryIterator, ImageDataGenerator


def configurar_geradores(
    diretorio_base: str | Path,
    batch_size: int = 32,
    target_size: tuple[int, int] = (224, 224),
) -> Tuple[DirectoryIterator, DirectoryIterator]:
    """Cria geradores de treino e validação prontos para a ResNet-50.

    Garante que os dados de treino recebam augmentation compatível com o cenário clínico
    enquanto os dados de validação passam apenas pelo pré-processamento do modelo.

    Args:
        diretorio_base: Caminho para o diretório contendo as pastas "train" e "validation".
        batch_size: Quantidade de amostras por batch.
        target_size: Dimensão final das imagens (altura, largura).

    Returns:
        Tupla com os geradores (treino, validacao).

    Raises:
        FileNotFoundError: Caso os diretórios esperados não existam.
    """

    base_path = Path(diretorio_base)
    treino_dir = base_path / "train"
    validacao_dir = base_path / "validation"

    if not treino_dir.exists():
        raise FileNotFoundError(f"Diretório de treino não encontrado: {treino_dir}")

    if not validacao_dir.exists():
        raise FileNotFoundError(f"Diretório de validação não encontrado: {validacao_dir}")

    # Augmentation moderado para refletir variações comuns nas radiografias de tórax.
    gerador_treino = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
    )

    gerador_validacao = ImageDataGenerator(
        preprocessing_function=preprocess_input,
    )

    fluxo_treino = gerador_treino.flow_from_directory(
        directory=str(treino_dir),
        target_size=target_size,
        batch_size=batch_size,
        class_mode="binary",
    )

    fluxo_validacao = gerador_validacao.flow_from_directory(
        directory=str(validacao_dir),
        target_size=target_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,
    )

    return fluxo_treino, fluxo_validacao
