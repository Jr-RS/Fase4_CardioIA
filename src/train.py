"""Orquestra o pipeline de treinamento do CardioIA."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent))
    import data_preprocessing  # type: ignore
    import model_resnet  # type: ignore
else:
    from . import data_preprocessing, model_resnet


def _assegurar_diretorio(caminho: str | Path) -> Path:
    """Garante que o diretório exista antes de salvar artefatos."""

    path_obj = Path(caminho)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def _plotar_curvas(history, destino: Path) -> None:
    """Cria gráfico de loss e accuracy, salvando em disco."""

    history_dict: dict[str, list[float]] = history.history
    epochs = range(1, len(history_dict["loss"]) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history_dict["loss"], label="Treino")
    plt.plot(epochs, history_dict["val_loss"], label="Validação")
    plt.title("Loss")
    plt.xlabel("Épocas")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history_dict["accuracy"], label="Treino")
    plt.plot(epochs, history_dict["val_accuracy"], label="Validação")
    plt.title("Acurácia")
    plt.xlabel("Épocas")
    plt.ylabel("Acurácia")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.fspath(destino))
    plt.close()


def treinar_modelo(
    data_dir: str | Path,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
) -> Optional[object]:
    """Executa o treinamento completo integrando geradores, modelo e callbacks."""

    base_path = Path(data_dir)
    models_dir = _assegurar_diretorio(base_path.parent / "models")
    reports_dir = _assegurar_diretorio(base_path.parent / "reports")

    treino_gen, valid_gen = data_preprocessing.configurar_geradores(
        diretorio_base=base_path,
        batch_size=batch_size,
    )

    modelo = model_resnet.construir_modelo(learning_rate=learning_rate)

    checkpoint_path = models_dir / "best_model.h5"

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        ),
        ModelCheckpoint(
            filepath=os.fspath(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=5,
            min_lr=1e-7,
        ),
    ]

    history = modelo.fit(
        treino_gen,
        epochs=epochs,
        validation_data=valid_gen,
        callbacks=callbacks,
    )

    grafico_path = reports_dir / "training_curves.png"
    _plotar_curvas(history, grafico_path)

    return history


if __name__ == "__main__":
    DATA_DIR = Path(__file__).resolve().parents[1] / "data"

    try:
        treinar_modelo(data_dir=DATA_DIR, epochs=1, batch_size=8)
        print("Treinamento quick-check concluído.")
    except FileNotFoundError as exc:
        print(f"Quick-check não executado: {exc}")
    except Exception as exc:  # noqa: BLE001
        print(f"Quick-check falhou: {exc}")
