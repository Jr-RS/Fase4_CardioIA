"""Script de treinamento do CardioIA com rastreamento automático de experimentos."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent))
    import auth  # type: ignore
    import data_preprocessing  # type: ignore
    import model_resnet  # type: ignore
    import model_simple_cnn  # type: ignore
    import utils_git  # type: ignore
else:  # pragma: no cover
    from . import auth, data_preprocessing, model_resnet, model_simple_cnn, utils_git


def _gerar_curvas(history) -> Figure:
    """Retorna figura Matplotlib com as curvas de loss e accuracy."""

    history_dict: Mapping[str, list[float]] = history.history

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    def _plot(ax, serie, label):
        if serie:
            ax.plot(range(1, len(serie) + 1), serie, label=label)

    _plot(axes[0], history_dict.get("loss", []), "Treino")
    _plot(axes[0], history_dict.get("val_loss", []), "Validação")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Épocas")
    axes[0].set_ylabel("Loss")
    if axes[0].lines:
        axes[0].legend()

    tre_acc = history_dict.get("accuracy") or history_dict.get("acc") or []
    val_acc = history_dict.get("val_accuracy") or history_dict.get("val_acc") or []
    _plot(axes[1], tre_acc, "Treino")
    _plot(axes[1], val_acc, "Validação")
    axes[1].set_title("Acurácia")
    axes[1].set_xlabel("Épocas")
    axes[1].set_ylabel("Acurácia")
    if axes[1].lines:
        axes[1].legend()

    fig.tight_layout()
    return fig


def _construir_metricas(
    history,
    params: Dict[str, float | int | str],
    modelo_path: Path,
) -> Dict[str, object]:
    """Prepara um dicionário serializável com métricas e histórico."""

    history_dict = {
        chave: [float(valor) for valor in valores]
        for chave, valores in history.history.items()
    }

    finais = {chave: valores[-1] for chave, valores in history_dict.items() if valores}

    val_loss = history_dict.get("val_loss") or []
    if val_loss:
        melhor_idx = min(range(len(val_loss)), key=val_loss.__getitem__)
        melhor_epoch = melhor_idx + 1
    else:
        melhor_epoch = len(next(iter(history_dict.values()), []))

    return {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        "params": params,
        "final_metrics": finais,
        "history": history_dict,
        "best_epoch": melhor_epoch,
        "model_artifact": modelo_path.name,
    }


def _criar_callbacks(checkpoint_path: Path) -> list:
    """Configura callbacks padrão utilizados durante o treinamento."""

    return [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ModelCheckpoint(filepath=os.fspath(checkpoint_path), monitor="val_loss", save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=1e-7),
    ]


def _exibir_download_colab(modelo_path: Path) -> None:
    """Gera link de download no Colab, se disponível."""

    if "google.colab" not in sys.modules:
        return

    try:
        from google.colab import files  # type: ignore

        print("[train] Disponibilizando download do modelo...")
        files.download(os.fspath(modelo_path))
    except Exception as exc:  # noqa: BLE001
        print(f"[train] Não foi possível gerar o link de download: {exc}")


def treinar(
    data_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    model_name: str,
    credenciais: Dict[str, str],
) -> None:
    """Executa o treinamento e registra o experimento correspondente."""

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Diretório de dados não encontrado: {data_dir}. Execute o ETL antes do treino."
        )

    models_dir = Path(__file__).resolve().parents[1] / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    treino_gen, valid_gen = data_preprocessing.configurar_geradores(
        diretorio_base=data_dir,
        batch_size=batch_size,
    )

    if model_name == "cnn":
        modelo = model_simple_cnn.construir_modelo(learning_rate=learning_rate)
    else:
        modelo = model_resnet.construir_modelo(learning_rate=learning_rate)

    checkpoint_path = models_dir / f"best_model_{model_name}.h5"

    history = modelo.fit(
        treino_gen,
        epochs=epochs,
        validation_data=valid_gen,
        callbacks=_criar_callbacks(checkpoint_path),
    )

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    modelo_path = models_dir / f"model_{model_name}.h5"
    modelo.save(os.fspath(modelo_path))
    print(f"[train] Modelo salvo em {modelo_path}")

    if model_name == "resnet":
        best_model_path = models_dir / "best_model.h5"
        source_best = checkpoint_path if checkpoint_path.exists() else modelo_path
        shutil.copy2(source_best, best_model_path)
        print(f"[train] Modelo principal atualizado em {best_model_path}")

    reports_dir = Path(__file__).resolve().parents[1] / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    figura = _gerar_curvas(history)
    curvas_path = reports_dir / f"training_curves_{model_name}.png"
    figura.savefig(curvas_path, dpi=150, bbox_inches="tight")
    print(f"[train] Curvas de treinamento salvas em {curvas_path}")

    params = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "model": model_name,
    }
    metricas = _construir_metricas(history, params, modelo_path)

    try:
        utils_git.registrar_experimento(
            metrics_dict=metricas,
            figures_dict={"training_curves": figura},
            credenciais=credenciais,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[train] Aviso: falha ao registrar experimento: {exc}")

    _exibir_download_colab(modelo_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treinamento CardioIA com rastreamento de experimentos")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.fspath(Path(__file__).resolve().parents[1] / "data"),
        help="Diretório contendo as pastas train/ e validation/",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Número de épocas para treinamento")
    parser.add_argument("--batch-size", type=int, default=32, help="Tamanho do batch")
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Taxa de aprendizado do otimizador"
    )
    parser.add_argument(
        "--model",
        choices=["resnet", "cnn"],
        default="resnet",
        help="Define qual arquitetura será treinada",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    data_dir = Path(args.data_dir)

    credenciais = auth.obter_credenciais()

    treinar(
        data_dir=data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        model_name=args.model,
        credenciais=credenciais,
    )


if __name__ == "__main__":
    main()
