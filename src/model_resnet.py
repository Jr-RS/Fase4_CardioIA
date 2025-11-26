"""Definição do modelo ResNet50 com cabeça customizada para o CardioIA."""

from typing import Tuple

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import (Dense, Dropout, GlobalAveragePooling2D,
                                     Input)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def construir_modelo(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    learning_rate: float = 1e-4,
) -> Model:
    """Monta um modelo de transferência de aprendizado baseado na ResNet50.

    Args:
        input_shape: Dimensão das imagens de entrada (altura, largura, canais).
        learning_rate: Taxa de aprendizado para o otimizador Adam.

    Returns:
        Instância compilada de `tensorflow.keras.Model` pronta para treinamento.
    """

    entradas = Input(shape=input_shape)
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_tensor=entradas,
    )

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    saidas = Dense(1, activation="sigmoid")(x)

    modelo = Model(inputs=entradas, outputs=saidas, name="CardioIA_ResNet50")

    modelo.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", "Precision", "Recall"],
    )

    return modelo
