"""Modelo base CNN simples para comparação no CardioIA."""

from typing import Tuple

from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten, Input,
                                     MaxPooling2D)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def construir_modelo(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    learning_rate: float = 1e-3,
) -> Model:
    """Constrói uma CNN rasa para servir de baseline ao projeto."""

    entradas = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), activation="relu", padding="same")(entradas)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.5)(x)
    saida = Dense(1, activation="sigmoid")(x)

    modelo = Model(inputs=entradas, outputs=saida, name="CardioIA_CNN_Simples")

    modelo.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return modelo
