"""Aplicação Streamlit para inferência do modelo CardioIA."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model


@st.cache_resource
def carregar_modelo():
    """Carrega o modelo treinado a partir do diretório models."""

    modelos_dir = Path(__file__).resolve().parents[1] / "models"
    candidatos = [
        modelos_dir / "model.h5",
        modelos_dir / "best_model.h5",
        modelos_dir / "model_resnet.h5",
    ]

    for caminho_modelo in candidatos:
        if caminho_modelo.exists():
            return load_model(caminho_modelo)

    st.error("Modelo não encontrado. Execute o pipeline de treinamento primeiro.")
    return None


def processar_imagem(imagem: Image.Image) -> np.ndarray:
    """Prepara a imagem no formato aceito pela ResNet-50."""

    imagem_redimensionada = imagem.convert("RGB").resize((224, 224))
    array_imagem = np.array(imagem_redimensionada, dtype="float32")
    array_imagem = np.expand_dims(array_imagem, axis=0)
    array_imagem = preprocess_input(array_imagem)
    return array_imagem


def principal():
    st.set_page_config(page_title="CardioIA", layout="centered")
    st.title("CardioIA - Sistema de Apoio ao Diagnóstico")

    # Sidebar removida para interface mais limpa

    st.write(
        "Carregue uma radiografia de tórax para que o CardioIA analise sinais de cardiomegalia."
    )

    modelo = carregar_modelo()
    if modelo is None:
        return

    arquivo = st.file_uploader(
        "Envie uma radiografia de tórax (PNG/JPG)",
        type=["png", "jpg", "jpeg"],
    )

    if arquivo is None:
        return

    imagem = Image.open(arquivo)
    st.image(imagem, caption="Imagem carregada", use_column_width=True)

    if st.button("Analisar Exame"):
        entrada = processar_imagem(imagem)
        probabilidade = float(modelo.predict(entrada)[0][0])

        classe = "Possível Cardiomegalia" if probabilidade > 0.5 else "Normal"
        st.subheader(classe)

        st.metric(
            label="Probabilidade de Cardiomegalia",
            value=f"{probabilidade * 100:.2f}%",
        )

        st.progress(min(max(probabilidade, 0.0), 1.0))

        st.warning(
            "Este é um protótipo acadêmico. Não substitui diagnóstico médico.",
        )


if __name__ == "__main__":
    principal()
