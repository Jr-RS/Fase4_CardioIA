"""Módulo de autenticação unificada para Kaggle e GitHub.

Prioriza credenciais armazenadas em ambientes gerenciados (Google Colab),
em seguida utiliza variáveis de ambiente ou um arquivo `.env` na raiz do
projeto. Caso não encontre, solicita a entrada manual do usuário.
"""

from __future__ import annotations

import getpass
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional

# Cache simples para evitar leituras repetidas do arquivo .env
_ENV_CACHE: Optional[Dict[str, str]] = None


def _ler_colab_secret(chave: str) -> Optional[str]:
    """Tenta ler a credencial a partir do Google Colab userdata."""

    try:
        from google.colab import userdata  # type: ignore
    except ImportError:
        return None

    try:
        valor = userdata.get(chave)
    except Exception:  # noqa: BLE001
        return None

    if valor:
        return str(valor).strip()
    return None


def _carregar_env() -> Dict[str, str]:
    """Carrega o arquivo .env manualmente para evitar dependências extras."""

    global _ENV_CACHE

    if _ENV_CACHE is not None:
        return _ENV_CACHE

    raiz_repo = Path(__file__).resolve().parents[1]
    env_path = raiz_repo / ".env"
    dados: Dict[str, str] = {}

    if env_path.exists():
        for linha in env_path.read_text(encoding="utf-8").splitlines():
            linha = linha.strip()
            if not linha or linha.startswith("#") or "=" not in linha:
                continue
            chave, valor = linha.split("=", 1)
            dados[chave.strip()] = valor.strip()

    _ENV_CACHE = dados
    return dados


def _prompt_interativo(chave: str) -> Optional[str]:
    """Solicita a credencial ao usuário respeitando ambientes não interativos."""

    if not sys.stdin or not sys.stdin.isatty():
        return None

    label = chave.replace("_", " ").title()
    if "KEY" in chave or "TOKEN" in chave:
        prompt = f"Informe {label}: "
        return getpass.getpass(prompt).strip() or None

    return input(f"Informe {label}: ").strip() or None


def obter_credenciais() -> Dict[str, str]:
    """Consolida credenciais de Kaggle e GitHub a partir das fontes disponíveis."""

    fontes = [
        _ler_colab_secret,
        lambda chave: os.environ.get(chave),
        lambda chave: _carregar_env().get(chave),
    ]

    chaves = ("KAGGLE_USERNAME", "KAGGLE_KEY", "GITHUB_TOKEN")
    credenciais: Dict[str, str] = {}

    for chave in chaves:
        valor: Optional[str] = None
        for fonte in fontes:
            try:
                valor = fonte(chave)
            except Exception:  # noqa: BLE001
                valor = None
            if valor:
                break

        if not valor:
            valor = _prompt_interativo(chave)

        if valor:
            credenciais[chave] = valor.strip()

    if not credenciais.get("KAGGLE_USERNAME") or not credenciais.get("KAGGLE_KEY"):
        raise RuntimeError(
            "Credenciais do Kaggle não encontradas. Configure .env ou segredos do Colab."
        )

    # Propaga para o ambiente em execução
    os.environ.setdefault("KAGGLE_USERNAME", credenciais["KAGGLE_USERNAME"])
    os.environ.setdefault("KAGGLE_KEY", credenciais["KAGGLE_KEY"])

    if credenciais.get("GITHUB_TOKEN"):
        os.environ.setdefault("GITHUB_TOKEN", credenciais["GITHUB_TOKEN"])
    else:
        print("[auth] Aviso: GITHUB_TOKEN ausente. Push automático ficará desabilitado.")

    return credenciais


def configurar_kaggle(credenciais: Optional[Dict[str, str]] = None) -> Path:
    """Cria o kaggle.json na pasta padrão com as credenciais fornecidas."""

    creds = credenciais or obter_credenciais()
    username = creds.get("KAGGLE_USERNAME")
    key = creds.get("KAGGLE_KEY")

    if not username or not key:
        raise RuntimeError("Credenciais do Kaggle incompletas.")

    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    kaggle_json = kaggle_dir / "kaggle.json"

    payload = {"username": username, "key": key}
    kaggle_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    try:
        os.chmod(kaggle_json, 0o600)
    except PermissionError:
        # Windows não suporta chmod 600, seguir em frente.
        pass

    print(f"[auth] kaggle.json configurado em {kaggle_json}")
    return kaggle_json


__all__ = ["obter_credenciais", "configurar_kaggle"]
