"""Utilitários para versionar experimentos diretamente no repositório Git."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping, Optional, Union

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent))
    import auth  # type: ignore
else:  # pragma: no cover
    from . import auth


FigureLike = Union[Figure, Path, str]


def _run_git(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    """Executa um comando git retornando o resultado completo."""

    result = subprocess.run(
        ["git", *args],
        cwd=os.fspath(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Erro ao executar git {' '.join(args)}: {result.stderr.strip() or result.stdout.strip()}"
        )
    return result


def _sanitize_remote(url: str, token: str) -> Optional[str]:
    """Insere o token na URL remota se o esquema suportar."""

    if not url.startswith("https://"):
        return None

    prefix = "https://"
    return url.replace(prefix, f"{prefix}{token}@", 1)


def _guardar_metricas(destino: Path, metrics: Mapping[str, object]) -> Path:
    """Salva as métricas em um arquivo JSON."""

    metrics_path = destino / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return metrics_path


def _guardar_figuras(destino: Path, figuras: Mapping[str, FigureLike]) -> list[Path]:
    """Persiste cada figura como PNG na pasta do experimento."""

    salvos: list[Path] = []

    for nome, figura in figuras.items():
        if figura is None:
            continue

        alvo = destino / f"{nome}.png"

        if hasattr(figura, "savefig"):
            figura.savefig(alvo, dpi=150, bbox_inches="tight")
            plt.close(figura)  # Libera recursos do Matplotlib
        else:
            origem = Path(figura)
            if origem.suffix.lower() == ".h5":
                print(f"[utils_git] Ignorando {origem.name} (formato .h5 não permitido).")
                continue
            shutil.copy2(origem, alvo)

        salvos.append(alvo)

    return salvos


def registrar_experimento(
    metrics_dict: Mapping[str, object],
    figures_dict: Mapping[str, FigureLike],
    credenciais: Optional[Dict[str, str]] = None,
) -> Optional[Path]:
    """Salva artefatos e realiza commit + push automático do experimento."""

    repo_root = Path(__file__).resolve().parents[1]
    experiments_dir = repo_root / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    exp_dir = experiments_dir / f"exp_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=False)

    artefatos: list[Path] = []

    metrics_path = _guardar_metricas(exp_dir, metrics_dict)
    artefatos.append(metrics_path)

    artefatos.extend(_guardar_figuras(exp_dir, figures_dict))

    # Filtra arquivos pesados
    finais: list[Path] = []
    for arquivo in artefatos:
        if not arquivo.exists():
            continue
        if arquivo.stat().st_size > MAX_FILE_SIZE:
            print(f"[utils_git] Removendo {arquivo.name}: arquivo excede 50MB.")
            arquivo.unlink(missing_ok=True)
            continue
        finais.append(arquivo)

    if not finais:
        print("[utils_git] Nenhum artefato válido para registro.")
        return None

    credenciais = credenciais or auth.obter_credenciais()
    token = credenciais.get("GITHUB_TOKEN") if credenciais else None

    _run_git(["config", "user.name", "cardioia-bot"], cwd=repo_root)
    _run_git(["config", "user.email", "cardioia-bot@example.com"], cwd=repo_root)

    _run_git(["add", os.fspath(exp_dir.relative_to(repo_root))], cwd=repo_root)

    diff_cached = _run_git(["diff", "--cached", "--name-only"], cwd=repo_root)
    if not diff_cached.stdout.strip():
        print("[utils_git] Nenhuma alteração para commit.")
        return exp_dir

    _run_git(["commit", "-m", "Add experiment results"], cwd=repo_root)

    if not token:
        print("[utils_git] GITHUB_TOKEN ausente. Realize o push manualmente.")
        return exp_dir

    remote_url = _run_git(["remote", "get-url", "origin"], cwd=repo_root).stdout.strip()
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root).stdout.strip()

    auth_url = _sanitize_remote(remote_url, token)
    if not auth_url:
        print(
            "[utils_git] Remote não usa HTTPS ou token inválido. Push automático não realizado."
        )
        return exp_dir

    resultado = subprocess.run(
        ["git", "push", auth_url, f"HEAD:{branch}"],
        cwd=os.fspath(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if resultado.returncode != 0:
        raise RuntimeError("Falha no git push automático. Verifique o GITHUB_TOKEN e permissões.")

    print(f"[utils_git] Experimento registrado e enviado: {exp_dir}")
    return exp_dir


__all__ = ["registrar_experimento"]
