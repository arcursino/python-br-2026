"""
setup_colab.py — bootstrap idempotente para Google Colab.

USO NO NOTEBOOK (célula 0, duas linhas):

    !curl -sSL https://raw.githubusercontent.com/arcursino/python-br-2026/main/setup_colab.py -o /tmp/s.py
    %run /tmp/s.py

Ou, se o repositório já foi clonado:

    %run setup_colab.py

O que faz, em ordem:
  1. clona (ou atualiza) o repositório em /content/python-br-2026
  2. instala o pacote em modo editável  →  `import driftkit` e CLI `driftkit`
  3. baixa os dados derivados do GitHub Releases (~180 MB, não os 14 GB da Kaggle)
  4. imprime um diagnóstico linha a linha
  5. liga %autoreload para que edições em src/ tenham efeito sem reiniciar

É seguro rodar quantas vezes quiser. Se o runtime do Colab cair, rode de novo:
volta ao estado inicial em ~90 segundos.
"""

from __future__ import annotations

import importlib
import importlib.metadata as md
import os
import shutil
import subprocess
import sys
from pathlib import Path

# =============================================================================
#  CONFIGURAÇÃO — ajuste estas três linhas para o seu repositório
# =============================================================================
GH_USER = "arcursino"
GH_REPO = "python-br-2026"
TAG_DADOS = "dados-v1"  # tag do Release que contém os .parquet

REPO_URL = f"https://github.com/{GH_USER}/{GH_REPO}.git"
BASE_RELEASE = f"https://github.com/{GH_USER}/{GH_REPO}/releases/download/{TAG_DADOS}"

RAIZ = Path("/content") / GH_REPO if Path("/content").exists() else Path.cwd()
DATA = RAIZ / "data"

# Arquivos de dados: (nome, obrigatório_para, tamanho_aprox_mb)
ARQUIVOS_DADOS = [
    ("detector_config_referencia_v1.json", "fallback da Parte I", 0.01),
    ("meta_bosch.parquet", "Parte II — eixo temporal", 48),
    ("bosch_num_160feats.parquet", "Parte II — features", 132),
]

IN_COLAB = "google.colab" in sys.modules


# =============================================================================
#  utilitários
# =============================================================================
def sh(cmd: str, *, critico: bool = True) -> subprocess.CompletedProcess:
    """Roda um comando de shell mostrando saída só em caso de erro."""
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if r.returncode and critico:
        print(f"\n❌ comando falhou: {cmd}\n")
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        raise SystemExit(
            "\nSETUP INTERROMPIDO. Levante o cartão vermelho 🔴 e chame o monitor.\n"
            "Não avance para as próximas células."
        )
    return r


def baixar(url: str, destino: Path, *, mb: float = 0) -> bool:
    """Baixa com barra de progresso. Devolve False se falhar (não aborta)."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_suffix(destino.suffix + ".parcial")
    print(f"   baixando {destino.name} (~{mb:.0f} MB)...", end=" ", flush=True)
    r = sh(f'curl -fsSL --retry 3 --retry-delay 2 "{url}" -o "{tmp}"', critico=False)
    if r.returncode or not tmp.exists() or tmp.stat().st_size < 100:
        tmp.unlink(missing_ok=True)
        print("falhou ⚠️")
        return False
    tmp.rename(destino)
    print("ok ✅")
    return True


# =============================================================================
#  1. código
# =============================================================================
print("=" * 62)
print("  PyBR 2026 · Rumo ao Desconhecido: Tratando Drift em ML")
print("  setup do ambiente")
print("=" * 62)

if IN_COLAB:
    if (RAIZ / ".git").exists():
        print("\n[1/4] repositório já presente — atualizando...")
        sh(f"git -C {RAIZ} pull -q --ff-only", critico=False)
    else:
        print("\n[1/4] clonando repositório...")
        if RAIZ.exists():
            shutil.rmtree(RAIZ)
        sh(f"git clone --depth 1 -q {REPO_URL} {RAIZ}")
    os.chdir(RAIZ)
else:
    print("\n[1/4] fora do Colab — usando diretório atual como raiz")
    os.chdir(RAIZ)

# =============================================================================
#  2. instalação
# =============================================================================
print("[2/4] instalando driftkit (modo editável) + dependências...")
extras = "[dev,mercado]" if IN_COLAB else "[dev]"
sh(f'pip install -q -e ".{extras}"')

# cinto e suspensório: garante o src no path mesmo se o editable falhar
src = str(RAIZ / "src")
if src not in sys.path:
    sys.path.insert(0, src)

# =============================================================================
#  3. dados
# =============================================================================
print("[3/4] verificando dados derivados...")
DATA.mkdir(parents=True, exist_ok=True)

# Estratégia de banda: só o essencial agora. O Bosch (grande) fica para o
# intervalo técnico, via driftkit.data.baixar_bosch() — evita 40 pessoas
# puxando 180 MB no mesmo segundo.
ESSENCIAL = {"detector_config_referencia_v1.json"}
for nome, _para, mb in ARQUIVOS_DADOS:
    destino = DATA / nome
    if destino.exists():
        continue
    if nome in ESSENCIAL or not IN_COLAB:
        baixar(f"{BASE_RELEASE}/{nome}", destino, mb=mb)

# =============================================================================
#  4. diagnóstico
# =============================================================================
print("[4/4] diagnóstico\n")

VERSOES_ESPERADAS = {
    "numpy": "1.26|2.",
    "pandas": "2.",
    "scipy": "1.",
    "scikit-learn": "1.",
    "pyarrow": "1",
    "typer": "0.",
    "pytest": "8.",
    "evidently": "0.",
    "river": "0.",
}

print(f"  {'pacote':<26}{'':<3}{'versão'}")
print("  " + "-" * 46)
_falhas: list[str] = []
for pkg, prefixos in VERSOES_ESPERADAS.items():
    try:
        v = md.version(pkg)
        ok = any(v.startswith(p) for p in prefixos.split("|"))
        icone = "✅" if ok else "⚠️"
        if not ok:
            _falhas.append(f"{pkg} {v} (esperado ~{prefixos})")
    except md.PackageNotFoundError:
        v, icone = "AUSENTE", "❌"
        # evidently/river/alibi são opcionais: ausência é aviso, não erro
        (_falhas if pkg not in {"evidently", "river"} else []).append(f"{pkg} ausente")
    print(f"  {pkg:<26}{icone:<3}{v}")

print()
print(f"  {'dado':<36}{'':<3}{'tamanho'}")
print("  " + "-" * 52)
for nome, para, _mb in ARQUIVOS_DADOS:
    f = DATA / nome
    if f.exists():
        print(f"  {nome:<36}{'✅':<3}{f.stat().st_size / 1e6:>7.1f} MB")
    else:
        pendente = nome not in ESSENCIAL
        print(f"  {nome:<36}{'⏳' if pendente else '❌':<3}{'(' + para + ')'}")

print()
try:
    import src.driftkit as driftkit  # noqa: F401

    importlib.reload(driftkit)
    from src.driftkit.detectors import psi  # noqa: F401

    print(f"  {'import driftkit':<36}{'✅':<3}v{driftkit.__version__}")
except Exception as e:  # noqa: BLE001
    print(f"  {'import driftkit':<36}{'❌':<3}{type(e).__name__}: {e}")
    _falhas.append("import driftkit")

r = sh("driftkit --help", critico=False)
if r.returncode == 0:
    print(f"  {'CLI driftkit':<36}{'✅':<3}disponível no PATH")
else:
    print(f"  {'CLI driftkit':<36}{'⚠️':<3}use `python -m driftkit` no lugar")

# suíte rápida: se ela passa, o ambiente está bom de verdade
r = sh("pytest -q -m 'not lento and not requer_bosch and not mercado' --no-header", critico=False)
ultima = (r.stdout or "").strip().splitlines()
print(f"  {'suíte de testes':<36}{'✅' if r.returncode == 0 else '⚠️':<3}"
      f"{ultima[-1] if ultima else 'sem saída'}")

# =============================================================================
#  autoreload + variáveis de conveniência
# =============================================================================
try:
    ip = get_ipython()  # type: ignore[name-defined]  # noqa: F821
    ip.run_line_magic("load_ext", "autoreload")
    ip.run_line_magic("autoreload", "2")
except Exception:  # noqa: BLE001
    pass

print("\n" + "=" * 62)
if _falhas:
    print("  ⚠️  ATENÇÃO — pendências encontradas:")
    for f in _falhas:
        print(f"       • {f}")
    print("\n  Levante o cartão 🔴 AGORA. Não avance sozinho.")
else:
    print("  ✅ AMBIENTE PRONTO. Levante o cartão 🟢.")
print(f"\n  RAIZ = {RAIZ}")
print(f"  DATA = {DATA}")
print("=" * 62)

# expostos para as células seguintes
RAIZ_REPO, DATA_DIR = RAIZ, DATA
