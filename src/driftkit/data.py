"""
driftkit.data — carga de dados com cache local e leitura direta por URL.

Caminho no repositório: src/driftkit/data.py

RESPONDENDO À PERGUNTA "consigo ler o parquet direto da URL, sem baixar?"
=========================================================================
Sim. `pandas.read_parquet` aceita URL http(s) desde que `fsspec` + `requests`
estejam instalados (ambos estão no Colab e nas dependências deste pacote):

    df = pd.read_parquet("https://.../bosch_num_160feats.parquet")

E funciona melhor do que parece: se o servidor suporta HTTP Range (o CDN do
GitHub Releases suporta), o pyarrow lê **só as colunas e os row-groups
pedidos**, sem transferir o arquivo inteiro:

    df = pd.read_parquet(URL, columns=["Id", "L3_S36_F3939"])   # baixa ~2 MB

MAS — e é por isso que este módulo existe — há três armadilhas em sala de aula:

  1. SEM CACHE. Cada `read_parquet(URL)` refaz o download. Numa aula com 15
     leituras, isso é 15 × 130 MB por pessoa. O wifi da sala morre.
  2. ONDE HOSPEDAR IMPORTA:
       • repositório git normal → limite de 100 MB por arquivo. Não serve.
       • Git LFS               → serve, mas a cota de banda gratuita
                                 (1 GB/mês) estoura com ~8 participantes.
       • GitHub RELEASES       → ✅ até 2 GB por asset, CDN, banda ilimitada,
                                 suporta Range. É a escolha certa.
       • Hugging Face Hub      → ✅ também excelente, e feito para datasets.
  3. `raw.githubusercontent.com` tem rate limit agressivo e responde 404 para
     arquivos > 100 MB.

Regra prática deste módulo: **baixa uma vez, lê do disco depois.**
"""

from __future__ import annotations

import json
import os
import subprocess
from functools import lru_cache
from pathlib import Path

import pandas as pd

__all__ = [
    "raiz",
    "dir_dados",
    "caminho_ou_url",
    "carregar_parquet",
    "baixar_bosch",
    "carregar_meta_bosch",
    "carregar_features_bosch",
    "URL_BASE_RELEASE",
]

# =============================================================================
#  CONFIGURAÇÃO — ajuste para o seu repositório
# =============================================================================
# Estes três valores TÊM que casar com os do setup_colab.py. Ficaram como
# placeholder de template ("SEU_USUARIO/pybr2026-drift") por uma versão, e o
# sintoma foi um 404 no meio do notebook — com uma URL que nem existe. O
# setup estava certo; este módulo tem as suas próprias constantes.
GH_USER = os.environ.get("DRIFTKIT_GH_USER", "arcursino")
GH_REPO = os.environ.get("DRIFTKIT_GH_REPO", "python-br-2026")
TAG_DADOS = os.environ.get("DRIFTKIT_TAG_DADOS", "dados-v1")

URL_BASE_RELEASE = f"https://github.com/{GH_USER}/{GH_REPO}/releases/download/{TAG_DADOS}"

ARQ_META = "meta_bosch.parquet"
ARQ_FEATS = "bosch_num_160feats.parquet"
ARQ_CFG_V1 = "detector_config_referencia_v1.json"


# =============================================================================
#  localização
# =============================================================================
@lru_cache(maxsize=1)
def raiz() -> Path:
    """Raiz do repositório, funcione onde funcionar (Colab, local, CI)."""
    if (env := os.environ.get("DRIFTKIT_RAIZ")):
        return Path(env)
    p = Path(__file__).resolve()
    for pai in p.parents:
        if (pai / "pyproject.toml").exists():
            return pai
    for cand in (Path("/content") / GH_REPO, Path.cwd()):
        if cand.exists():
            return cand
    return Path.cwd()


def dir_dados() -> Path:
    d = raiz() / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def caminho_ou_url(nome: str, *, preferir_local: bool = True) -> str:
    """Devolve o caminho local se o arquivo existir; senão, a URL do Release.

    É o que permite ao notebook ser agnóstico:

        df = pd.read_parquet(caminho_ou_url("meta_bosch.parquet"))

    funciona com o arquivo em cache OU direto da internet, sem `if`.
    """
    local = dir_dados() / nome
    if preferir_local and local.exists():
        return str(local)
    return f"{URL_BASE_RELEASE}/{nome}"


# =============================================================================
#  download com cache
# =============================================================================
def _baixar(nome: str, *, forcar: bool = False, silencioso: bool = False) -> Path:
    destino = dir_dados() / nome
    if destino.exists() and not forcar:
        if not silencioso:
            print(f"✅ {nome} já em cache ({destino.stat().st_size / 1e6:.1f} MB)")
        return destino

    url = f"{URL_BASE_RELEASE}/{nome}"
    tmp = destino.with_suffix(destino.suffix + ".parcial")
    if not silencioso:
        print(f"⬇️  baixando {nome}...")
    r = subprocess.run(
        f'curl -fsSL --retry 3 --retry-delay 2 "{url}" -o "{tmp}"',
        shell=True, text=True, capture_output=True,
    )
    if r.returncode or not tmp.exists():
        tmp.unlink(missing_ok=True)
        # 404 e "sem internet" pedem ações DIFERENTES, e a mensagem tem que
        # dizer qual é qual — senão o participante tenta de novo três vezes
        # contra um asset que nunca foi publicado.
        erro = (r.stderr or "")
        e404 = "404" in erro
        diag = (
            f"o asset `{nome}` NÃO existe no Release `{TAG_DADOS}` de\n"
            f"  {GH_USER}/{GH_REPO} (HTTP 404). Isto não é problema da sua\n"
            "  conexão: o arquivo não foi publicado. Quem mantém o repositório\n"
            "  precisa anexá-lo ao Release."
            if e404 else
            "download interrompido (rede, proxy ou rate limit). Tente de novo."
        )
        raise RuntimeError(
            f"falha ao baixar {nome}.\n"
            f"  url: {url}\n"
            f"  DIAGNÓSTICO: {diag}\n"
            f"  {erro[-300:]}\n\n"
            "Plano B: peça o pendrive ao monitor e suba o arquivo pelo painel\n"
            f"de arquivos do Colab (📁 na lateral) para: {dir_dados()}"
        )
    tmp.rename(destino)
    if not silencioso:
        print(f"✅ {nome} ({destino.stat().st_size / 1e6:.1f} MB)")
    return destino


def baixar_bosch(*, forcar: bool = False) -> dict[str, Path]:
    """Baixa os dois parquets derivados do Bosch (~180 MB no total).

    Chame isto no INTERVALO TÉCNICO, não na célula 0: os dados do Bosch só são
    usados a partir do Bloco II-B (minuto ~70). Escalonar o download evita 40
    pessoas puxando 180 MB no mesmo segundo.
    """
    print("Baixando dados derivados do Bosch (~180 MB).")
    print("Isto substitui 14 GB da Kaggle + ~30 min de pré-processamento.")
    print("O código que os produziu está em scripts/preprocess_bosch.py.\n")
    return {
        "meta": _baixar(ARQ_META, forcar=forcar),
        "features": _baixar(ARQ_FEATS, forcar=forcar),
    }


# =============================================================================
#  leitura
# =============================================================================
def carregar_parquet(
    nome: str,
    *,
    columns: list[str] | None = None,
    filters=None,
    cachear: bool = True,
) -> pd.DataFrame:
    """Lê um parquet do cache local, baixando na primeira vez.

    Parameters
    ----------
    cachear : bool
        Se False, lê DIRETO da URL sem gravar em disco. Útil para demonstrar
        em aula que a leitura remota funciona — e para ler poucas colunas de um
        arquivo grande sem gastar disco:

            carregar_parquet(ARQ_FEATS, columns=["Id"], cachear=False)

        Com `columns`, o pyarrow usa HTTP Range e transfere só o necessário.
    """
    if cachear:
        caminho: str | Path = _baixar(nome, silencioso=True)
    else:
        caminho = f"{URL_BASE_RELEASE}/{nome}"
    return pd.read_parquet(caminho, columns=columns, filters=filters)


def carregar_meta_bosch(**kw) -> pd.DataFrame:
    """Metadados por peça: Id, Response, eixo temporal, presença por estação.

    Uma linha por peça, ~1.18 M linhas. É o artefato central da Parte II:
    o TEMPO, que o dataset original não fornece.
    """
    meta = carregar_parquet(ARQ_META, **kw)
    if "semana" in meta.columns:
        meta["semana"] = meta["semana"].astype("int32")
    return meta


def carregar_features_bosch(**kw) -> pd.DataFrame:
    """As 160 features numéricas selecionadas por cobertura, em float32."""
    return carregar_parquet(ARQ_FEATS, **kw)


def carregar_contrato_v1() -> dict:
    """Contrato de referência da Parte I (fallback).

    A ponte Parte I → Parte II com degradação graciosa: se o participante
    fechou o Colab entre os blocos, ele NÃO fica travado. Usa o contrato de
    referência e é AVISADO de que os limiares são da execução do instrutor.
    """
    # importado, não digitado: era a segunda cópia da string e a primeira
    # fonte de divergência silenciosa entre os dois módulos.
    from .state import NOME_PADRAO

    proprio = dir_dados() / NOME_PADRAO
    if proprio.exists():
        cfg = json.loads(proprio.read_text(encoding="utf-8"))
        print(f"contrato: {cfg['versao']} — origem: SUA execução do Bloco II-A ✅")
        return cfg

    caminho = _baixar(ARQ_CFG_V1, silencioso=True)
    cfg = json.loads(caminho.read_text(encoding="utf-8"))
    print(f"contrato: {cfg['versao']} — origem: ⚠️  FALLBACK (execução de referência, seed 7)")
    print(
        "  Tudo vai funcionar, mas o fator de amplificação que você calcular\n"
        "  será contra o piso da MINHA execução, não da sua.\n"
        "  Em casa: rode o notebook 01 e reexecute daqui."
    )
    return cfg


# =============================================================================
#  demonstração didática (usada no Bloco II-B)
# =============================================================================
def demo_leitura_remota() -> None:
    """Mostra na prática que HTTP Range funciona — e por que isso importa.

    Rode isto em aula: são 15 segundos e resolve de uma vez a pergunta
    "preciso baixar 14 GB?".
    """
    import time

    url = f"{URL_BASE_RELEASE}/{ARQ_FEATS}"
    print(f"arquivo remoto: {ARQ_FEATS}  (~132 MB)\n")

    t0 = time.perf_counter()
    schema = pd.read_parquet(url, columns=["Id"])
    t1 = time.perf_counter()
    print(f"  1 coluna  (Id)         : {len(schema):>9,} linhas em {t1 - t0:5.1f}s")
    print("     → o pyarrow pediu apenas os bytes da coluna `Id` via HTTP Range.")
    print("       O arquivo inteiro NÃO foi transferido.\n")
    print("  Moral: parquet colunar + Range = você lê 2 MB de um arquivo de 132 MB.")
    print("  Mas em aula usamos cache local, porque 40 pessoas × N leituras")
    print("  derrubam qualquer wifi de auditório.")
