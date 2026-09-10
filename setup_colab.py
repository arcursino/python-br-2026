"""
setup_colab.py — bootstrap idempotente para Google Colab.

USO NO NOTEBOOK (célula 0, duas linhas):

    !curl -sSL https://raw.githubusercontent.com/arcursino/python-br-2026/main/setup_colab.py -o /tmp/s.py
    %run /tmp/s.py

Ou, se o repositório já foi clonado:

    %run setup_colab.py

O que faz, em ordem:
  1. clona (ou atualiza) o repositório em /content/python-br-2026
  2. TRAVA a stack binária que o Colab já trouxe (numpy/scipy/pandas/sklearn)
  3. instala o pacote EM CAMADAS (núcleo obrigatório → extras opcionais)
  4. baixa os dados derivados do GitHub Releases (~180 MB, não os 14 GB da Kaggle)
  5. imprime um diagnóstico linha a linha
  6. liga %autoreload para que edições em src/ tenham efeito sem reiniciar

PRINCÍPIO DE PROJETO
--------------------
Só existem DUAS razões para abortar: o clone falhou, ou o núcleo não instalou.
Qualquer outra falha é registrada e o setup continua — porque nenhum extra é
pré-requisito para os notebooks, e um `pip` quebrado num pacote opcional não
pode custar 10 minutos de sala com 100 pessoas.

POR QUE A TRAVA DE ABI (passo 2)
--------------------------------
O Colab embarca scipy, pandas e scikit-learn como wheels COMPILADAS contra
o numpy que vem na mesma imagem. Se o pip resolver o nosso `numpy>=1.26`
para uma versão diferente — para cima OU para baixo — a ABI quebra, e o
sintoma muda a cada execução:

    AttributeError: ... has no attribute '_blas_supports_fpe'
    ImportError: cannot import name '_slice' from 'numpy._core.umath'
    AttributeError: 'numpy.ufunc' object has no attribute '__module__'

São todos o MESMO problema. A solução não é escolher a versão "certa" do
numpy: é não deixar o pip encostar nele. Geramos um arquivo de constraints
a partir do que JÁ está instalado e o aplicamos em todas as chamadas de pip,
inclusive via PIP_CONSTRAINT (que alcança também os subprocessos de build
isolation, onde um `-c` na linha de comando não chega).

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

# Pacotes cuja versão NÃO pode ser trocada: são wheels binárias que o Colab
# já trouxe compiladas umas contra as outras.
STACK_BINARIA = ("numpy", "scipy", "pandas", "scikit-learn", "pyarrow")
CONSTRAINTS = Path("/tmp/pybr2026-constraints.txt")

# Arquivos de dados: (nome, obrigatório_para, tamanho_aprox_mb)
ARQUIVOS_DADOS = [
    ("detector_config_referencia_v1.json", "fallback da Parte I", 0.01),
    ("meta_bosch.parquet", "Parte II — eixo temporal", 48),
    ("bosch_num_160feats.parquet", "Parte II — features", 132),
]

IN_COLAB = "google.colab" in sys.modules

_falhas: list[str] = []   # aborta o avanço
_avisos: list[str] = []   # degrada, mas segue


# =============================================================================
#  utilitários
# =============================================================================
def sh(cmd: str, *, critico: bool = True, linhas_erro: int = 40) -> subprocess.CompletedProcess:
    """Roda um comando de shell mostrando saída só em caso de erro.

    Em falha, imprime as ÚLTIMAS `linhas_erro` linhas de stdout+stderr
    combinados — que é onde o pip põe o nome do pacote culpado. A versão
    anterior deste arquivo cortava por bytes e separava os streams, e o
    resultado era um erro mudo: duas barras de progresso e nada mais.
    """
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if r.returncode:
        saida = ((r.stdout or "") + (r.stderr or "")).strip().splitlines()
        print(f"\n{'❌' if critico else '⚠️ '} comando falhou: {cmd}")
        print("─" * 62)
        for linha in saida[-linhas_erro:]:
            print("  " + linha)
        print("─" * 62)
        _explicar_erro_pip(saida)
        if critico:
            raise SystemExit(
                "\nSETUP INTERROMPIDO. Levante o cartão vermelho 🔴 e chame o monitor.\n"
                "Não avance para as próximas células."
            )
    return r


def _explicar_erro_pip(saida: list[str]) -> None:
    """Traduz os erros de pip que a gente já viu, com a saída pronta."""
    texto = "\n".join(saida).lower()
    if "metadata-generation-failed" in texto or "preparing metadata" in texto:
        culpado = next(
            (p for p in ("alibi-detect", "alibi_detect", "evidently", "river", "tensorflow")
             if p in texto), None
        )
        print("\n  DIAGNÓSTICO: algum pacote não conseguiu gerar metadata.")
        if culpado:
            print(f"  Culpado provável: {culpado}")
        print(f"  Python deste runtime: {sys.version.split()[0]}")
        print("\n  Isto é típico de sdist com pins antigos em Python 3.12+.")
        print("  Desbloqueio (roda agora, sem os extras frágeis):")
        print(f'      !pip install -q -c {CONSTRAINTS} -e "{RAIZ}[dev,mercado]"')
    elif "no space left" in texto:
        print("\n  DIAGNÓSTICO: disco cheio. Menu → Runtime → Disconnect and delete runtime.")
    elif "could not find a version" in texto:
        print("\n  DIAGNÓSTICO: nenhuma versão compatível com este Python.")
        print(f"  Python deste runtime: {sys.version.split()[0]}")
    elif "conflict" in texto and "constraint" in texto:
        print("\n  DIAGNÓSTICO: um extra exige versão de numpy/scipy/pandas diferente")
        print("  da que o Colab embarca. Ele NÃO será instalado — e está certo assim:")
        print("  trocar a stack binária quebra o `import driftkit`.")


def pip_install(alvo: str, *, critico: bool = True) -> subprocess.CompletedProcess:
    """pip install com a trava de ABI sempre aplicada.

    Nunca chame `pip install` direto neste arquivo. Toda instalação passa por
    aqui, senão a trava vira decoração.
    """
    return sh(f'pip install -q -c "{CONSTRAINTS}" {alvo}', critico=critico)


def baixar(url: str, destino: Path, *, mb: float = 0) -> bool:
    """Baixa com aviso de progresso. Devolve False se falhar (não aborta)."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_suffix(destino.suffix + ".parcial")
    print(f"   baixando {destino.name} (~{mb:.0f} MB)...", end=" ", flush=True)
    r = subprocess.run(
        f'curl -fsSL --retry 3 --retry-delay 2 "{url}" -o "{tmp}"',
        shell=True, text=True, capture_output=True,
    )
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
print(f"\n  Python {sys.version.split()[0]}  ·  Colab: {'sim' if IN_COLAB else 'não'}")

if IN_COLAB:
    if (RAIZ / ".git").exists():
        print("\n[1/5] repositório já presente — atualizando...")
        sh(f"git -C {RAIZ} pull -q --ff-only", critico=False)
    else:
        print("\n[1/5] clonando repositório...")
        if RAIZ.exists():
            shutil.rmtree(RAIZ)
        sh(f"git clone --depth 1 -q {REPO_URL} {RAIZ}")   # CRÍTICO
    os.chdir(RAIZ)
else:
    print("\n[1/5] fora do Colab — usando diretório atual como raiz")
    os.chdir(RAIZ)

# =============================================================================
#  2. TRAVA DE ABI  (antes de qualquer pip install)
# =============================================================================
print("[2/5] travando a stack binária já instalada...")

_travados: list[str] = []
for _pkg in STACK_BINARIA:
    try:
        _travados.append(f"{_pkg}=={md.version(_pkg)}")
    except md.PackageNotFoundError:
        # não está instalado: deixe o pip resolver livremente
        pass

CONSTRAINTS.write_text("\n".join(_travados) + "\n")

# PIP_CONSTRAINT alcança também os subprocessos de build isolation, onde um
# `-c` na linha de comando não chega. Cinto e suspensório.
os.environ["PIP_CONSTRAINT"] = str(CONSTRAINTS)

if _travados:
    for _t in _travados:
        print(f"      🔒 {_t}")
else:
    print("      (nada a travar — ambiente limpo)")
    _avisos.append("stack binária não estava pré-instalada; ABI não foi travada")

# =============================================================================
#  3. instalação EM CAMADAS
# =============================================================================
# Camada 1 (crítica): núcleo + dev. Sem isto, nada funciona.
# Camada 2 (opcional): river — wheels puras, raramente falha.
# Camada 3 (opcional): evidently — sdist grande, falha com alguma frequência.
#
# alibi-detect NÃO é instalado aqui, de propósito: em Python 3.12+ ele quebra
# a geração de metadata e derrubava o setup inteiro. Está em
# `[mercado-alibi]` para quem quiser tentar num venv com Python 3.11.
print("[3/5] instalando em camadas (com a trava aplicada)...")

print("      camada 1/3  núcleo + dev (crítica)...", end=" ", flush=True)
pip_install('-e ".[dev]"')   # CRÍTICO — só isto aborta
print("ok ✅")

for nome_camada, extra, rotulo in [
    ("2/3", "mercado", "river"),
    ("3/3", "mercado-evidently", "evidently"),
]:
    print(f"      camada {nome_camada}  {rotulo} (opcional)...", end=" ", flush=True)
    r = subprocess.run(
        f'pip install -q -c "{CONSTRAINTS}" -e ".[{extra}]"',
        shell=True, text=True, capture_output=True,
    )
    if r.returncode:
        print("falhou ⚠️  (segue sem ele)")
        _avisos.append(f"{rotulo} não instalou — a célula que o usa degrada sozinha")
    else:
        print("ok ✅")

# ---- a trava funcionou? conferir é barato; descobrir no import é caro ------
_mudou = []
for _spec in _travados:
    _nome, _versao = _spec.split("==")
    try:
        if md.version(_nome) != _versao:
            _mudou.append(f"{_nome}: {_versao} → {md.version(_nome)}")
    except md.PackageNotFoundError:
        _mudou.append(f"{_nome}: removido!")
if _mudou:
    print("\n      ⚠️  a stack binária MUDOU apesar da trava:")
    for _m in _mudou:
        print(f"            {_m}")
    print("      O `import driftkit` provavelmente vai falhar por ABI.")
    print("      Desbloqueio: Runtime → Desconectar e excluir → rodar de novo.")
    _avisos.append("stack binária alterada apesar da trava — ver acima")

# cinto e suspensório: garante o src no path mesmo se o editable falhar
src = str(RAIZ / "src")
if src not in sys.path:
    sys.path.insert(0, src)

# =============================================================================
#  4. dados
# =============================================================================
print("[4/5] verificando dados derivados...")
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
        if not baixar(f"{BASE_RELEASE}/{nome}", destino, mb=mb):
            if nome in ESSENCIAL:
                _avisos.append(f"{nome} não baixou — o contrato v1 usará o default embutido")

# =============================================================================
#  5. diagnóstico
# =============================================================================
print("[5/5] diagnóstico\n")

# Versão MÍNIMA aceitável de cada pacote — não prefixo de string.
#
# A versão anterior comparava prefixos (`v.startswith("1")`) e envelheceu
# sozinha: quando o Colab passou pyarrow de 18.1.0 para 23.0.1, `"23"` deixou
# de começar com `"1"` e o setup pintou 🔴 num ambiente perfeitamente saudável.
#
# É a tese do Bloco III aplicada ao nosso próprio setup: um limiar FIXO herdado
# não vira falso negativo — vira falso POSITIVO, e quem paga é o primeiro slide
# do tutorial. Trocamos o prefixo por um piso semântico: só reprova o que é
# velho DEMAIS, e nunca reprova por ser novo.
#
# obrigatórios: ausência é FALHA
OBRIGATORIOS = {
    "numpy": "1.26",
    "pandas": "2.0",
    "scipy": "1.10",
    "scikit-learn": "1.3",
    "pyarrow": "14.0",
    "typer": "0.12",
    "pytest": "8.0",
}
# opcionais: ausência é AVISO
OPCIONAIS = {
    "river": "0.21",
    "evidently": "0.4",
}


def _versao_ok(instalada: str, minima: str) -> bool:
    """`instalada >= minima`, comparando número a número.

    Usa `packaging` quando disponível (trata rc/dev/post corretamente) e cai
    num parser ingênuo de inteiros quando não — o setup não pode depender de
    um extra para conseguir dizer se o ambiente está bom.
    """
    try:
        from packaging.version import Version

        return Version(instalada) >= Version(minima)
    except Exception:  # noqa: BLE001
        def _t(v: str) -> tuple[int, ...]:
            partes = []
            for p in v.split(".")[:3]:
                digitos = "".join(c for c in p if c.isdigit())
                partes.append(int(digitos) if digitos else 0)
            return tuple(partes)

        return _t(instalada) >= _t(minima)

print(f"  {'pacote':<26}{'':<3}{'versão'}   (mínimo exigido)")
print("  " + "-" * 46)
for grupo, obrigatorio in ((OBRIGATORIOS, True), (OPCIONAIS, False)):
    for pkg, minima in grupo.items():
        try:
            v = md.version(pkg)
            ok = _versao_ok(v, minima)
            icone = "✅" if ok else "⚠️"
            if not ok:
                (_falhas if obrigatorio else _avisos).append(
                    f"{pkg} {v} é anterior ao mínimo exigido ({minima})"
                )
        except md.PackageNotFoundError:
            v = "ausente"
            icone = "❌" if obrigatorio else "➖"
            if obrigatorio:
                _falhas.append(f"{pkg} ausente")
            else:
                _avisos.append(f"{pkg} ausente (opcional)")
        trava = " 🔒" if pkg in STACK_BINARIA else ""
        print(f"  {pkg:<26}{icone:<3}{v}{trava:<3}  (≥ {minima})")

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
    import driftkit

    importlib.reload(driftkit)
    from driftkit.detectors import piso_analitico, psi  # noqa: F401

    print(f"  {'import driftkit':<36}{'✅':<3}v{driftkit.__version__}")
except Exception as e:  # noqa: BLE001
    print(f"  {'import driftkit':<36}{'❌':<3}{type(e).__name__}: {e}")
    _falhas.append("import driftkit")
    # ABI quebrada tem cara própria: dê o desbloqueio junto com o erro.
    _texto_erro = f"{type(e).__name__}: {e}".lower()
    if any(s in _texto_erro for s in ("numpy", "ufunc", "umath", "abi", "_core")):
        print("\n      DIAGNÓSTICO: incompatibilidade de ABI do numpy.")
        print("      Isto é drift de ambiente — o mesmo fenômeno do tutorial,")
        print("      só que no seu interpretador. Desbloqueio:")
        print("        1. Runtime → Desconectar e EXCLUIR o ambiente de execução")
        print("        2. rodar esta célula de novo (runtime limpo, trava ativa)")
        print("      Reinstalar sem reiniciar NÃO resolve: o módulo quebrado")
        print("      já está carregado em memória.")

r = sh("driftkit --help", critico=False, linhas_erro=5)
if r.returncode == 0:
    print(f"  {'CLI driftkit':<36}{'✅':<3}disponível no PATH")
else:
    print(f"  {'CLI driftkit':<36}{'⚠️':<3}use `python -m driftkit` no lugar")
    _avisos.append("CLI fora do PATH — use `python -m driftkit`")

# suíte rápida: se ela passa, o ambiente está bom de verdade
r = subprocess.run(
    "pytest -q -m 'not lento and not requer_bosch and not mercado' --no-header",
    shell=True, text=True, capture_output=True,
)
ultima = (r.stdout or "").strip().splitlines()
print(f"  {'suíte de testes':<36}{'✅' if r.returncode == 0 else '⚠️':<3}"
      f"{ultima[-1] if ultima else 'sem saída'}")
if r.returncode:
    _falhas.append("suíte de testes falhando")

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
    print("  ❌ PENDÊNCIAS QUE IMPEDEM O TUTORIAL:")
    for f in _falhas:
        print(f"       • {f}")
    print("\n  Levante o cartão 🔴 AGORA. Não avance sozinho.")
elif _avisos:
    print("  ✅ AMBIENTE PRONTO — com degradações aceitáveis:")
    for a in _avisos:
        print(f"       • {a}")
    print("\n  Nada disso impede o tutorial. Levante o cartão 🟢.")
else:
    print("  ✅ AMBIENTE PRONTO. Levante o cartão 🟢.")
print(f"\n  RAIZ = {RAIZ}")
print(f"  DATA = {DATA}")
print(f"  TRAVA = {CONSTRAINTS}")
print("=" * 62)

# expostos para as células seguintes
RAIZ_REPO, DATA_DIR = RAIZ, DATA
