"""
driftkit.notebook — utilitários de sala de aula.

Caminho no repositório: src/driftkit/notebook.py

Este módulo não tem nada de sofisticado. Ele existe porque um tutorial de 3h30
com 40 pessoas em máquinas de laboratório tem problemas que nenhum livro de
engenharia de software menciona:

  • gente que executa células fora de ordem e vê erro incompreensível
  • gente que trava e não levanta a mão
  • gente que perde a sessão do Colab e não sabe onde retomar
  • você, que precisa saber em 3 segundos se a sala está acompanhando

Cada função aqui resolve um desses.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any

__all__ = ["requer", "checkpoint", "cli", "aposte", "banner", "resposta"]


# =============================================================================
def requer(*nomes: str, escopo: dict[str, Any] | None = None) -> None:
    """Falha cedo e com mensagem útil se células anteriores não rodaram.

    Coloque como PRIMEIRA LINHA de toda célula pesada:

        requer("df", "detector", "CFG_V1")

    Sem isto, executar fora de ordem produz `NameError: name 'df' is not
    defined` — que é tecnicamente correto e pedagogicamente inútil.
    """
    if escopo is None:
        import inspect

        frame = inspect.currentframe()
        escopo = frame.f_back.f_globals if frame and frame.f_back else {}

    faltando = [n for n in nomes if n not in escopo]
    if faltando:
        raise RuntimeError(
            f"\n{'=' * 58}\n"
            f"  Faltam objetos desta sessão: {faltando}\n"
            f"{'=' * 58}\n"
            "  Você provavelmente pulou células ou o runtime reiniciou.\n\n"
            "  No Colab:  Ambiente de execução → Executar antes  (Ctrl+F8)\n"
            "  Se o runtime caiu, rode a Célula 0 de novo (leva ~90s).\n"
        )


# =============================================================================
_CHECKPOINTS = {
    1: "ambiente pronto, dados baixados",
    2: "suíte verde e contrato v1 salvo",
    3: "eixo temporal do Bosch construído",
    4: "limiares recalibrados (v2) e teste A/A sem falso positivo",
    5: "política decidindo e escada de intervenção medida",
}


def checkpoint(n: int, **condicoes: bool) -> bool:
    """Autoverificação visível. O participante SABE se está em dia.

        checkpoint(2,
            suite_verde=not falhas,
            contrato_salvo=(DATA / "detector_config.json").exists(),
            piso_medido=piso < 0.05,
        )

    Perguntar "todo mundo bem?" não funciona: ninguém levanta a mão para
    admitir que travou. Uma célula que imprime ❌ e manda levantar o cartão
    vermelho remove o custo social de pedir ajuda.
    """
    falhou = [k for k, v in condicoes.items() if not v]
    titulo = _CHECKPOINTS.get(n, "")
    largura = 58
    print("=" * largura)
    if falhou:
        print(f"  ❌ CHECKPOINT {n}/5 — {titulo}")
        print(f"     pendências: {', '.join(falhou)}")
        print("     Levante o cartão 🔴. NÃO avance sozinho —")
        print("     os próximos blocos dependem deste.")
    else:
        print(f"  ✅ CHECKPOINT {n}/5 — {titulo}")
        print("     Você está em dia. Levante o cartão 🟢.")
    print("=" * largura)
    return not falhou


# =============================================================================
def cli(*args: str, mostrar: bool = True) -> int:
    """Roda o CLI `driftkit` e devolve o EXIT CODE de verdade.

    ⚠️  Por que não usar `!driftkit decide ...`?

    A magia `!` do Jupyter NÃO propaga o código de saída. Ela imprime a saída
    padrão e descarta o exit code — e o exit code é justamente onde mora a
    política inteira deste tutorial. Use isto:

        cli("decide", "data/janela_TC3.parquet")   # → 20
    """
    p = subprocess.run(
        [sys.executable, "-m", "driftkit", *args], capture_output=True, text=True
    )
    if mostrar:
        print(p.stdout or "", end="")
        if p.stderr.strip():
            print(p.stderr, end="")

    rotulos = {
        0: "🟢 nada a fazer",
        10: "🔁 RETREINO APROVADO",
        20: "⛔ RETREINO BLOQUEADO (causa não-ML)",
        30: "⚪ dados insuficientes",
    }
    if mostrar:
        print(f"\n  exit={p.returncode} → {rotulos.get(p.returncode, 'erro do processo')}")
    return p.returncode


# =============================================================================
def aposte(pergunta: str, opcoes: tuple[str, ...] = ("SIM", "NÃO")) -> None:
    """Momento 'aposte antes de rodar'.

    Custa 90 segundos e é o mecanismo de aprendizagem mais eficiente do
    tutorial: quem apostou errado NÃO esquece o resultado. Use antes das três
    células mais contraintuitivas (TC-3, o retreino que "funciona", o fator de
    amplificação).

    Imprima, conte as mãos em voz alta, ANOTE o placar no flipchart, e só
    então execute a célula seguinte.
    """
    largura = 62
    print("┌" + "─" * largura + "┐")
    print("│" + "  🎲  APOSTE ANTES DE RODAR".ljust(largura) + "│")
    print("├" + "─" * largura + "┤")
    for linha in _quebrar(pergunta, largura - 4):
        print("│  " + linha.ljust(largura - 2) + "│")
    print("│" + " " * largura + "│")
    print("│  " + ("  ".join(f"[ {o} ]" for o in opcoes)).ljust(largura - 2) + "│")
    print("└" + "─" * largura + "┘")
    print("  Levante a mão. Conte o placar. DEPOIS rode a célula seguinte.")


def _quebrar(texto: str, largura: int) -> list[str]:
    palavras, linhas, atual = texto.split(), [], ""
    for p in palavras:
        if len(atual) + len(p) + 1 > largura:
            linhas.append(atual)
            atual = p
        else:
            atual = f"{atual} {p}".strip()
    if atual:
        linhas.append(atual)
    return linhas


# =============================================================================
def banner(titulo: str, subtitulo: str = "", *, icone: str = "🔬") -> None:
    """Separador visual entre blocos. Ajuda quem se perdeu a se reencontrar."""
    print()
    print("━" * 62)
    print(f"  {icone}  {titulo}")
    if subtitulo:
        print(f"      {subtitulo}")
    print("━" * 62)


# =============================================================================
def resposta(chave: str) -> None:
    """Revela o gabarito de uma atividade — só quando o instrutor manda.

    Mantém os gabaritos no notebook (para levar para casa) sem estragar a
    atividade de quem lê adiante.
    """
    from .testing import GABARITOS

    if chave not in GABARITOS:
        print(f"gabarito `{chave}` não existe. Disponíveis: {sorted(GABARITOS)}")
        return
    print("─" * 62)
    print(GABARITOS[chave].strip())
    print("─" * 62)
