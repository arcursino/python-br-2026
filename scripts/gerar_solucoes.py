#!/usr/bin/env python
"""
scripts/gerar_solucoes.py — gera os notebooks *_SOLUCAO.ipynb automaticamente.

Caminho no repositório: scripts/gerar_solucoes.py

O PROBLEMA QUE ISTO RESOLVE
===========================
Manter duas versões de cada notebook à mão é garantia de divergência: você
corrige um bug na solução, esquece o notebook com TODO, e em sala a célula
gabarito não bate com o exercício.

Aqui a fonte da verdade é uma só: o notebook com `TODO`, mais um dicionário de
soluções neste arquivo. As células de exercício são identificadas por **tag**
(`exercicio-1`, `exercicio-2`, ...) nos metadados da célula — não por número de
linha, não por regex no conteúdo.

Uso
---
    python scripts/gerar_solucoes.py                    # gera todos
    python scripts/gerar_solucoes.py --check            # só verifica (CI)
    make solucoes

No CI, `--check` falha se algum SOLUCAO estiver desatualizado. Assim é
impossível chegar no evento com os dois arquivos fora de sincronia.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
NOTEBOOKS = RAIZ / "notebooks"

# =============================================================================
#  As soluções oficiais. Uma por tag de célula.
#
#  Mantenha-as IDÊNTICAS à implementação em src/driftkit/ — os testes em
#  tests/ cobrem a versão do pacote, e é ela que o participante importa
#  depois. Divergência aqui é bug didático.
# =============================================================================
SOLUCOES: dict[str, str] = {
    # -------------------------------------------------------------------------
    "exercicio-1": '''\
# ✅ SOLUÇÃO — camada 1 da pirâmide
import pandas as pd


def viola_range(cur: pd.DataFrame, ranges: dict) -> dict[str, float]:
    """Fração de registros fora do range de engenharia, por feature.

    Três linhas de lógica. É a camada mais barata da pirâmide — O(n), sem
    referência, sem rótulo, sem modelo — e a única que pega troca de unidade,
    o caso em que retreinar grava o erro dentro do modelo.

    Duas decisões de projeto que valem comentário:

    1. `if feat not in cur.columns: continue`
       Feature ausente não é feature conforme. Se você usasse `cur[feat]`
       direto, um schema mudado viraria KeyError no meio da madrugada.
       Ausência é reportada por omissão e tratada por quem consome.

    2. `pd.to_numeric(..., errors="coerce")`
       Se o PLC começar a publicar string, `<` compara str com float e
       levanta TypeError. Com `coerce`, vira NaN — e NaN é excluído do
       cálculo em vez de derrubar o monitor.
    """
    fora: dict[str, float] = {}
    for feat, (lo, hi) in ranges.items():
        if feat not in cur.columns:
            continue
        s = pd.to_numeric(cur[feat], errors="coerce").dropna()
        if s.empty:
            continue
        fora[feat] = float(((s < lo) | (s > hi)).mean())
    return fora''',
    # -------------------------------------------------------------------------
    "exercicio-2": '''\
# ✅ SOLUÇÃO — o predicado de alarme
import numpy as np


def alarma(p_valor: float, efeito: float, alpha_corrigido: float, psi_limiar: float) -> bool:
    """Alarme = significância E magnitude. Nunca OU.

        (p < alpha/m)  E  (efeito > limiar)

    Por que `E`:

      • significância SEM magnitude → fadiga de alerta.
        Com 50 mil peças/dia, todo p-valor é zero. 160 features × alpha=0.05
        são ~8 alarmes falsos por janela; 5 janelas/semana são 40 alarmes
        falsos por semana. Em duas semanas ninguém olha o dashboard, e você
        construiu um gerador de ruído com aparência de observabilidade.

      • magnitude SEM significância → ruído amostral.
        Efeito de 0.85 numa janela de 40 observações é sorte, não drift.

    E o `np.isfinite`: NaN em comparação devolve False silenciosamente.
    `nan > 0.1` é False — ou seja, drift NÃO MENSURÁVEL passaria como
    ausência de drift. É o pior modo de falha possível para um monitor, e
    é por isso que `RelatorioDrift` expõe a coluna `mensuravel` em vez de
    confiar na sorte.
    """
    if not (np.isfinite(p_valor) and np.isfinite(efeito)):
        return False
    return bool(p_valor < alpha_corrigido and efeito > psi_limiar)''',
    # -------------------------------------------------------------------------
    "exercicio-3": '''\
# ✅ SOLUÇÃO — limiar ajustado pela cobertura
import numpy as np


def limiar_de(cobertura: float, piso_base: float) -> float:
    """Limiar que cresce quando a cobertura cai.

    O erro padrão de um estimador escala com 1/sqrt(n). Se a cobertura cai
    por um fator k, o n efetivo cai por k, e o ruído do PSI cresce por
    sqrt(k). Logo o limiar precisa crescer pelo mesmo fator — senão você
    gera alarme falso sistemático justamente nas features mais frágeis,
    que são as que menos deveriam disparar ordem de serviço.

    O `max(cobertura, 0.01)` evita divisão por zero e limita o limiar a
    10× o piso: acima disso a feature simplesmente não é monitorável e
    deve ser EXCLUÍDA do painel, não monitorada com limiar absurdo.
    Fingir que se monitora é pior que não monitorar.
    """
    cob = max(float(cobertura), 0.01)
    return float(piso_base / np.sqrt(cob))''',
    # -------------------------------------------------------------------------
    "exercicio-5": '''\
# ✅ SOLUÇÃO — guarda 5b: desgaste de ferramenta (tendência, não nível)
import numpy as np
from scipy.stats import kendalltau


def guarda_5b(historico_efeitos: list[float], n_janelas: int, *,
              alpha: float = 0.05, tau_minimo: float = 0.6) -> bool:
    """Bloqueia por TENDÊNCIA monotônica, não por nível.

    O insight: todas as outras guardas olham o NÍVEL do efeito nesta janela.
    Desgaste de ferramenta nunca cruza limiar nenhum — ele sobe devagar, e
    cada janela isolada parece saudável. A informação está na SEQUÊNCIA.

    Mann-Kendall (aqui via tau de Kendall contra o índice da janela) testa
    monotonicidade sem supor linearidade nem normalidade — o que importa,
    porque desgaste raramente é linear.

    As duas condições fazem trabalhos diferentes:
      • p < alpha        → a tendência não é sorte
      • tau >= 0.6       → a tendência é FORTE e consistente

    O tau alto é o que rejeita o DEGRAU: um degrau tem tau moderado (metade
    dos pares está em cada patamar), enquanto uma rampa tem tau próximo de 1.
    Degrau já é coberto pelas guardas 1 a 5; a 5b é só para o que rasteja.

    Em produção, acrescente:
      • um mínimo de janelas (n >= 6) para não decidir com 3 pontos
      • persistência: exija a tendência em duas avaliações consecutivas
      • destino do ticket = manutenção PREDITIVA, não corretiva —
        a ferramenta ainda está dentro da spec, e é justamente por isso
        que dá tempo de trocar antes do refugo
    """
    y = np.asarray(historico_efeitos[-n_janelas:], dtype=float)
    y = y[np.isfinite(y)]
    if y.size < 6:
        return False  # poucos pontos: sem decisão. Nunca chute tendência.

    tau, p = kendalltau(np.arange(y.size), y)
    if not (np.isfinite(tau) and np.isfinite(p)):
        return False
    return bool(p < alpha and tau >= tau_minimo)''',
}

# Notebooks a processar
ALVOS = ["01_sintetico.ipynb", "02_real.ipynb"]

CABECALHO_MD = (
    "> ### ✅ Versão com SOLUÇÕES\n"
    ">\n"
    "> Este notebook é **gerado automaticamente** por "
    "`scripts/gerar_solucoes.py` a partir da versão com `TODO`.\n"
    "> Não edite aqui — edite o notebook original ou o dicionário `SOLUCOES` "
    "no script.\n"
    ">\n"
    "> Use-o se você travar num exercício: copie a célula, siga adiante, e "
    "volte depois.\n"
    "> **Ninguém fica para trás por causa de um exercício.**\n"
)


def tags(celula: dict) -> list[str]:
    return list(celula.get("metadata", {}).get("tags", []))


def transformar(nb: dict) -> tuple[dict, list[str]]:
    """Devolve (notebook com soluções, lista de tags substituídas)."""
    novo = json.loads(json.dumps(nb))  # deep copy
    trocadas: list[str] = []

    for cel in novo["cells"]:
        for tag in tags(cel):
            if tag in SOLUCOES:
                fonte = SOLUCOES[tag]
                cel["source"] = [li + "\n" for li in fonte.splitlines()]
                cel["outputs"] = []
                cel["execution_count"] = None
                cel.setdefault("metadata", {})["tags"] = [tag, "solucao"]
                trocadas.append(tag)
                break

    # banner no topo
    novo["cells"].insert(0, {
        "cell_type": "markdown",
        "metadata": {"tags": ["banner-solucao"]},
        "source": [li + "\n" for li in CABECALHO_MD.splitlines()],
    })
    return novo, trocadas


def caminho_solucao(p: Path) -> Path:
    return p.with_name(p.stem + "_SOLUCAO" + p.suffix)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="não escreve; falha se algum SOLUCAO estiver desatualizado")
    args = ap.parse_args()

    desatualizados: list[str] = []
    tags_usadas: set[str] = set()
    erro = False

    for nome in ALVOS:
        origem = NOTEBOOKS / nome
        if not origem.exists():
            print(f"❌ não encontrado: {origem}")
            erro = True
            continue

        nb = json.loads(origem.read_text(encoding="utf-8"))
        novo, trocadas = transformar(nb)
        tags_usadas |= set(trocadas)
        destino = caminho_solucao(origem)
        conteudo = json.dumps(novo, indent=1, ensure_ascii=False) + "\n"

        if args.check:
            atual = destino.read_text(encoding="utf-8") if destino.exists() else ""
            if atual != conteudo:
                desatualizados.append(destino.name)
            estado = "⚠️  desatualizado" if destino.name in desatualizados else "✅ em dia"
            print(f"  {destino.name:<34} {estado}  ({len(trocadas)} exercícios)")
        else:
            destino.write_text(conteudo, encoding="utf-8")
            print(f"  ✅ {destino.name:<32} {len(trocadas)} exercício(s): "
                  f"{', '.join(trocadas) or '—'}")

    # soluções órfãs: definidas aqui mas sem célula correspondente
    if orfas := set(SOLUCOES) - tags_usadas:
        print(f"\n⚠️  soluções sem célula correspondente nos notebooks: {sorted(orfas)}")
        print("    Confira as tags nos metadados das células de exercício.")
        erro = True

    if args.check and desatualizados:
        print(f"\n❌ desatualizados: {desatualizados}")
        print("   rode: python scripts/gerar_solucoes.py")
        return 1
    if erro:
        return 1

    print("\n✅ tudo em sincronia.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
