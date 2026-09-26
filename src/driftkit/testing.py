"""
driftkit.testing — gabaritos executáveis dos exercícios "você implementa".

Caminho no repositório: src/driftkit/testing.py

Cada `checar_*` recebe a FUNÇÃO escrita pelo participante e a submete aos
mesmos testes que a implementação oficial passa. A mensagem de erro é o
material didático: ela diz o que falhou E por que aquilo importa.

Regra de projeto das mensagens: nunca "assertion failed". Sempre
"você retornou X, esperava Y, e o motivo é Z".
"""

from __future__ import annotations

from typing import Callable

import numpy as np

__all__ = [
    "checar_viola_range",
    "checar_predicado_alarme",
    "checar_limiar_por_estrato",
    "checar_guarda_5b",
    "GABARITOS",
]


def _ok(nome: str) -> None:
    print(f"  ✅ {nome}")


def _falha(nome: str, obtido, esperado, porque: str) -> None:
    raise AssertionError(
        f"\n  ❌ {nome}\n"
        f"     você retornou : {obtido}\n"
        f"     esperado      : {esperado}\n"
        f"     por quê       : {porque}\n"
    )


# =============================================================================
#  Exercício 1 (Bloco II-A) — a camada mais barata da pirâmide
# =============================================================================
def checar_viola_range(fn: Callable) -> None:
    """Gabarito de `viola_range(cur, ranges) -> dict[str, float]`."""
    from .fixtures import RANGES, gerar_fixture, janela

    print("verificando `viola_range`...")
    df = gerar_fixture(seed=7)

    # TC-4: pressão em kPa → praticamente tudo fora do range
    w4 = janela(df, 128, w=2)
    r4 = fn(w4, RANGES)
    if not isinstance(r4, dict):
        _falha("tipo de retorno", type(r4).__name__, "dict",
               "o relatório precisa dizer QUAL feature violou, não só que houve violação")
    if r4.get("pressao_psi", 0) < 0.9:
        _falha("TC-4 pressão em kPa", f"{r4.get('pressao_psi')}", "> 0.9",
               "com o fator 6.895 aplicado, ~100% dos registros saem do range 25–45 psi")
    _ok("TC-4: detecta a troca de unidade")

    if r4.get("torque_medido", 1.0) > 0.02:
        _falha("TC-4 torque", f"{r4.get('torque_medido')}", "< 0.02",
               "o torque está conforme neste período — alarmar aqui é falso positivo, "
               "e falso positivo é o que faz o time parar de olhar o dashboard")
    _ok("TC-4: NÃO alarma nas features saudáveis")

    # TC-1: controle negativo
    w1 = janela(df, 45)
    r1 = fn(w1, RANGES)
    if max(r1.values(), default=0) > 0.02:
        _falha("TC-1 controle negativo", f"máx {max(r1.values()):.4f}", "< 0.02",
               "em regime estável NENHUMA feature deve violar range. "
               "Este é o teste mais importante da suíte: um detector que sempre "
               "alarma é indistinguível de um detector quebrado")
    _ok("TC-1: silêncio em regime estável")

    # TC-3: concept drift NÃO viola range — e é esse o ponto
    w3 = janela(df, 110)
    r3 = fn(w3, RANGES)
    if r3.get("torque_medido", 0) > 0.05:
        _falha("TC-3 não deve violar range", f"{r3.get('torque_medido')}", "< 0.05",
               "o offset de +10 Nm mantém a leitura DENTRO da faixa física plausível. "
               "É por isso que a camada 1 é cega ao TC-3 — e por isso existem as "
               "camadas 2, 3 e 4")
    _ok("TC-3: camada 1 é cega ao concept drift (e isso é esperado!)")

    print("\n  🎉 Você implementou a camada mais barata da pirâmide.")
    print("     Custo: O(n), sem referência, sem rótulo, sem modelo.")
    print("     E ela é a única que pega o caso em que retreinar é o pior erro.")


# =============================================================================
#  Exercício 2 (Bloco II-A) — o predicado que define fadiga de alerta
# =============================================================================
def checar_predicado_alarme(fn: Callable) -> None:
    """Gabarito de `alarma(p_valor, efeito, alpha_corrigido, psi_limiar) -> bool`."""
    print("verificando o predicado de alarme...")

    casos = [
        # (p, efeito, alpha_c, psi_lim, esperado, explicação)
        (1e-30, 0.004, 5e-4, 0.05, False,
         "p astronomicamente pequeno com efeito ínfimo: é o paradoxo do N grande. "
         "Com 50 mil peças/dia, TODO p-valor é zero. Alarmar aqui gera 40 alertas "
         "falsos por semana e mata o dashboard em duas semanas"),
        (0.30, 0.85, 5e-4, 0.05, False,
         "efeito enorme mas sem significância: amostra pequena demais. "
         "É ruído amostral com cara de catástrofe"),
        (1e-12, 0.42, 5e-4, 0.05, True,
         "significativo E relevante: este é o único caso que merece alarme"),
        (np.nan, 0.9, 5e-4, 0.05, False,
         "p-valor não mensurável. NaN em comparação devolve False silenciosamente — "
         "por isso o relatório expõe a coluna `mensuravel` em vez de confiar na sorte"),
    ]
    for p, ef, a, lim, esperado, porque in casos:
        obtido = bool(fn(p, ef, a, lim))
        if obtido != esperado:
            _falha(f"p={p:.1e}, efeito={ef}", obtido, esperado, porque)
    _ok("os quatro casos passaram")
    print("\n  A resposta é `E`, não `OU`:")
    print("      (p < alpha/m)  E  (efeito > limiar)")
    print("  significância sem magnitude → fadiga de alerta")
    print("  magnitude sem significância → ruído amostral")


# =============================================================================
#  Exercício 3 (Bloco III) — limiar por estrato
# =============================================================================
def checar_limiar_por_estrato(fn: Callable) -> None:
    """Gabarito de `limiar_de(cobertura, piso_base) -> float`."""
    print("verificando `limiar_de`...")
    l_alta = fn(0.95, 0.01)
    l_baixa = fn(0.10, 0.01)
    if not (l_baixa > l_alta):
        _falha("monotonicidade", f"cob 10% → {l_baixa:.4f}, cob 95% → {l_alta:.4f}",
               "limiar MAIOR para cobertura MENOR",
               "feature com 10% de cobertura tem 10× menos amostras por janela; "
               "o ruído do estimador cresce, e usar o mesmo limiar produz "
               "alarme falso sistemático justamente nas features mais frágeis")
    _ok("limiar cresce quando a cobertura cai")
    if l_alta <= 0:
        _falha("piso positivo", l_alta, "> 0", "limiar zero alarma sempre")
    _ok("limiar sempre positivo")


# =============================================================================
#  Exercício 5 (Bloco V) — a guarda que VOCÊ inventa
# =============================================================================
def checar_guarda_5b(fn: Callable) -> None:
    """Gabarito aberto de uma guarda para DESGASTE DE FERRAMENTA.

    Exercício difícil e deliberadamente subespecificado. O desgaste é gradual:
    não tem degrau, não viola range, e a magnitude por janela fica sempre
    abaixo do limiar. Todas as guardas anteriores passam, janela após janela,
    até o processo sair de especificação.

    Assinatura esperada:
        fn(historico_efeitos: list[float], n_janelas: int) -> bool
        (True = bloquear e encaminhar para manutenção preditiva)
    """
    print("verificando a guarda 5b (desgaste gradual)...")
    rampa = [0.004, 0.007, 0.011, 0.016, 0.022, 0.029, 0.037, 0.046]
    ruido = [0.021, 0.008, 0.019, 0.011, 0.024, 0.009, 0.017, 0.013]
    degrau = [0.005, 0.006, 0.005, 0.007, 0.31, 0.29, 0.33, 0.30]

    if not fn(rampa, 8):
        _falha("rampa monotônica", False, True,
               "nenhum valor cruza 0.05, mas a tendência é inequívoca. "
               "É a assinatura do desgaste de ferramenta — o modo de drift mais "
               "comum em manufatura e o menos coberto em tutoriais")
    _ok("detecta a rampa")

    if fn(ruido, 8):
        _falha("ruído sem tendência", True, False,
               "oscilação em torno de uma média estável não é desgaste. "
               "Uma guarda que dispara aqui gera ordem de serviço toda semana "
               "e a manutenção para de atender")
    _ok("ignora ruído sem tendência")

    if fn(degrau, 8):
        _falha("degrau súbito", True, False,
               "degrau é troca de lote ou firmware — as guardas 1 a 5 já cobrem. "
               "A 5b é especificamente para o que cresce devagar")
    _ok("não confunde degrau com desgaste")
    print("\n  🏆 Dica: teste de tendência de Mann-Kendall, ou regressão do efeito")
    print("     contra o índice da janela exigindo coeficiente > 0 e p < 0.05.")


# =============================================================================
GABARITOS: dict[str, str] = {
    "atividade1": """
GABARITO — Atividade 1 (classifique os cinco casos)

1. Rolamento trocado na CEL-03, features na faixa histórica
   → CONCEPT DRIFT por mudança física. Camada 3, e SÓ se segmentada por célula.
   → Retreinar apenas após confirmar que a máquina está conforme.
     Senão você ensina o defeito ao modelo.

2. `pressao_psi` chega multiplicada por 6.895
   → DATA QUALITY. Camada 1 pega em segundos.
   → Corrigir a tag. JAMAIS retreinar: gravaria a unidade errada no modelo.

3. Campanha de 30 dias com 80% de aro R20
   → DATA DRIFT BENIGNO. Camada 2 alarma, camada 3 não.
   → Registrar e seguir. É a causa nº 1 de retreino desnecessário na indústria.

4. Novo fornecedor com 25% do volume, categoria inédita
   → COVARIATE SHIFT com categoria nova. Camadas 1 e 2.
   → Retreinar INCLUINDO o novo domínio é legítimo. É o único dos cinco casos
     em que retreinar é a resposta.

5. Turno C com falha maior há dois anos
   → NÃO É DRIFT. É padrão estrutural estável que deveria ser feature.
   → Se seu monitor alarma aqui, o problema é a referência, não o processo.

Placar típico: erram-se mais os casos 2, 3 e 5 — exatamente os três em que
retreinar é a decisão errada.
""",
    "atividade2": """
GABARITO — Atividade 2 (as quatro hipóteses para drift quase contínuo)

Cenário: o monitor acusa drift em quase todo o período. Quatro hipóteses:

(a) A referência está mal escolhida (curta, atípica, ou dentro de um evento).
    Como testar: mude a janela de referência e veja se o padrão sobrevive.

(b) O limiar está abaixo do piso de ruído do detector.
    Como testar: teste A/A. Se ele alarma contra a PRÓPRIA referência,
    o problema é o instrumento. É o teste de 15 minutos do call-to-action.

(c) Há sazonalidade real não modelada (turno, semana, campanha).
    Como testar: agregue por dia da semana / turno. Se o padrão for cíclico,
    modele — não combata.

(d) Há drift genuíno e contínuo (a fábrica realmente mudou).
    Como testar: é a hipótese de EXCLUSÃO. Só se sustenta depois que
    (a), (b) e (c) foram descartadas.

A ordem importa: sempre suspeite do seu instrumento antes de suspeitar
do mundo.
""",
}
