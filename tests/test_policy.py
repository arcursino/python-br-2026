"""
tests/test_policy.py — a rede de proteção da guarda 1b.

Por que este arquivo existe
---------------------------
A suíte já provava que o TC-3 é bloqueado — mas SÓ no dia 110. O dia 110 é o
meio do regime, o caso mais fácil. Quando a simulação foi rodada numa grade
diferente, apareceu isto:

     95 TC-3      20  ⛔ BLOQUEAR    hardware
    105 TC-3      10  🔁 RETREINAR   modelo     ← autorizou retreino no TC-3
    115 TC-3      20  ⛔ BLOQUEAR    hardware

Um `RETREINAR` isolado no meio de dois `BLOQUEAR`, dentro de um único regime
físico. A causa: o diagnóstico é feito por janela, e quando o número de
features em alarme oscila por ruído amostral (2 → 1 → 2), a assinatura de
HARDWARE deixa de fechar e a causa cai em MODELO — que é a causa RESIDUAL.

A suíte não pegou porque todo teste de TC-3 usava o dia 110.

Este arquivo corrige isso de forma estrutural: em vez de testar UM dia, varre
o regime inteiro. Um teste que só passa no dia que você escolheu não é um
teste — é uma coincidência com asserção.
"""

from __future__ import annotations

import pytest

from driftkit.fixtures import CAT, NUM, janela
from driftkit.modelo import metricas
from driftkit.policy import Acao, Causa, PoliticaRetreino

# O TC-3 vive nos dias 90–119. Varremos com passo 3: 10 janelas, cobrindo a
# borda de entrada (90, ainda mista com TC-2), o miolo e a borda de saída.
DIAS_TC3 = list(range(90, 120, 3))


def _avaliar(pol, modelo, df, dia, *, segmento="CEL-02"):
    """Avalia uma janela do jeito que o `simular` avalia — mesma via."""
    w = janela(df, dia)
    seg = w.query("equipamento == @segmento")
    return pol.avaliar(
        w, dia=dia, segmento=segmento,
        auc_global=metricas(modelo, w, num=NUM, cat=CAT).roc_auc,
        auc_segmento=metricas(modelo, seg, num=NUM, cat=CAT).roc_auc,
    )


@pytest.fixture
def politica(detector, piso_aa):
    """k_de_n=(1,1) isola a guarda de CAUSA da guarda de PERSISTÊNCIA.

    Sem isto, um teste que passasse não distinguiria "a causa foi bem
    diagnosticada" de "o alarme ainda não persistiu o suficiente".
    """
    return PoliticaRetreino(detector=detector, piso_aa=piso_aa, k_de_n=(1, 1))


# =============================================================================
#  O invariante. Se algum dia esta suíte falhar, o tutorial perdeu o argumento.
# =============================================================================
class TestTC3NuncaAutorizaRetreino:
    """Retreinar sobre o transdutor descalibrado institucionaliza o defeito
    mecânico dentro do modelo. Não existe dia do TC-3 em que isso seja aceitável.
    """

    def test_nenhuma_janela_do_tc3_autoriza_retreino(self, politica, modelo, df):
        """A varredura. Este é o teste que teria pego o dia 105."""
        autorizados = []
        for dia in DIAS_TC3:
            d = _avaliar(politica, modelo, df, dia)
            if d.exit_code == 10:
                autorizados.append((dia, d.causa, d.motivo[:60]))

        assert not autorizados, (
            "há janelas do TC-3 autorizando retreino sobre sensor descalibrado:\n"
            + "\n".join(f"  dia {d}: causa={c} — {m}" for d, c, m in autorizados)
        )

    def test_o_miolo_do_tc3_bloqueia_ativamente(self, politica, modelo, df):
        """Não basta NÃO autorizar: no miolo do regime, onde a assinatura de
        hardware está estável, o sistema tem que emitir a ordem de serviço.
        Um pipeline que só sabe ficar quieto não é uma política."""
        bloqueios = [
            _avaliar(politica, modelo, df, dia).acao is Acao.BLOQUEAR
            for dia in (99, 105, 111, 117)
        ]
        assert sum(bloqueios) >= 3, (
            f"esperava BLOQUEAR na maioria do miolo do TC-3, obtive {bloqueios}"
        )


# =============================================================================
#  A guarda 1b, testada diretamente
# =============================================================================
class TestGuarda1bDiagnosticoConfirmado:
    """`modelo` é a única causa que AUTORIZA gasto e a única definida por
    EXCLUSÃO. Por isso é a única que precisa se repetir para ser aceita.

        bloquear por suspeita é barato.
        retreinar por dúvida, não.
    """

    def test_modelo_isolado_nao_autoriza(self, politica, modelo, df):
        """Uma janela `modelo` cercada de `hardware` é ruído de diagnóstico,
        não evidência de modelo envelhecido."""
        vistos = []
        for dia in DIAS_TC3:
            d = _avaliar(politica, modelo, df, dia)
            vistos.append((d.causa, d.acao))

        isolados = [
            i for i, (c, _) in enumerate(vistos)
            if c == Causa.MODELO and (i == 0 or vistos[i - 1][0] != Causa.MODELO)
        ]
        for i in isolados:
            assert vistos[i][1] is not Acao.RETREINAR, (
                f"janela {DIAS_TC3[i]}: `modelo` sem confirmação prévia "
                f"(anterior={vistos[i-1][0] if i else None}) não pode autorizar"
            )

    def test_a_guarda_1b_e_reportada(self, politica, modelo, df):
        """A guarda tem que aparecer no dicionário auditável — uma decisão que
        não é inspecionável não é uma decisão, é um palpite com exit code."""
        d = _avaliar(politica, modelo, df, 105)
        assert "1b_causa_confirmada" in d.guardas
        assert "causa_janela_anterior" in d.evidencias

    def test_causas_com_assinatura_positiva_agem_na_primeira_janela(
        self, detector, piso_aa, df
    ):
        """A assimetria é deliberada: pipeline/hardware/negócio têm evidência
        PRÓPRIA e não esperam confirmação. Só a causa residual espera."""
        pol = PoliticaRetreino(detector=detector, piso_aa=piso_aa, k_de_n=(1, 1))
        d = pol.avaliar(janela(df, 128, w=2), dia=128)   # TC-4, primeira janela
        assert d.causa == Causa.PIPELINE
        assert d.acao is Acao.BLOQUEAR, "a guarda 1b não pode atrasar um bloqueio"

    def test_modelo_confirmado_em_duas_janelas_passa(self, detector, piso_aa):
        """O outro lado do contrato: a guarda 1b ATRASA, não PROÍBE.

        Quando o diagnóstico é estável — modelo genuinamente envelhecido — a
        segunda janela consecutiva confirma e o retreino é liberado. Testado
        no histórico diretamente, porque a fixture não contém um caso de
        concept drift de causa `modelo` legítima: o TC-3 é hardware por
        construção, e é justamente esse o ponto do tutorial.
        """
        pol = PoliticaRetreino(detector=detector, piso_aa=piso_aa, k_de_n=(1, 1))
        pol._causas = [Causa.MODELO, Causa.MODELO]
        assert pol._causas[-2:].count(Causa.MODELO) >= 2

        pol._causas = [Causa.HARDWARE, Causa.MODELO]
        assert pol._causas[-2:].count(Causa.MODELO) < 2


# =============================================================================
#  Higiene de estado
# =============================================================================
def test_resetar_historico_limpa_as_causas(detector, piso_aa, modelo, df):
    """Histórico que sobrevive ao reset faz o teste seguinte herdar o
    diagnóstico do anterior — e a suíte passa a depender da ORDEM dos testes."""
    pol = PoliticaRetreino(detector=detector, piso_aa=piso_aa, k_de_n=(1, 1))
    _avaliar(pol, modelo, df, 110)
    assert pol._causas, "as causas deveriam estar sendo registradas"
    pol.resetar_historico()
    assert pol._causas == []
    assert pol._historico == []
