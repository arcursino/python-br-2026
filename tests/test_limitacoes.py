"""
tests/test_limitacoes.py — o que a suíte NÃO pode provar, provado por teste.

Este é o arquivo mais incomum do repositório, e o de maior valor didático.

A maioria das suítes documenta o que o sistema FAZ. Esta documenta o que ele
NÃO FAZ — usando `@pytest.mark.xfail(strict=True)`, que:

  • falha o CI se o teste passar inesperadamente (`strict=True`);
  • transforma "limitação conhecida" de comentário em contrato executável;
  • impede que alguém "conserte" a limitação sem atualizar a documentação.

Um monitor de drift honesto precisa saber dizer onde é cego. Um que não sabe
está apenas silencioso — e silêncio e cegueira produzem exatamente o mesmo
dashboard verde.
"""

from __future__ import annotations

import numpy as np
import pytest

from driftkit.detectors import DriftDetector
from driftkit.fixtures import CAT, NUM, RANGES, gerar_fixture, janela
from driftkit.modelo import metricas
from driftkit.policy import Acao, Causa, PoliticaRetreino


# =============================================================================
#  Limitação 1 — referência móvel cega a deriva lenta
# =============================================================================
@pytest.mark.xfail(
    strict=True,
    reason=(
        "LIMITAÇÃO CONHECIDA E DOCUMENTADA: com referência móvel (janela recente "
        "em vez de baseline congelado), o detector compara o presente com um "
        "passado que JÁ DERIVOU. Deriva lenta se torna invisível — o sapo cozinha "
        "devagar. É por isso que DriftDetector é frozen=True."
    ),
)
def test_referencia_movel_detecta_deriva_lenta():
    df = gerar_fixture(seed=7)
    ref_movel = df.query("100 <= dia < 110")  # já dentro do TC-3
    det = DriftDetector.from_reference(ref_movel, num=list(NUM), cat=list(CAT))
    assert det.report(janela(df, 118)).n_drift > 0


# =============================================================================
#  Limitação 2 — monitor global não é CEGO, é TARDIO (a autocorreção honesta)
# =============================================================================
def test_monitor_global_alarma_mas_tarde_e_com_magnitude_descartavel(modelo, df):
    """A versão CORRETA do argumento — e ela é mais forte que o exagero.

    É tentador dizer "o monitor global é cego ao problema da CEL-02".
    Não é verdade, e a primeira pessoa que rodar o notebook vai te corrigir.

    O que realmente acontece: o global TAMBÉM cai, só que ~4× menos.
    E uma queda dessa magnitude é exatamente o que qualquer engenheiro
    sênior descarta como variação amostral — com razão.

        A segmentação não transforma o invisível em visível.
        Ela transforma o DESCARTÁVEL em ACIONÁVEL.
    """
    base_g = metricas(modelo, janela(df, 45), num=NUM, cat=CAT).roc_auc
    tc3_g = metricas(modelo, janela(df, 110), num=NUM, cat=CAT).roc_auc

    base_s = metricas(modelo, janela(df, 45).query("equipamento=='CEL-02'"),
                      num=NUM, cat=CAT).roc_auc
    tc3_s = metricas(modelo, janela(df, 110).query("equipamento=='CEL-02'"),
                     num=NUM, cat=CAT).roc_auc

    queda_g, queda_s = base_g - tc3_g, base_s - tc3_s
    assert queda_g > 0, "o global também cai — não afirme que é cego"
    assert queda_s > 3 * queda_g, (
        f"segmentado deve cair muito mais: global={queda_g:.3f} segmento={queda_s:.3f}"
    )


# =============================================================================
#  Limitação 3 — P(X) não tem obrigação de ver concept drift
# =============================================================================
@pytest.mark.xfail(
    strict=True,
    reason=(
        "LIMITAÇÃO ESTRUTURAL, não bug: concept drift é mudança em P(Y|X). "
        "Nenhuma quantidade de vigilância sobre P(X) tem obrigação matemática "
        "de detectá-lo. Saber disto ANTES de montar o dashboard evita construir "
        "um sistema cego por design."
    ),
)
def test_monitor_de_distribuicao_detecta_concept_drift(detector, df):
    rel = detector.report(janela(df, 110).query("equipamento == 'CEL-02'"))
    assert rel.n_drift > 0


# =============================================================================
#  Limitação 4 — janela pequena não é "sem drift"
# =============================================================================
def test_janela_insuficiente_nao_e_confundida_com_ausencia_de_drift(detector, df, piso_aa):
    """`nan > 0.1` é `False`. Sem tratamento explícito, a janela não mensurável
    passa como saudável — o pior modo de falha possível para um monitor."""
    pol = PoliticaRetreino(detector=detector, piso_aa=piso_aa)
    d = pol.avaliar(janela(df, 110).head(20))
    assert d.acao is Acao.INSUFICIENTE
    assert d.exit_code == 30, "30 ≠ 0: 'não sei' precisa ser distinto de 'está tudo bem'"


# =============================================================================
#  A guarda 5 — e a prova de que as outras quatro não bastam
# =============================================================================
class TestGuardaCinco:
    """O clímax intelectual do tutorial, em forma de teste."""

    def _politica(self, detector, piso_aa):
        return PoliticaRetreino(detector=detector, piso_aa=piso_aa, k_de_n=(1, 1))

    def test_tc3_e_bloqueado_por_causa_de_hardware(self, detector, modelo, df, piso_aa):
        pol = self._politica(detector, piso_aa)
        w = janela(df, 110)
        seg = w.query("equipamento == 'CEL-02'")
        d = pol.avaliar(
            w, dia=110, segmento="CEL-02",
            auc_global=metricas(modelo, w, num=NUM, cat=CAT).roc_auc,
            auc_segmento=metricas(modelo, seg, num=NUM, cat=CAT).roc_auc,
        )
        assert d.acao is Acao.BLOQUEAR
        assert d.causa == Causa.HARDWARE
        assert d.exit_code == 20

    def test_as_quatro_guardas_classicas_passariam(self, detector, modelo, df, piso_aa):
        """A tabela que deveria ser um slide inteiro.

        No TC-3, as quatro guardas de mercado NÃO impedem o retreino. Elas
        foram desenhadas para bloquear retreino DESNECESSÁRIO — nenhuma foi
        desenhada para bloquear retreino DANOSO, porque danoso e necessário
        são indistinguíveis por qualquer métrica de ML.
        """
        pol = self._politica(detector, piso_aa)
        w = janela(df, 110)
        d = pol.avaliar(
            w, dia=110,
            auc_global=metricas(modelo, w, num=NUM, cat=CAT).roc_auc,
            auc_segmento=metricas(modelo, w.query("equipamento=='CEL-02'"),
                                  num=NUM, cat=CAT).roc_auc,
        )
        assert d.guardas["5_causa_e_ml"] is False, "só a quinta bloqueia"
        assert d.acao is Acao.BLOQUEAR

    def test_tc4_bloqueado_por_pipeline(self, detector, df, piso_aa):
        d = self._politica(detector, piso_aa).avaliar(janela(df, 128, w=2), dia=128)
        assert d.acao is Acao.BLOQUEAR
        assert d.causa == Causa.PIPELINE

    def test_tc2_nao_dispara_retreino(self, detector, modelo, df, piso_aa):
        """Drift real, magnitude alta, persistente — e ainda assim NÃO retreinar."""
        pol = self._politica(detector, piso_aa)
        w = janela(df, 80)
        d = pol.avaliar(
            w, dia=80,
            auc_global=metricas(modelo, w, num=NUM, cat=CAT).roc_auc,
            auc_segmento=metricas(modelo, w.query("equipamento=='CEL-02'"),
                                  num=NUM, cat=CAT).roc_auc,
        )
        assert d.acao is not Acao.RETREINAR
        assert d.causa == Causa.NEGOCIO

    def test_blip_nao_dispara_retreino(self, detector, df, piso_aa):
        """Guarda 1: um pico isolado é falha de rede, não drift.
        Retreinar num blip é o erro mais caro da lista."""
        pol = PoliticaRetreino(detector=detector, piso_aa=piso_aa, k_de_n=(3, 5))
        pol._historico = [False, False, False, False]
        d = pol.avaliar(janela(df, 128, w=2), dia=128)
        assert d.guardas.get("1_persistencia") is False or d.acao is Acao.BLOQUEAR


# =============================================================================
#  Contrato: falha ruidosa é melhor que silêncio
# =============================================================================
class TestContrato:
    def test_contrato_rejeita_limiar_dentro_do_ruido(self):
        from driftkit.state import Contrato

        with pytest.raises(ValueError, match="ruído"):
            Contrato(
                versao="v0", origem="teste",
                features={"num": ["a"], "cat": []},
                limiares={"psi_piso_aa": 0.20, "psi_alarme": 0.10,
                          "alpha": 0.05, "min_obs": 200},
            )

    def test_contrato_rejeita_suite_vermelha(self):
        from driftkit.state import Contrato

        with pytest.raises(ValueError, match="falhas"):
            Contrato(
                versao="v0", origem="teste",
                features={"num": ["a"], "cat": []},
                limiares={"psi_piso_aa": 0.01, "psi_alarme": 0.05,
                          "alpha": 0.05, "min_obs": 200},
                suite={"pass": 11, "total": 13},
            )

    def test_detector_sem_features_falha(self, referencia):
        with pytest.raises(ValueError, match="sem features"):
            DriftDetector(referencia=referencia)

    def test_detector_e_imutavel(self, detector):
        with pytest.raises(Exception):
            detector.psi_limiar = 0.99  # type: ignore[misc]


# =============================================================================
#  Fator de amplificação — o entregável da ponte Parte I → Parte II
# =============================================================================
def test_comparar_regimes_detecta_limiar_cego():
    """O limiar do sintético, no Bosch, deixa o detector CEGO — não ruidoso.

    Substitui test_fator_de_amplificacao_e_reportado. A premissa antiga era
    que o piso CRESCE do sintético para o real; ele cai, porque
    piso ∝ (1/n + 1/m) e a janela do Bosch é ~15x maior.
    """
    from driftkit.detectors import piso_analitico
    from driftkit.state import comparar_regimes

    v1 = {"n_referencia": 27_000, "k_features_num": 9, "bins": 10,
          "piso_aa": 0.0166, "psi_alarme_sugerido": 0.050,
          "piso_analitico": piso_analitico(13_500, 13_500, bins=10, k_features=9)}
    v2 = {"n_referencia": 160_000, "k_features_num": 160, "bins": 10,
          "piso_aa": 0.0017, "psi_alarme_sugerido": 0.005,
          "piso_analitico": piso_analitico(160_000, 40_000, bins=10, k_features=160)}

    r = comparar_regimes(v1, v2, nome_v1="sintético", nome_v2="Bosch")

    assert r["limiar_v1_em_pisos_de_v2"] > 10
    assert "CEGO" in r["veredicto"]
    assert r["razao_dos_pisos"] < 1.0      # o piso CAI, não cresce


@pytest.mark.xfail(strict=True, reason=(
    "fator_amplificacao() foi removida: partia da premissa falsa de que o "
    "piso de ruído CRESCE do sintético para o real. Ele cai com o tamanho "
    "da janela. Use comparar_regimes()."
))
def test_fator_de_amplificacao_e_reportado():
    from driftkit.state import fator_amplificacao
    fator_amplificacao(0.0166, 0.0017)

