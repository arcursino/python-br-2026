"""
tests/test_detectores.py — a suíte de regressão do detector.

Isto é o que substitui o `def T(id_, desc, cond, obs)` artesanal do notebook
original. Ganhos concretos:

  • `assert` com introspecção do pytest (você vê os valores, sem `print`)
  • `@pytest.mark.parametrize` roda o mesmo teste em 8 seeds sem duplicar código
  • `pytest.approx` distingue tolerância de limiar
  • `@pytest.mark.xfail(strict=True)` DOCUMENTA limitação conhecida — e quebra
    o CI se alguém "consertar" sem atualizar o teste
  • exit code para CI: a suíte pode virar um portão de merge

Rode:  pytest tests/ -v
"""

from __future__ import annotations

import numpy as np
import pytest

from driftkit.detectors import DriftDetector, psi, viola_range
from driftkit.fixtures import (
    CAT,
    NUM,
    RANGES,
    TORQUE_ALVO,
    cpk,
    gerar_fixture,
    janela,
    validar_fixture,
)
from driftkit.modelo import metricas


# =============================================================================
#  TC-0 — a FIXTURE. Antes de testar o detector, teste o gabarito.
# =============================================================================
class TestFixture:
    """Fixture errada faz detector correto parecer errado.

    Este bloco é o que quase todo tutorial de drift esquece: se o dataset
    sintético não contém o que você acha que contém, todos os resultados
    seguintes são ficção.
    """

    def test_fixture_contem_o_que_promete(self, df):
        checks = validar_fixture(df)
        falhas = checks.query("status == 'FAIL'")
        assert falhas.empty, f"fixture inválida:\n{falhas.to_string(index=False)}"

    @pytest.mark.parametrize("seed", [7, 42, 101])
    def test_fixture_valida_em_varios_seeds(self, seed):
        checks = validar_fixture(gerar_fixture(seed=seed))
        assert (checks["status"] == "PASS").all()

    def test_tc3_nao_altera_a_fisica_das_outras_celulas(self, df):
        """O offset é do SENSOR da CEL-02. Vazar para outras células invalidaria
        a prova de que a degradação é localizada."""
        outras = df.query("90 <= dia <= 119 and equipamento != 'CEL-02'")
        assert (outras["torque_real"] - outras["torque_medido"]).abs().max() < 1e-4


# =============================================================================
#  TC-1 — controle negativo. O teste mais importante da suíte.
# =============================================================================
class TestControleNegativo:
    """Um detector que sempre alarma é indistinguível de um detector quebrado.

    O controle negativo é o único teste que separa "detectou" de "alarma sempre",
    e é o primeiro a ser cortado quando o prazo aperta. Não corte.
    """

    def test_regime_estavel_nao_alarma(self, detector, df):
        rel = detector.report(janela(df, 45))
        assert rel.n_drift == 0, (
            f"falso positivo em regime estável: {rel.top_features}\n{rel.resumo()}"
        )

    def test_controle_negativo_robusto_a_seed(self, seed_robustez):
        """Roda em 8 fixtures independentes. Um seed passando não é um teste."""
        d = gerar_fixture(seed=seed_robustez)
        det = DriftDetector.from_reference(d.query("dia < 30"), num=list(NUM), cat=list(CAT))
        det = det.com(psi_limiar=det.calibrar_piso(n_repeticoes=15,
                                                   seed=seed_robustez)["psi_alarme_sugerido"])
        assert det.report(janela(d, 45)).n_drift == 0

    def test_aa_contra_a_propria_referencia(self, detector):
        """Teste A/A: o detector alarma contra a referência que o criou?

        Se sim, o problema não é o processo — é o instrumento. Este é o teste
        de 15 minutos que vale como call-to-action da palestra.
        """
        piso = detector.calibrar_piso(n_repeticoes=20)
        assert piso["piso_aa"] < detector.psi_limiar, (
            f"o piso de ruído ({piso['piso_aa']:.5f}) está acima do limiar "
            f"({detector.psi_limiar:.5f}): alarme perpétuo garantido"
        )


# =============================================================================
#  TC-2 × TC-3 — a prova cruzada. O experimento fundamental do tutorial.
# =============================================================================
class TestProvaCruzada:
    """Duas situações que exigem respostas OPOSTAS.

    Um detector que responde igual às duas é cego — não importa quão
    sofisticada seja a estatística por trás dele.
    """

    def test_tc2_covariate_grita_em_px(self, detector, df):
        rel = detector.report(janela(df, 80))
        assert rel.n_drift > 0, "a campanha de R20 muda o mix: P(X) TEM que alarmar"
        assert "aro" in rel.top_features

    def test_tc2_nao_degrada_performance(self, modelo, df):
        """O ponto contraintuitivo: P(X) grita e o modelo está intacto.

        Se P(Y|X) não mudou, a fronteira de decisão continua correta.
        Retreinar aqui é gastar GPU para não mudar nada.
        """
        base = metricas(modelo, janela(df, 45), num=NUM, cat=CAT)
        tc2 = metricas(modelo, janela(df, 80), num=NUM, cat=CAT)
        assert tc2.roc_auc == pytest.approx(base.roc_auc, abs=0.06), (
            f"AUC base={base.roc_auc:.3f} TC-2={tc2.roc_auc:.3f}"
        )

    def test_tc3_concept_fica_mudo_em_px(self, detector, df):
        """O espelho do teste anterior — e o motivo de existir a camada 3.

        O offset desloca a média de 10 Nm num range de ~40. É detectável se
        você procurar por ele, mas fica ABAIXO de um limiar calibrado para não
        gerar falso positivo. É esse o dilema: o limiar que te protege do
        alarme falso é o mesmo que te cega ao concept drift.
        """
        rel = detector.report(janela(df, 110))
        assert rel.efeito_max < 10 * detector.psi_limiar

    def test_tc3_colapsa_a_performance_do_segmento(self, modelo, df):
        seg = janela(df, 110).query("equipamento == 'CEL-02'")
        base = janela(df, 45).query("equipamento == 'CEL-02'")
        m_base = metricas(modelo, base, num=NUM, cat=CAT)
        m_tc3 = metricas(modelo, seg, num=NUM, cat=CAT)
        assert m_tc3.roc_auc < 0.55, (
            f"o modelo no segmento deve virar quase-aleatório: tc3={m_tc3.roc_auc:.3f}"
        )
        assert m_tc3.roc_auc < m_base.roc_auc - 0.10, (
            f"esperava colapso localizado: base={m_base.roc_auc:.3f} tc3={m_tc3.roc_auc:.3f}"
        )

    def test_a_diagonal(self, detector, modelo, df):
        """O tutorial inteiro em um assert.

        TC-2: efeito ALTO em P(X),  performance PRESERVADA
        TC-3: efeito BAIXO em P(X), performance COLAPSADA

        A diagonal invertida é o resultado que reorganiza a prática de MLOps.
        """
        ef2 = detector.report(janela(df, 80)).efeito_max
        ef3 = detector.report(janela(df, 110)).efeito_max
        auc2 = metricas(modelo, janela(df, 80).query("equipamento=='CEL-02'"),
                        num=NUM, cat=CAT).roc_auc
        auc3 = metricas(modelo, janela(df, 110).query("equipamento=='CEL-02'"),
                        num=NUM, cat=CAT).roc_auc
        assert ef2 > ef3, "TC-2 deve mover P(X) mais que TC-3"
        assert auc3 < auc2, "e ainda assim TC-3 é o que machuca"


# =============================================================================
#  TC-4 — data quality
# =============================================================================
class TestDataQuality:
    def test_range_pega_troca_de_unidade(self, df):
        fora = viola_range(janela(df, 128, w=2), RANGES)
        assert fora["pressao_psi"] > 0.95

    def test_range_nao_alarma_em_regime_estavel(self, df):
        fora = viola_range(janela(df, 45), RANGES)
        assert max(fora.values()) < 0.02


# =============================================================================
#  Camada 4 — o KPI que faz a fábrica parar
# =============================================================================
class TestCapabilidade:
    def test_baseline_e_processo_conforme(self, df):
        """Cpk ≥ 1.33 no baseline. Se o processo já fosse incapaz, a história
        seria outra — e um engenheiro de qualidade na plateia perguntaria."""
        base = df.query("dia < 30")["torque_real"]
        assert cpk(base) >= 1.33, f"baseline com Cpk={cpk(base):.2f}"

    def test_tc3_destroi_a_capabilidade(self, df):
        tc3 = df.query("90 <= dia <= 119 and equipamento == 'CEL-02'")["torque_real"]
        assert cpk(tc3) < 0.8, (
            f"Cpk={cpk(tc3):.2f} — a tradução de 'AUC caiu' para a linguagem "
            "que faz a fábrica parar"
        )


# =============================================================================
#  Propriedades matemáticas do PSI
# =============================================================================
class TestPSI:
    def test_psi_de_amostra_consigo_mesma_e_pequeno(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 5000)
        assert psi(x[:2500], x[2500:]) < 0.05

    def test_psi_cresce_com_o_deslocamento(self):
        rng = np.random.default_rng(0)
        ref = rng.normal(0, 1, 5000)
        vals = [psi(ref, rng.normal(d, 1, 5000)) for d in (0.1, 0.5, 1.0, 2.0)]
        assert all(a < b for a, b in zip(vals, vals[1:])), f"não monotônico: {vals}"

    def test_psi_devolve_nan_quando_nao_mensuravel(self):
        assert np.isnan(psi(np.arange(5), np.arange(5)))

    def test_psi_usa_bins_da_referencia(self):
        """Recalcular bins nos dados atuais é o bug silencioso clássico:
        o PSI tenderia a zero justamente quando há drift."""
        rng = np.random.default_rng(0)
        assert psi(rng.normal(0, 1, 8000), rng.normal(3, 1, 8000)) > 1.0
