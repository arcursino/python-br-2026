"""
driftkit.policy — as cinco guardas de retreino.

Caminho no repositório: src/driftkit/policy.py

    Um pipeline que sabe se RECUSAR a retreinar é mais maduro
    que um que retreina rápido.

As quatro primeiras guardas são padrão de mercado. Todas foram desenhadas para
impedir retreino DESNECESSÁRIO (blip de rede, sazonalidade, ruído amostral).

A quinta é a contribuição deste tutorial, e existe porque as outras quatro
falham juntas num caso específico: quando o drift é REAL, PERSISTENTE, de
MAGNITUDE ALTA e o retreino MELHORA a métrica offline — e ainda assim
retreinar é a pior decisão possível.

É o TC-3: sensor descalibrado. Retreinar sobre dado de sensor mentiroso
institucionaliza o defeito mecânico dentro do modelo. O dashboard fica verde,
o Cpk vai a 0.35, e quando a metrologia finalmente calibrar a ferramenta o
modelo quebra — porque ele aprendeu a mentira como se fosse o normal.

    Nenhuma métrica de ML distingue "retreino necessário" de
    "retreino danoso". A distinção é FÍSICA, e por isso a guarda 5
    é um BLOQUEIO, não um score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
import pandas as pd

from .detectors import DriftDetector, RelatorioDrift, viola_range

__all__ = ["Acao", "Causa", "Decisao", "PoliticaRetreino"]


class Acao(IntEnum):
    """A ação decidida — e o EXIT CODE do CLI.

    O exit code não é detalhe de implementação: é o contrato com a esteira de
    CI/CD. É ele que faz a guarda 5 sair do slide e virar infraestrutura.
    """

    NADA = 0            # sem drift acionável
    RETREINAR = 10      # drift acionável, causa compatível com ML → aprovado
    BLOQUEAR = 20       # drift acionável, causa NÃO-ML → ordem de serviço
    INSUFICIENTE = 30   # sem dado suficiente para decidir (≠ sem drift)


class Causa(str):
    """Hipótese de causa. String simples de propósito: vai para JSON e para ticket."""

    NENHUMA = "nenhuma"
    NEGOCIO = "negocio"       # campanha, mix, sazonalidade → registrar
    MODELO = "modelo"         # o mundo mudou de verdade → retreinar
    HARDWARE = "hardware"     # sensor/ferramenta → metrologia (BLOQUEIA)
    PIPELINE = "pipeline"     # schema/unidade/ETL → engenharia (BLOQUEIA)
    INDEFINIDA = "indefinida"


CAUSAS_QUE_BLOQUEIAM = frozenset({Causa.HARDWARE, Causa.PIPELINE})

# Quantos múltiplos do limiar calibrado ainda contam como "P(X) calmo".
#
# Não é um número escolhido no olho: é a região ENTRE o limiar de detecção e a
# magnitude de um covariate drift genuíno. Na fixture, com psi_limiar≈0.0102:
#
#     TC-3 (concept, exige BLOQUEIO) : efeito máx 0.0178   ≈  1.7× o limiar
#     TC-2 (covariate, exige registro): efeito máx 1.9136  ≈ 187× o limiar
#
# Duas ordens de grandeza separam os dois casos. Qualquer teto entre ~3× e ~50×
# os distingue; 10× fica no meio do vão, em escala logarítmica.
FATOR_PX_CALMO = 10.0

# Quantas features podem estar em drift e a janela ainda ser considerada calma.
# O TC-3 vaza UMA feature (`torque_medido`): a compensação do operador cancela
# 87.5% do offset, não 100%. Exigir n_drift == 0 é exigir que o resíduo físico
# não exista.
MAX_FEATURES_EM_DRIFT_CALMO = 1


@dataclass(frozen=True, slots=True)
class Decisao:
    """O resultado de uma avaliação de política. Serializável, auditável."""

    acao: Acao
    causa: str
    motivo: str
    guardas: dict[str, bool]
    evidencias: dict[str, float | str | None] = field(default_factory=dict)
    dia: int | None = None
    segmento: str | None = None

    @property
    def exit_code(self) -> int:
        return int(self.acao)

    def to_dict(self) -> dict:
        return {
            "acao": self.acao.name,
            "exit_code": self.exit_code,
            "causa": self.causa,
            "motivo": self.motivo,
            "guardas": self.guardas,
            "evidencias": self.evidencias,
            "dia": self.dia,
            "segmento": self.segmento,
        }

    def __str__(self) -> str:
        marcas = "  ".join(f"{k}={'✅' if v else '⛔'}" for k, v in self.guardas.items())
        icone = {Acao.NADA: "🟢", Acao.RETREINAR: "🔁",
                 Acao.BLOQUEAR: "⛔", Acao.INSUFICIENTE: "⚪"}[self.acao]
        return (
            f"{icone} decisão : {self.acao.name}  (exit {self.exit_code})\n"
            f"   causa   : {self.causa}\n"
            f"   motivo  : {self.motivo}\n"
            f"   guardas : {marcas}"
        )


@dataclass
class PoliticaRetreino:
    """Aplica as cinco guardas em cascata.

    Parameters
    ----------
    detector : DriftDetector
        Já calibrado. A política NÃO calibra nada — separação de papéis.
    k_de_n : (int, int)
        Guarda 1 (persistência): exige k alarmes nas últimas n janelas.
        Mata blip. Padrão (3, 5).
    cooldown_janelas : int
        Guarda 2: nº mínimo de janelas entre dois retreinos. Mata retreino
        perpétuo por sazonalidade.
    fator_piso : float
        Guarda 3: exige efeito > fator × piso medido em teste A/A.
        É o que substitui o "PSI > 0.1" folclórico.
    ganho_minimo : float
        Guarda 4 (shadow validation): challenger só promove se superar o
        champion por esta margem em dado de validação.
    contextos_conhecidos : dict
        Guarda 2b: janelas de calendário com causa já conhecida (campanha
        planejada, parada programada). Alarme aqui é registro, não incidente.
    """

    detector: DriftDetector
    piso_aa: float
    k_de_n: tuple[int, int] = (3, 5)
    cooldown_janelas: int = 14
    fator_piso: float = 3.0
    ganho_minimo: float = 0.01
    contextos_conhecidos: dict[str, tuple[int, int]] = field(default_factory=dict)

    _historico: list[bool] = field(default_factory=list, repr=False)
    _ultimo_retreino: int | None = field(default=None, repr=False)

    # =========================================================================
    #  diagnóstico de causa — o insumo da guarda 5
    # =========================================================================
    def diagnosticar(
        self,
        atual: pd.DataFrame,
        relatorio: RelatorioDrift,
        *,
        dia: int | None = None,
        auc_global: float | None = None,
        auc_segmento: float | None = None,
    ) -> tuple[str, str]:
        """Classifica a causa provável por ASSINATURA, não por magnitude.

        Cada assinatura é uma conjunção verificável, e a ordem importa:
        vai do mais barato e mais conclusivo para o mais especulativo.

        Devolve (causa, evidência textual).
        """
        # ---- assinatura PIPELINE: violação de range de engenharia -----------
        # A mais barata e a mais conclusiva. Sensor não sai de faixa física
        # sozinho: ou o ETL trocou a unidade, ou o schema mudou.
        fora = viola_range(atual, self.detector.ranges)
        graves = {f: v for f, v in fora.items() if v > 0.10}
        if graves:
            f, v = max(graves.items(), key=lambda kv: kv[1])
            return Causa.PIPELINE, f"{v:.0%} dos registros de `{f}` fora do range de engenharia"

        # ---- assinatura de CONTEXTO CONHECIDO -------------------------------
        if dia is not None:
            for nome, (ini, fim) in self.contextos_conhecidos.items():
                if ini <= dia <= fim:
                    return Causa.NEGOCIO, f"janela dentro do contexto planejado `{nome}`"

        # ---- assinatura HARDWARE: localizada + P(X) calmo -------------------
        # A marca do sensor descalibrado: a degradação é de UM segmento e o
        # monitor de distribuição está calmo (ou quase). Se P(X) grita E a
        # performance cai, é mais provável que o mundo tenha mudado de fato.
        if auc_global is not None and auc_segmento is not None:
            # (a) A degradação é LOCALIZADA?
            #
            # Critério de RAZÃO, não de diferença. Mede quanto do poder
            # discriminativo acima do acaso sobrou no segmento em relação ao
            # que sobrou na janela agregada.
            #
            # Por que não `auc_global - auc_segmento > 0.15`: o TC-3 arrasta a
            # AUC global para baixo junto com a do segmento, o que COMPRIME a
            # diferença exatamente quando o problema é mais grave. Um limiar
            # absoluto sobre a diferença de duas métricas que caem juntas é
            # frágil por construção — precisa ser recalibrado a cada mudança de
            # magnitude da fixture, e falha silenciosamente quando não é.
            lift_g = max(auc_global - 0.5, 1e-6)
            lift_s = max(auc_segmento - 0.5, 0.0)
            razao_lift = lift_s / lift_g
            localizada = razao_lift < 0.35 and auc_segmento < 0.55

            # (b) P(X) está calmo?
            #
            # "Calmo" é RELATIVO, e essa é a correção mais importante deste
            # módulo. Concept drift NÃO é obrigatoriamente invisível em P(X):
            # no TC-3 o resíduo não compensado pelo operador desloca o
            # `torque_medido` em -0.45 Nm, o que dá efeito 0.0178 com
            # p ≈ 8e-07. É pequeno, é real, e é ESTATISTICAMENTE INEGÁVEL.
            #
            # Exigir silêncio ABSOLUTO (n_drift == 0) torna a assinatura refém
            # do tamanho da amostra: com n suficiente, todo resíduo físico vira
            # significativo, e a assinatura de hardware deixa de existir
            # justamente nas instalações mais bem instrumentadas.
            #
            # O que separa TC-3 de TC-2 não é presença vs. ausência de sinal em
            # P(X) — é a ORDEM DE GRANDEZA do sinal (0.018 vs 1.91, ~107×).
            calmo_em_magnitude = (
                not np.isfinite(relatorio.efeito_max)
                or relatorio.efeito_max < FATOR_PX_CALMO * self.detector.psi_limiar
            )
            px_calmo = (
                relatorio.n_drift <= MAX_FEATURES_EM_DRIFT_CALMO
                and calmo_em_magnitude
            )

            if localizada and px_calmo:
                return Causa.HARDWARE, (
                    f"degradação LOCALIZADA (AUC segmento {auc_segmento:.3f} vs "
                    f"global {auc_global:.3f}; lift residual {razao_lift:.0%}) "
                    f"com P(X) calmo ({relatorio.n_drift} feature(s), efeito máx "
                    f"{relatorio.efeito_max:.4f} = "
                    f"{relatorio.efeito_max / self.detector.psi_limiar:.1f}× o limiar) — "
                    "assinatura de instrumentação, não de modelo"
                )

        # ---- assinatura NEGÓCIO: P(X) grita e performance intacta ----------
        if relatorio.n_drift > 0 and auc_global is not None and auc_segmento is not None:
            if (auc_global - auc_segmento) < 0.05:
                return Causa.NEGOCIO, (
                    f"{relatorio.n_drift} feature(s) em drift com performance preservada "
                    "— mix/campanha, não degradação"
                )

        # ---- resta: o mundo mudou ------------------------------------------
        if relatorio.n_drift > 0:
            return Causa.MODELO, (
                f"drift persistente em {relatorio.n_drift} feature(s) "
                "sem assinatura física identificada"
            )
        return Causa.NENHUMA, "sem drift acionável"

    # =========================================================================
    #  as cinco guardas
    # =========================================================================
    def avaliar(
        self,
        atual: pd.DataFrame,
        *,
        dia: int | None = None,
        segmento: str | None = None,
        auc_global: float | None = None,
        auc_segmento: float | None = None,
        ganho_shadow: float | None = None,
    ) -> Decisao:
        """Aplica as guardas em cascata e devolve uma `Decisao`."""
        guardas: dict[str, bool] = {}
        ev: dict[str, float | str | None] = {}

        # --- guarda 0: é mensurável? ----------------------------------------
        if len(atual) < self.detector.min_obs:
            return Decisao(
                acao=Acao.INSUFICIENTE,
                causa=Causa.INDEFINIDA,
                motivo=(f"janela com {len(atual)} obs (< min_obs={self.detector.min_obs}). "
                        "Sem decisão. ATENÇÃO: isto NÃO é 'sem drift'."),
                guardas={"0_mensuravel": False},
                dia=dia,
                segmento=segmento,
            )
        guardas["0_mensuravel"] = True

        rel = self.detector.report(atual)
        alarmou = rel.n_drift > 0
        self._historico.append(alarmou)
        ev["n_drift"] = rel.n_drift
        ev["efeito_max"] = None if not np.isfinite(rel.efeito_max) else round(rel.efeito_max, 6)
        ev["features"] = ", ".join(rel.top_features[:5]) or None

        # --- guarda 1: persistência (k de n) --------------------------------
        k, n = self.k_de_n
        ultimas = self._historico[-n:]
        persistente = sum(ultimas) >= k
        guardas["1_persistencia"] = persistente
        ev["janelas_alarmadas"] = f"{sum(ultimas)}/{len(ultimas)}"

        # --- guarda 3: magnitude sobre o piso MEDIDO ------------------------
        limiar_efetivo = self.fator_piso * self.piso_aa
        acima_do_piso = bool(np.isfinite(rel.efeito_max) and rel.efeito_max > limiar_efetivo)
        guardas["3_magnitude"] = acima_do_piso
        ev["limiar_efetivo"] = round(limiar_efetivo, 6)

        # --- diagnóstico (insumo das guardas 2 e 5) -------------------------
        causa, evidencia = self.diagnosticar(
            atual, rel, dia=dia, auc_global=auc_global, auc_segmento=auc_segmento
        )
        ev["evidencia_causa"] = evidencia

        # --- guarda 2: contexto conhecido -----------------------------------
        contexto_ok = causa != Causa.NEGOCIO
        guardas["2_contexto"] = contexto_ok

        # --- guarda 5: BLOQUEIO por causa não-ML ----------------------------
        # Vem ANTES da guarda 4 de propósito: shadow validation custa treinar
        # um modelo. Se a causa é física, não gastamos GPU para descobrir que
        # o modelo "melhorou" aprendendo o defeito.
        causa_e_ml = causa not in CAUSAS_QUE_BLOQUEIAM
        guardas["5_causa_e_ml"] = causa_e_ml

        if not causa_e_ml:
            destino = ("metrologia / manutenção" if causa == Causa.HARDWARE
                       else "engenharia de dados")
            return Decisao(
                acao=Acao.BLOQUEAR,
                causa=causa,
                motivo=(
                    f"RETREINO BLOQUEADO. Causa provável `{causa}` não é corrigível por "
                    f"modelo: {evidencia}. Encaminhar para {destino}. "
                    "Retreinar aqui institucionalizaria o defeito no modelo."
                ),
                guardas=guardas,
                evidencias=ev,
                dia=dia,
                segmento=segmento,
            )

        # --- sem drift acionável --------------------------------------------
        if not alarmou:
            return Decisao(Acao.NADA, causa, "sem alarme nesta janela.",
                           guardas, ev, dia, segmento)
        if not persistente:
            return Decisao(
                Acao.NADA, causa,
                f"alarme não persistente ({ev['janelas_alarmadas']}, exigido {k}/{n}). "
                "Provável blip — retreinar aqui é o erro mais caro da lista.",
                guardas, ev, dia, segmento,
            )
        if not acima_do_piso:
            return Decisao(
                Acao.NADA, causa,
                f"magnitude {rel.efeito_max:.4f} abaixo de {self.fator_piso}× o piso A/A "
                f"({limiar_efetivo:.4f}). Dentro do ruído do próprio instrumento.",
                guardas, ev, dia, segmento,
            )
        if not contexto_ok:
            return Decisao(
                Acao.NADA, causa,
                f"drift com causa de negócio conhecida: {evidencia}. Registrar, não retreinar.",
                guardas, ev, dia, segmento,
            )

        # --- guarda 4: shadow validation ------------------------------------
        if ganho_shadow is not None:
            passou = ganho_shadow >= self.ganho_minimo
            guardas["4_shadow"] = passou
            ev["ganho_shadow"] = round(float(ganho_shadow), 6)
            if not passou:
                return Decisao(
                    Acao.NADA, causa,
                    f"challenger não superou o champion (Δ={ganho_shadow:+.4f} < "
                    f"{self.ganho_minimo}). Modelo atual mantido.",
                    guardas, ev, dia, segmento,
                )
        else:
            guardas["4_shadow"] = True  # a ser avaliado pelo comando `retrain`

        if dia is not None and self._ultimo_retreino is not None:
            if dia - self._ultimo_retreino < self.cooldown_janelas:
                guardas["2b_cooldown"] = False
                return Decisao(
                    Acao.NADA, causa,
                    f"cooldown ativo (último retreino no dia {self._ultimo_retreino}, "
                    f"exigido intervalo de {self.cooldown_janelas}).",
                    guardas, ev, dia, segmento,
                )
            guardas["2b_cooldown"] = True

        self._ultimo_retreino = dia
        return Decisao(
            Acao.RETREINAR, causa,
            f"todas as guardas passaram. {evidencia}. Retreino AUTORIZADO.",
            guardas, ev, dia, segmento,
        )

    # =========================================================================
    def resetar_historico(self) -> None:
        self._historico.clear()
        self._ultimo_retreino = None

    @classmethod
    def from_contract(cls, contrato: dict, detector: DriftDetector) -> PoliticaRetreino:
        g = contrato.get("guardas_retreino", {})
        return cls(
            detector=detector,
            piso_aa=contrato["limiares"]["psi_piso_aa"],
            k_de_n=tuple(g.get("k_de_n", (3, 5))),  # type: ignore[arg-type]
            cooldown_janelas=g.get("cooldown_janelas", 14),
            fator_piso=g.get("fator_piso", 3.0),
            ganho_minimo=g.get("ganho_minimo", 0.01),
            contextos_conhecidos={k: tuple(v) for k, v in g.get("contextos", {}).items()},
        )


def tabela_das_quatro_guardas(decisao: Decisao) -> pd.DataFrame:
    """Monta a tabela que é o clímax da palestra.

    Mostra que, no caso TC-3, as QUATRO guardas clássicas PASSAM — e só a
    quinta impede o desastre. Projete isto e faça a pausa de três segundos.
    """
    rotulos = {
        "1_persistencia": "1. Persistência (k de n)",
        "2_contexto": "2. Contexto operacional conhecido",
        "3_magnitude": "3. Magnitude sobre o piso medido",
        "4_shadow": "4. Shadow validation (challenger > champion)",
        "5_causa_e_ml": "5. Diagnóstico não-ML  ← A GUARDA QUE FALTAVA",
    }
    linhas = [
        {"guarda": rotulos[k], "resultado": "✅ passa" if decisao.guardas.get(k) else "⛔ BLOQUEIA"}
        for k in rotulos
        if k in decisao.guardas
    ]
    return pd.DataFrame(linhas)
