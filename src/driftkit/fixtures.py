"""
driftkit.fixtures — o dataset sintético "Tire & Wheel".

Caminho no repositório: src/driftkit/fixtures.py

    Isto NÃO é um dataset de demonstração. É uma FIXTURE DE TESTE.

A diferença é inteira: aqui nós plantamos o drift. Sabemos o dia, a célula, o
mecanismo físico e a resposta correta. É o peso padrão de 1 kg com que se
calibra a balança — e ninguém valida um instrumento contra uma amostra de
valor desconhecido.

Linha do tempo (130 dias, 4 células de aperto):

    dia    1– 29  REFERÊNCIA        janela-base do detector
    dia   30– 59  TC-1  estável     controle NEGATIVO: exigimos SILÊNCIO
    dia   60– 89  TC-2  covariate   campanha de aro R20 (planejada)
                                    → P(X) muda muito, P(Y|X) intacto
    dia   90–119  TC-3  concept     transdutor da CEL-02 com offset +10 Nm
                                    → P(X) quase intacto, P(Y|X) MENTE
    dia  120–124  TC-3' recalibrado metrologia corrige → recuperação imediata
    dia  125–130  TC-4  data qual.  tag do PLC volta a publicar em kPa
                                    → violação de range de engenharia

A prova cruzada TC-2 × TC-3 é o experimento fundamental do tutorial:
duas situações que exigem respostas OPOSTAS e que um detector ingênuo
confunde.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "gerar_fixture",
    "janela",
    "NUM",
    "CAT",
    "FEATS",
    "RANGES",
    "EVENTOS",
    "TORQUE_ALVO",
    "TORQUE_LSL",
    "TORQUE_USL",
    "Evento",
]

# =============================================================================
#  Especificação de engenharia — a fonte da verdade física
# =============================================================================
TORQUE_ALVO = 110.0   # Nm
TORQUE_LSL = 105.0    # limite inferior de especificação
TORQUE_USL = 115.0    # limite superior
TORQUE_SIGMA = 1.15   # dá Cpk ≈ 1.45 no baseline → processo CONFORME

CELULAS = ("CEL-01", "CEL-02", "CEL-03", "CEL-04")
AROS = ("R16", "R17", "R18", "R20")
TURNOS = ("A", "B", "C")
FORNECEDORES = ("FORN-A", "FORN-B", "FORN-C")

NUM = ("torque_medido", "pressao_psi", "temperatura_c", "tempo_ciclo_s", "angulo_giro_deg")
CAT = ("aro", "turno", "fornecedor", "equipamento")
FEATS = (*NUM, *CAT)

# Ranges de engenharia: camada 1 da pirâmide. Generosos de propósito —
# não são limites de controle estatístico, são limites do que é FISICAMENTE
# plausível o sensor reportar. Quem cruza isto tem problema de contrato de
# dados, não de modelo.
RANGES: dict[str, tuple[float, float]] = {
    "torque_medido": (90.0, 130.0),
    "pressao_psi": (25.0, 45.0),
    "temperatura_c": (10.0, 45.0),
    "tempo_ciclo_s": (8.0, 40.0),
    "angulo_giro_deg": (200.0, 460.0),
}

PSI_TO_KPA = 6.894757  # fator da troca de unidade do TC-4


@dataclass(frozen=True, slots=True)
class Evento:
    """Um drift plantado — com gabarito de resposta esperada."""

    tc: str
    dia_inicio: int
    dia_fim: int
    nome: str
    tipo: str            # covariate | concept | data_quality | nenhum
    camada_detecta: int  # camada da pirâmide que pega primeiro
    causa: str           # negocio | hardware | pipeline | nenhuma
    acao_correta: str
    px_deve_alarmar: bool
    perf_deve_cair: bool


EVENTOS: tuple[Evento, ...] = (
    Evento("TC-1", 30, 59, "operação normal", "nenhum", 0, "nenhuma",
           "nada — este é o controle negativo", False, False),
    Evento("TC-2", 60, 89, "campanha de aro R20", "covariate", 2, "negocio",
           "registrar; NÃO retreinar", True, False),
    Evento("TC-3", 90, 119, "offset +10 Nm no transdutor da CEL-02", "concept", 3, "hardware",
           "ordem de serviço para metrologia; BLOQUEAR retreino", False, True),
    Evento("TC-3'", 120, 124, "ferramenta recalibrada", "nenhum", 0, "nenhuma",
           "encerrar o incidente", False, False),
    Evento("TC-4", 125, 130, "tag do PLC publicando em kPa", "data_quality", 1, "pipeline",
           "corrigir o pipeline; BLOQUEAR retreino", True, True),
)


# =============================================================================
#  Gerador
# =============================================================================
def gerar_fixture(
    *,
    seed: int = 7,
    n_por_dia: int = 900,
    dias: int = 130,
    offset_torque: float = 10.0,
    celula_afetada: str = "CEL-02",
) -> pd.DataFrame:
    """Gera a fixture completa, determinística dado `seed`.

    O mecanismo do TC-3, que é o coração do tutorial
    ------------------------------------------------
    A falha é aplicada ao valor *reportado*, não ao processo:

        torque_real     ~ N(110, 1.15)      ← a física NÃO mudou
        torque_medido    = torque_real - 10  ← o SENSOR mente (lê 10 a menos)

    O operador vê 100 Nm, acha que está frouxo, e o aparafusador compensa
    apertando mais. O torque real vai a ~120 Nm: fora da especificação
    superior, gerando trinca no cubo.

    Consequência para o modelo: a feature `torque_medido` continua com uma
    distribuição PARECIDA com a de referência (média deslocada de 10 num range
    de 40) — mas o mapeamento feature → falha inverteu. Isso é concept drift
    puro: P(Y|X) mudou. Nenhum teste sobre P(X) tem obrigação de pegar isso,
    e é exatamente por isso que a suíte precisa de um teste que PROVE a
    limitação em vez de assumi-la.

    Parameters
    ----------
    seed : int
        Um seed que passa não é um teste que passa. Veja
        `tests/test_detectores.py::test_controle_negativo_robusto_a_seed`,
        que roda o controle negativo em 8 seeds diferentes.
    """
    rng = np.random.default_rng(seed)
    partes: list[pd.DataFrame] = []

    for dia in range(1, dias + 1):
        n = int(rng.normal(n_por_dia, n_por_dia * 0.05))
        n = max(n, 200)

        # --- mix de aro: TC-2 é uma campanha planejada de R20 ----------------
        if 60 <= dia <= 89:
            p_aro = [0.05, 0.08, 0.07, 0.80]
        else:
            p_aro = [0.30, 0.30, 0.25, 0.15]
        aro = rng.choice(AROS, n, p=p_aro)

        equipamento = rng.choice(CELULAS, n)
        turno = rng.choice(TURNOS, n, p=[0.40, 0.35, 0.25])
        fornecedor = rng.choice(FORNECEDORES, n, p=[0.5, 0.3, 0.2])

        # --- processo físico (o que REALMENTE acontece no parafuso) ----------
        torque_real = rng.normal(TORQUE_ALVO, TORQUE_SIGMA, n)

        afetada = equipamento == celula_afetada
        if 90 <= dia <= 119:
            # O operador compensa a leitura baixa apertando mais.
            # Compensação parcial e ruidosa — como na vida real.
            torque_real = np.where(
                afetada,
                torque_real + offset_torque * rng.uniform(0.75, 1.0, n),
                torque_real,
            )

        # --- o que o SENSOR reporta -----------------------------------------
        torque_medido = torque_real.copy()
        if 90 <= dia <= 119:
            torque_medido = np.where(afetada, torque_real - offset_torque, torque_real)

        pressao = rng.normal(35.0, 1.6, n)
        temperatura = rng.normal(24.0, 2.2, n) + 3.0 * (turno == "C")
        # aro maior => ciclo mais longo: correlação legítima que o TC-2 move
        base_ciclo = np.select(
            [aro == "R16", aro == "R17", aro == "R18", aro == "R20"],
            [18.0, 19.5, 21.0, 25.5],
        )
        tempo_ciclo = rng.normal(base_ciclo, 1.4)
        angulo = rng.normal(360.0, 18.0, n) + 0.8 * (torque_real - TORQUE_ALVO)

        # --- TC-4: troca de unidade no PLC ----------------------------------
        if dia >= 125:
            pressao = pressao * PSI_TO_KPA

        # --- rótulo: função do MUNDO FÍSICO, nunca da leitura do sensor -----
        # Falha por torque fora de especificação, com margem suave.
        desvio = np.abs(torque_real - TORQUE_ALVO)
        logito = (
            -3.05
            + 0.62 * desvio
            + 0.030 * (temperatura - 24.0)
            + 0.055 * (tempo_ciclo - base_ciclo)
            + 0.35 * (fornecedor == "FORN-C")
            + 0.30 * (turno == "C")            # padrão ESTRUTURAL e estável
        )
        p_falha = 1.0 / (1.0 + np.exp(-logito))
        falha = rng.binomial(1, p_falha)

        partes.append(
            pd.DataFrame(
                {
                    "dia": dia,
                    "equipamento": equipamento,
                    "aro": aro,
                    "turno": turno,
                    "fornecedor": fornecedor,
                    "torque_medido": torque_medido.astype(np.float32),
                    "torque_real": torque_real.astype(np.float32),  # ⚠️ só para gabarito
                    "pressao_psi": pressao.astype(np.float32),
                    "temperatura_c": temperatura.astype(np.float32),
                    "tempo_ciclo_s": tempo_ciclo.astype(np.float32),
                    "angulo_giro_deg": angulo.astype(np.float32),
                    "falha": falha.astype(np.int8),
                }
            )
        )

    df = pd.concat(partes, ignore_index=True)
    df["periodo"] = pd.cut(
        df["dia"],
        bins=[0, 29, 59, 89, 119, 124, dias],
        labels=["REF", "TC-1", "TC-2", "TC-3", "TC-3'", "TC-4"],
    )
    return df


def janela(df: pd.DataFrame, dia: int, *, w: int = 3) -> pd.DataFrame:
    """Janela deslizante de `w` dias terminando em `dia` (inclusive).

    A largura da janela é um TRADE-OFF, não um detalhe:
      janela estreita  → detecta rápido, alarme falso frequente
      janela larga     → estável, atraso de detecção maior
    Ela deve ser escolhida com a mesma seriedade do limiar — e testada.
    """
    return df.loc[(df["dia"] > dia - w) & (df["dia"] <= dia)]


def gabarito(dia: int) -> Evento | None:
    """Qual evento está ativo neste dia? O ground truth que o dado real não tem."""
    for ev in EVENTOS:
        if ev.dia_inicio <= dia <= ev.dia_fim:
            return ev
    return None


def validar_fixture(df: pd.DataFrame) -> pd.DataFrame:
    """Testa a FIXTURE, não o detector.

    Fixture errada faz o detector parecer errado. Antes de acreditar em
    qualquer resultado, verificamos que plantamos o que dissemos que plantamos.
    """
    checks: list[dict] = []

    def add(nome: str, cond: bool, obs) -> None:
        checks.append({"check": nome, "status": "PASS" if cond else "FAIL", "observado": obs})

    ref = df.query("dia < 30")
    prev = ref["falha"].mean()
    add("prevalência de referência em 5–45%", 0.05 < prev < 0.45, f"{prev:.1%}")

    r20_tc2 = df.query("60 <= dia <= 89")["aro"].eq("R20").mean()
    add("TC-2 tem ≥70% de R20", r20_tc2 >= 0.70, f"{r20_tc2:.1%}")

    tc3 = df.query("90 <= dia <= 119 and equipamento == 'CEL-02'")
    gap = (tc3["torque_real"] - tc3["torque_medido"]).mean()
    add("TC-3: sensor lê ~10 Nm a menos", 8.0 < gap < 12.0, f"{gap:.2f} Nm")

    outras = df.query("90 <= dia <= 119 and equipamento != 'CEL-02'")
    gap_o = (outras["torque_real"] - outras["torque_medido"]).abs().mean()
    add("TC-3 afeta SÓ a CEL-02", gap_o < 0.01, f"{gap_o:.4f} Nm")

    add(
        "TC-4: pressão fora do range de engenharia",
        df.query("dia >= 125")["pressao_psi"].gt(RANGES["pressao_psi"][1]).mean() > 0.95,
        f"{df.query('dia >= 125')['pressao_psi'].mean():.0f} (esperado ~241)",
    )

    f_tc1 = df.query("30 <= dia <= 59")["falha"].mean()
    add("TC-1 tem prevalência igual à referência (±3pp)", abs(f_tc1 - prev) < 0.03,
        f"{f_tc1:.1%} vs {prev:.1%}")

    f_tc3 = df.query("90 <= dia <= 119 and equipamento == 'CEL-02'")["falha"].mean()
    add("TC-3 eleva a falha na CEL-02", f_tc3 > prev * 1.5, f"{f_tc3:.1%} vs {prev:.1%}")

    return pd.DataFrame(checks)


def cpk(serie: pd.Series | np.ndarray, lsl: float = TORQUE_LSL, usl: float = TORQUE_USL) -> float:
    """Índice de capabilidade do processo. Cpk ≥ 1.33 é o mínimo automotivo.

    Métrica da CAMADA 4 (KPI de negócio). Está aqui porque é ela que
    transforma "a AUC caiu 0.37" em "12 mil peças fora de especificação" —
    a única frase que faz a fábrica parar.
    """
    s = np.asarray(serie, dtype=float)
    s = s[np.isfinite(s)]
    if s.size < 30:
        return np.nan
    mu, sd = s.mean(), s.std(ddof=1)
    if sd == 0:
        return np.nan
    return float(min((usl - mu) / (3 * sd), (mu - lsl) / (3 * sd)))


def ppm_fora_spec(serie: pd.Series | np.ndarray,
                  lsl: float = TORQUE_LSL, usl: float = TORQUE_USL) -> float:
    """Partes por milhão fora de especificação — a linguagem da qualidade."""
    s = np.asarray(serie, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return np.nan
    return float(((s < lsl) | (s > usl)).mean() * 1e6)