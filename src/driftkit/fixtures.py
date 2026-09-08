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
    dia   90–119  TC-3  concept     transdutor da CEL-02 com offset +3.5 Nm
                                    → P(X) quase intacto, P(Y|X) MENTE
    dia  120–124  TC-3' recalibrado metrologia corrige → recuperação imediata
    dia  125–130  TC-4  data qual.  tag do PLC volta a publicar em kPa
                                    → violação de range de engenharia

A prova cruzada TC-2 × TC-3 é o experimento fundamental do tutorial:
duas situações que exigem respostas OPOSTAS e que um detector ingênuo
confunde.

CALIBRAÇÃO DO OFFSET DO TC-3 (por que 3.5 e não 10)
---------------------------------------------------
O rótulo passa por uma logística com coeficiente 0.62 sobre o desvio absoluto
de torque. Com offset de 10 Nm, o logito no segmento afetado vai a ~+2.4 e a
prevalência de falha satura em ~91%: a classe negativa praticamente
desaparece, o AUC do segmento deixa de ter significado estatístico e — pior —
não tem mais de onde CAIR. A prevalência global subia para ~31%, o que fazia
o monitor agregado despencar mais que o segmentado, invertendo a tese do
tutorial.

Varredura medida (seed=7):

    offset   prev_seg   prev_glob
      2.0      0.164      0.116
      3.0      0.250      0.138
      3.5      ~0.30      ~0.15     ← escolhido
      4.0      0.347      0.161
      6.0      0.578      0.219
     10.0      0.909      0.301     ← saturado

Com 3.5 Nm a degradação é inequívoca (3× a prevalência de referência) e ainda
sobram ~70% de negativos no segmento, o que dá um AUC estável. O efeito na
janela global fica modesto — que é exatamente o ponto pedagógico: a
segmentação não transforma o invisível em visível, ela transforma o
DESCARTÁVEL em ACIONÁVEL.

POR QUE `desvio_torque_abs` EXISTE
----------------------------------
O rótulo depende de `abs(torque_real - ALVO)`: a peça falha tanto por aperto
excessivo quanto por aperto insuficiente. Essa é uma função em V, e um termo
LINEAR sobre `torque_medido` é cego a ela — os dois lados do V se cancelam.
Sem a feature de desvio absoluto, uma regressão logística fica com AUC ~0.56
já no baseline, e toda "queda de performance" medida depois é ruído em cima
de um modelo que nunca aprendeu nada.

`desvio_torque_abs` é calculado sobre a LEITURA DO SENSOR, nunca sobre
`torque_real`:

    desvio_torque_abs = abs(torque_medido - TORQUE_ALVO)

Isso é engenharia de feature que qualquer engenheiro de processo faria — e
torna a lição do TC-3 mais forte, não mais fraca: o modelo é razoável, tem
a feature fisicamente correta, e MESMO ASSIM o sensor o engana. Bem mais
convincente que "modelo ruim continuou ruim".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "gerar_fixture",
    "janela",
    "gabarito",
    "validar_fixture",
    "cpk",
    "ppm_fora_spec",
    "NUM",
    "CAT",
    "FEATS",
    "RANGES",
    "EVENTOS",
    "OFFSET_TORQUE_PADRAO",
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

# Offset do transdutor no TC-3. Ver a nota de calibração no topo do módulo
# antes de mexer: este número está amarrado ao coeficiente 0.62 da logística
# do rótulo e aos limiares de vários testes.
OFFSET_TORQUE_PADRAO = 3.5

CELULAS = ("CEL-01", "CEL-02", "CEL-03", "CEL-04")
AROS = ("R16", "R17", "R18", "R20")
TURNOS = ("A", "B", "C")
FORNECEDORES = ("FORN-A", "FORN-B", "FORN-C")

NUM = (
    "torque_medido",
    "desvio_torque_abs",   # ver "POR QUE `desvio_torque_abs` EXISTE" no topo
    "pressao_psi",
    "temperatura_c",
    "tempo_ciclo_s",
    "angulo_giro_deg",
)
CAT = ("aro", "turno", "fornecedor", "equipamento")
FEATS = (*NUM, *CAT)

# Ranges de engenharia: camada 1 da pirâmide. Generosos de propósito —
# não são limites de controle estatístico, são limites do que é FISICAMENTE
# plausível o sensor reportar. Quem cruza isto tem problema de contrato de
# dados, não de modelo.
RANGES: dict[str, tuple[float, float]] = {
    "torque_medido": (90.0, 130.0),
    "desvio_torque_abs": (0.0, 20.0),   # derivado de torque_medido
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
    Evento("TC-3", 90, 119, "offset +3.5 Nm no transdutor da CEL-02", "concept", 3, "hardware",
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
    offset_torque: float = OFFSET_TORQUE_PADRAO,
    celula_afetada: str = "CEL-02",
) -> pd.DataFrame:
    """Gera a fixture completa, determinística dado `seed`.

    O mecanismo do TC-3, que é o coração do tutorial
    ------------------------------------------------
    A falha é aplicada ao valor *reportado*, não ao processo:

        torque_real   ~ N(110, 1.15)                 ← a física NÃO mudou
        torque_real  += offset · U(0.75, 1.00)       ← o operador COMPENSA
        torque_medido = torque_real - offset         ← o SENSOR mente

    O transdutor lê baixo, o operador acha que está frouxo e o aparafusador
    compensa apertando mais. O torque real sobe e sai da especificação
    superior, gerando trinca no cubo — mas a LEITURA parece normal.

    Por que isto é concept drift quase PURO
    ---------------------------------------
    A compensação do operador cancela quase todo o deslocamento aparente:

        Δ_medido = 0.875 · offset − offset = −0.125 · offset

    Com offset de 3.5 Nm são −0.44 Nm no segmento afetado. Como a CEL-02 é
    ~25% da janela, sobram ~−0.11 Nm na média global: 0.3% de um range de
    40 Nm. Invisível para PSI, KS ou qualquer teste sobre P(X) — e é
    exatamente por isso que a suíte precisa de um teste que PROVE a limitação
    em vez de assumi-la.

    A pista que SOBRA (e que é didática de propósito)
    -------------------------------------------------
    `angulo_giro_deg` é função do torque REAL, não da leitura. Ele vaza a
    compensação do operador: ~0.8 · 3.06 ≈ 2.4° num sigma de 18°, isto é
    ~0.14σ, concentrado na CEL-02. Pequeno, real, e invisível na agregada.

    Não é bug: é fisicamente correto e é a única pista que um detector
    SEGMENTADO consegue encontrar. Não espere silêncio total de P(X) no TC-3;
    espere um sussurro numa única feature, audível só quando se segmenta.

    Parameters
    ----------
    seed : int
        Um seed que passa não é um teste que passa. Veja
        `tests/test_detectores.py::test_controle_negativo_robusto_a_seed`,
        que roda o controle negativo em 8 seeds diferentes.
    offset_torque : float
        Erro do transdutor, em Nm. O default está calibrado para dar
        prevalência de falha ~30% no segmento afetado. Ver a nota
        "CALIBRAÇÃO DO OFFSET DO TC-3" no topo do módulo antes de alterar.
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

        # --- feature derivada: desvio absoluto da ESPECIFICAÇÃO --------------
        # Calculada sobre a LEITURA (torque_medido), nunca sobre torque_real.
        # É o que dá ao modelo linear acesso à forma em V do rótulo — e é
        # justamente ela que o sensor mentiroso corrompe no TC-3.
        desvio_torque_abs = np.abs(torque_medido - TORQUE_ALVO)

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
                    "desvio_torque_abs": desvio_torque_abs.astype(np.float32),
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


def validar_fixture(
    df: pd.DataFrame,
    *,
    offset_esperado: float = OFFSET_TORQUE_PADRAO,
) -> pd.DataFrame:
    """Testa a FIXTURE, não o detector.

    Fixture errada faz o detector parecer errado. Antes de acreditar em
    qualquer resultado, verificamos que plantamos o que dissemos que plantamos.

    Vários checks aqui têm TETO, não só piso. Um piso solitário é cúmplice:
    a versão anterior exigia apenas `f_tc3 > prev * 1.5`, e 91% de falha
    passava folgado — foi assim que a saturação do offset sobreviveu a uma
    suíte verde e só apareceu como seis testes quebrados lá na frente.
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
    add(
        f"TC-3: sensor lê ~{offset_esperado:.1f} Nm a menos",
        abs(gap - offset_esperado) < 0.5 + 0.1 * offset_esperado,
        f"{gap:.2f} Nm",
    )

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

    # --- os três checks que faltavam ------------------------------------------
    f_tc3 = tc3["falha"].mean()
    add(
        "TC-3 eleva a falha na CEL-02 SEM SATURAR",
        prev * 1.5 < f_tc3 < 0.45,
        f"{f_tc3:.1%} (esperado entre {prev * 1.5:.1%} e 45%)",
    )

    f_tc3_glob = df.query("90 <= dia <= 119")["falha"].mean()
    add(
        "TC-3 NÃO contamina a prevalência global",
        f_tc3_glob < prev * 1.7,
        f"{f_tc3_glob:.1%} vs {prev:.1%} na referência",
    )

    # A classe negativa tem que sobrar no segmento, senão o AUC não significa
    # nada — e um AUC sem significado não pode CAIR, que é o que o TC-3 precisa
    # demonstrar.
    neg_seg = (1.0 - f_tc3) * len(tc3)
    add(
        "TC-3 preserva classe negativa suficiente no segmento",
        neg_seg > 500 and f_tc3 < 0.45,
        f"{neg_seg:.0f} negativos ({1 - f_tc3:.0%} do segmento)",
    )

    # O deslocamento APARENTE precisa continuar pequeno: é a definição
    # operacional de "concept drift quase puro".
    delta_medido = tc3["torque_medido"].mean() - ref["torque_medido"].mean()
    add(
        "TC-3: P(X) do torque_medido quase intacto (<1 Nm)",
        abs(delta_medido) < 1.0,
        f"{delta_medido:+.2f} Nm",
    )

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
