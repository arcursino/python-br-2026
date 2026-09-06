"""
driftkit.detectors — os instrumentos de medição.

Caminho no repositório: src/driftkit/detectors.py

Princípio de projeto deste módulo:

    Um detector de drift é um INSTRUMENTO DE MEDIÇÃO.

Instrumento de medição tem três propriedades que código de notebook normalmente
não tem, e que aqui são garantidas por construção:

  1. Referência congelada.   `DriftDetector` é `frozen=True`. A referência não
     muda depois da construção. Referência que se move sozinha é a causa nº 1
     de detector cego a deriva lenta (vide tests/test_limitacoes.py).
  2. Dependências explícitas. Nada de `global df`. Tudo entra pelo construtor.
  3. Resultado auditável.    `report()` devolve DataFrame com a estatística, o
     p-valor, o tamanho de efeito E a coluna `mensuravel` — porque
     "não deu para medir" é diferente de "não tem drift".
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp, wasserstein_distance

__all__ = [
    "psi",
    "js_cat",
    "wasserstein",
    "viola_range",
    "DriftDetector",
    "RelatorioDrift",
]

# -----------------------------------------------------------------------------
#  Constantes com nome. Números soltos no código são dívida técnica.
# -----------------------------------------------------------------------------
PSI_ATENCAO = 0.10  # o famoso "limiar da indústria" — que vamos DESMENTIR
PSI_ALARME = 0.25
BINS_PADRAO = 10
MIN_OBS_PADRAO = 200


# =============================================================================
#  Métricas de magnitude
# =============================================================================
def psi(
    ref: np.ndarray | pd.Series,
    cur: np.ndarray | pd.Series,
    *,
    bins: int = BINS_PADRAO,
    eps: float = 1e-6,
) -> float:
    """Population Stability Index entre duas amostras numéricas.

    PSI = Σ (p_cur - p_ref) · ln(p_cur / p_ref)

    Bins são quantis DA REFERÊNCIA — nunca dos dados atuais. Recalcular os bins
    a cada janela é o bug silencioso clássico: o detector passa a comparar a
    janela consigo mesma e o PSI tende a zero justamente quando há drift.

    Devolve np.nan quando não é mensurável (amostra pequena, referência
    degenerada). Quem consome DEVE tratar o nan — veja `DriftDetector.report`,
    que expõe a coluna `mensuravel` em vez de silenciar.

    >>> rng = np.random.default_rng(0)
    >>> round(psi(rng.normal(0, 1, 5000), rng.normal(0, 1, 5000)), 3) < 0.05
    True
    >>> psi(rng.normal(0, 1, 5000), rng.normal(2, 1, 5000)) > 0.5
    True
    """
    r = np.asarray(ref, dtype=float)
    c = np.asarray(cur, dtype=float)
    r = r[np.isfinite(r)]
    c = c[np.isfinite(c)]

    if r.size < 20 or c.size < 20:
        return np.nan

    bordas = np.unique(np.quantile(r, np.linspace(0, 1, bins + 1)))
    if bordas.size < 3:  # referência quase constante: PSI não é definível
        return np.nan
    bordas[0], bordas[-1] = -np.inf, np.inf

    p_ref = np.histogram(r, bins=bordas)[0] / r.size
    p_cur = np.histogram(c, bins=bordas)[0] / c.size
    p_ref = np.clip(p_ref, eps, None)
    p_cur = np.clip(p_cur, eps, None)

    return float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))


def js_cat(
    ref: pd.Series,
    cur: pd.Series,
    *,
    eps: float = 1e-12,
) -> float:
    """Divergência de Jensen-Shannon (base 2) entre duas distribuições categóricas.

    Resultado em [0, 1]: 0 = idênticas, 1 = suportes disjuntos.
    Preferida ao qui-quadrado como MAGNITUDE, porque é limitada e não cresce
    com o tamanho da amostra.

    Implementação vetorizada: `value_counts` + `reindex` em vez do
    `[(ref == k).mean() for k in categorias]`, que é O(n · k) e domina o tempo
    de execução quando há 160 features.
    """
    categorias = pd.Index(
        sorted(set(ref.dropna().unique()) | set(cur.dropna().unique()), key=str)
    )
    if categorias.empty:
        return np.nan

    p = ref.value_counts(normalize=True).reindex(categorias, fill_value=0.0).to_numpy()
    q = cur.value_counts(normalize=True).reindex(categorias, fill_value=0.0).to_numpy()
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / np.clip(b[mask], eps, None))))

    return float(np.clip(0.5 * _kl(p, m) + 0.5 * _kl(q, m), 0.0, 1.0))


def wasserstein(ref: np.ndarray | pd.Series, cur: np.ndarray | pd.Series) -> float:
    """Distância de Wasserstein-1, NA UNIDADE FÍSICA DA FEATURE.

    É a métrica para conversar com engenharia de processo: em vez de
    "o PSI deu 0.42", você diz "o torque deslocou 9.8 Nm". A segunda frase
    gera ordem de serviço; a primeira gera reunião.
    """
    r = np.asarray(ref, dtype=float)
    c = np.asarray(cur, dtype=float)
    r, c = r[np.isfinite(r)], c[np.isfinite(c)]
    if r.size < 20 or c.size < 20:
        return np.nan
    return float(wasserstein_distance(r, c))


def viola_range(
    cur: pd.DataFrame,
    ranges: dict[str, tuple[float, float]],
) -> dict[str, float]:
    """Camada 1 da pirâmide: fração de registros fora da especificação de engenharia.

    A mais barata das quatro camadas (custo O(n), sem referência, sem rótulo) e
    a única que pega troca de unidade — o caso em que retreinar grava o erro no
    modelo. Roda ANTES de qualquer teste estatístico.

    Devolve {feature: fração_fora}, só para features presentes em `cur`.
    """
    fora: dict[str, float] = {}
    for feat, (lo, hi) in ranges.items():
        if feat not in cur.columns:
            continue
        s = pd.to_numeric(cur[feat], errors="coerce").dropna()
        if s.empty:
            continue
        fora[feat] = float(((s < lo) | (s > hi)).mean())
    return fora

def piso_analitico(
    n: int,
    m: int,
    *,
    bins: int = 10,
    k_features: int = 1,
    quantil: float = 0.95,
) -> float:
    """Piso de ruído do PSI sob H0, em forma fechada.

    O PSI é a divergência de Jeffreys. Sua expansão de 2ª ordem sob a hipótese
    nula (duas amostras da MESMA distribuição) é um qui-quadrado escalado:

        PSI  ~  (1/n + 1/m) · χ²_{B-1}
        E[PSI]   = (B-1)·(1/n + 1/m)
        p95[PSI] = χ²_.95(B-1)·(1/n + 1/m)

    Para o MÁXIMO sobre k features aproximadamente independentes, a q-ésima
    quantil do máximo é a (q^{1/k})-ésima quantil da marginal — daí o
    `quantil ** (1/k)`. Com features correlacionadas isto é conservador
    (superestima o piso), o que é o lado certo para errar.

    Verificado empiricamente em 12 famílias de marginal (normal, lognormal,
    cauchy, pareto, bimodal, contagem, zero-inflada): razão medido/previsto
    com mediana 0.93 e 91% das células em [0.3, 3.0]. Distribuições
    degeneradas ficam ABAIXO da previsão, porque bins colapsam e os graus de
    liberdade efetivos caem. Ou seja: **a fórmula é um teto**.

    ⚠️  Vale para features numéricas (PSI). Features categóricas usam JS
    reescalada, que não tem esta forma fechada — exclua-as de `k_features`.

    Notas de uso
    ------------
    O que este número mata: o "PSI > 0.1" como regra universal. Resolvendo
    p95 = 0.1 com B=10 dá n ≈ 338. O 0.1 é o piso de ruído de uma janela de
    ~300 observações — que é exatamente o regime de scorecard de crédito, de
    onde a regra veio. A 40.000 observações por janela ele está 118× solto.
    """
    if min(n, m) < 20 or bins < 3:
        return np.nan
    q_eff = quantil ** (1.0 / max(int(k_features), 1))
    return float(chi2.ppf(q_eff, bins - 1) * (1.0 / n + 1.0 / m))


def n_equivalente(limiar: float, *, bins: int = 10, quantil: float = 0.95) -> float:
    """A que tamanho de janela `limiar` é puro ruído? (n = m)

    >>> round(n_equivalente(0.10))     # o folclore
    338
    """
    return float(chi2.ppf(quantil, bins - 1) * 2.0 / limiar)


# =============================================================================
#  Relatório
# =============================================================================
@dataclass(frozen=True, slots=True)
class RelatorioDrift:
    """Resultado de uma medição. Imutável — é um registro, não um rascunho."""

    tabela: pd.DataFrame
    n_ref: int
    n_cur: int
    alpha_corrigido: float
    psi_limiar: float

    # -- atalhos de leitura ---------------------------------------------------
    @property
    def em_drift(self) -> pd.DataFrame:
        return self.tabela.loc[self.tabela["drift"]].sort_values("efeito", ascending=False)

    @property
    def n_drift(self) -> int:
        return int(self.tabela["drift"].sum())

    @property
    def n_nao_mensuravel(self) -> int:
        return int((~self.tabela["mensuravel"]).sum())

    @property
    def efeito_max(self) -> float:
        vals = self.tabela.loc[self.tabela["mensuravel"], "efeito"]
        return float(vals.max()) if len(vals) else np.nan

    @property
    def top_features(self) -> list[str]:
        return self.em_drift["feature"].tolist()

    def resumo(self) -> str:
        linhas = [
            f"janela: n_ref={self.n_ref:,} n_cur={self.n_cur:,}",
            f"limiares: psi>{self.psi_limiar:.4f} E p<{self.alpha_corrigido:.2e} (Bonferroni)",
            f"features em drift: {self.n_drift}/{len(self.tabela)}"
            f"  |  efeito máximo: {self.efeito_max:.4f}",
        ]
        if self.n_nao_mensuravel:
            linhas.append(
                f"⚠️  {self.n_nao_mensuravel} feature(s) NÃO MENSURÁVEIS "
                "(amostra insuficiente) — ausência de alarme aqui não é ausência de drift"
            )
        if self.n_drift:
            top = self.em_drift.head(5)
            linhas.append("\n  top contribuintes:")
            for _, r in top.iterrows():
                linhas.append(f"    {r.feature:<24} efeito={r.efeito:.4f}  p={r.p_valor:.2e}")
        return "\n".join(linhas)

    def to_dict(self) -> dict:
        return {
            "n_ref": self.n_ref,
            "n_cur": self.n_cur,
            "n_drift": self.n_drift,
            "n_nao_mensuravel": self.n_nao_mensuravel,
            "efeito_max": None if np.isnan(self.efeito_max) else round(self.efeito_max, 6),
            "features": self.top_features,
        }


# =============================================================================
#  O detector
# =============================================================================
@dataclass(frozen=True)
class DriftDetector:
    """Detector de data drift com referência congelada.

    `frozen=True` não é preciosismo: é a garantia de que ninguém troca a
    referência no meio da execução. Para mudar um parâmetro, use `.com(...)`,
    que devolve um NOVO detector — o antigo continua auditável.

    Parameters
    ----------
    referencia : DataFrame
        A janela-base. Cópia defensiva é feita na construção.
    num, cat : sequência de nomes de coluna
        Features numéricas e categóricas a monitorar.
    psi_limiar : float
        Piso de MAGNITUDE. Padrão 0.10 é o folclore da indústria; o valor
        correto é medido por teste A/A na sua própria referência
        (veja `calibrar_piso`).
    alpha : float
        Nível do teste, ANTES da correção de Bonferroni por nº de features.
    min_obs : int
        Abaixo disso a janela não é mensurável — e isso é reportado, não
        silenciado.

    Examples
    --------
    >>> from driftkit.fixtures import gerar_fixture, janela
    >>> df = gerar_fixture(seed=7)
    >>> det = DriftDetector.from_reference(df.query("dia < 30"))
    >>> det.report(janela(df, 55)).n_drift          # regime estável
    0
    """

    referencia: pd.DataFrame
    num: tuple[str, ...] = ()
    cat: tuple[str, ...] = ()
    psi_limiar: float = PSI_ATENCAO
    alpha: float = 0.05
    min_obs: int = MIN_OBS_PADRAO
    bins: int = BINS_PADRAO
    ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    metrica: Literal["psi", "wasserstein"] = "psi"

    # -- construção -----------------------------------------------------------
    def __post_init__(self) -> None:
        object.__setattr__(self, "referencia", self.referencia.copy())
        if not self.num and not self.cat:
            raise ValueError(
                "detector sem features: informe `num` e/ou `cat`. "
                "Um detector que não monitora nada passa em todos os testes."
            )
        faltando = [c for c in (*self.num, *self.cat) if c not in self.referencia.columns]
        if faltando:
            raise KeyError(f"features ausentes na referência: {faltando}")

    @classmethod
    def from_reference(
        cls,
        referencia: pd.DataFrame,
        *,
        num: list[str] | None = None,
        cat: list[str] | None = None,
        **kw,
    ) -> DriftDetector:
        """Constrói inferindo num/cat pelos dtypes, se não informados."""
        if num is None:
            num = referencia.select_dtypes("number").columns.tolist()
        if cat is None:
            cat = referencia.select_dtypes(["object", "category", "bool"]).columns.tolist()
        return cls(referencia=referencia, num=tuple(num), cat=tuple(cat), **kw)

    @classmethod
    def from_contract(cls, contrato: dict, referencia: pd.DataFrame) -> DriftDetector:
        """Reconstrói o detector a partir de um detector_config.json.

        É isto que faz a ponte Parte I → Parte II: o MÉTODO atravessa,
        os PARÂMETROS são remedidos.
        """
        lim = contrato["limiares"]
        return cls(
            referencia=referencia,
            num=tuple(contrato["features"]["num"]),
            cat=tuple(contrato["features"]["cat"]),
            psi_limiar=lim["psi_alarme"],
            alpha=lim["alpha"],
            min_obs=lim["min_obs"],
            bins=lim.get("bins", BINS_PADRAO),
            ranges={k: tuple(v) for k, v in contrato.get("ranges", {}).items()},
        )

    def com(self, **mudancas) -> DriftDetector:
        """Devolve um novo detector com parâmetros alterados (o original é imutável)."""
        return replace(self, **mudancas)

    # -- medição --------------------------------------------------------------
    def report(self, atual: pd.DataFrame) -> RelatorioDrift:
        """Mede drift de P(X). NÃO decide nada — decisão é papel de `policy.py`.

        Predicado de alarme:

            drift = (p_valor < alpha / m)  E  (efeito > psi_limiar)

        O `E` (não `OU`) é o ponto pedagógico central: significância sem
        magnitude é fadiga de alerta; magnitude sem significância é ruído.
        Em 50 mil peças/dia, todo p-valor é zero — o `E` é o que salva o
        dashboard de ser ignorado.
        """
        m = max(1, len(self.num) + len(self.cat))
        alpha_c = self.alpha / m
        linhas: list[dict] = []

        mensuravel_global = len(atual) >= self.min_obs

        for c in self.num:
            r = pd.to_numeric(self.referencia[c], errors="coerce").dropna()
            a = pd.to_numeric(atual[c], errors="coerce").dropna() if c in atual else pd.Series(dtype=float)
            ok = mensuravel_global and len(r) >= 20 and len(a) >= 20
            if ok:
                ks = ks_2samp(r, a)
                stat, p = float(ks.statistic), float(ks.pvalue)
                efeito = psi(r, a, bins=self.bins) if self.metrica == "psi" else wasserstein(r, a)
            else:
                stat = p = efeito = np.nan
            linhas.append(
                dict(feature=c, tipo="num", stat=stat, p_valor=p, efeito=efeito,
                     mensuravel=bool(ok) and np.isfinite(efeito))
            )

        for c in self.cat:
            r = self.referencia[c].dropna().astype(str)
            a = atual[c].dropna().astype(str) if c in atual else pd.Series(dtype=str)
            ok = mensuravel_global and len(r) >= 20 and len(a) >= 20
            if ok:
                cats = sorted(set(r.unique()) | set(a.unique()))
                tab = np.vstack([
                    r.value_counts().reindex(cats, fill_value=0).to_numpy(),
                    a.value_counts().reindex(cats, fill_value=0).to_numpy(),
                ])
                tab = tab[:, tab.sum(axis=0) > 0]
                if tab.shape[1] >= 2:
                    stat, p = chi2_contingency(tab)[:2]
                    stat, p = float(stat), float(p)
                else:
                    stat = p = np.nan
                efeito = js_cat(r, a)
                # JS é limitada em [0,1]; PSI não. Escalamos para comparar na
                # mesma coluna `efeito` sem misturar unidades sem aviso.
                efeito = efeito * 2.5 if np.isfinite(efeito) else np.nan
            else:
                stat = p = efeito = np.nan
            linhas.append(
                dict(feature=c, tipo="cat", stat=stat, p_valor=p, efeito=efeito,
                     mensuravel=bool(ok) and np.isfinite(efeito))
            )

        tab = pd.DataFrame(linhas)
        tab["drift"] = (
            tab["mensuravel"]
            & (tab["p_valor"] < alpha_c)
            & (tab["efeito"] > self.psi_limiar)
        )
        return RelatorioDrift(
            tabela=tab,
            n_ref=len(self.referencia),
            n_cur=len(atual),
            alpha_corrigido=alpha_c,
            psi_limiar=self.psi_limiar,
        )

    def violacoes_de_range(self, atual: pd.DataFrame) -> dict[str, float]:
        """Camada 1 — atalho que usa os ranges configurados no detector."""
        return viola_range(atual, self.ranges)

    # -- calibração -----------------------------------------------------------
        def calibrar_piso(
        self,
        *,
        modo: Literal["bloco", "aleatorio", "ambos"] = "ambos",
        n_repeticoes: int = 40,
        n_blocos: int = 2,
        frac: float = 0.5,
        quantil: float = 0.95,
        seed: int = 0,
    ) -> dict[str, float | str]:
            """Calibra o limiar E diagnostica se a referência merece confiança.

            Faz DUAS coisas diferentes, e a distinção entre elas é o ponto:

            1. SPLIT ALEATÓRIO (`modo="aleatorio"`)
            Permuta a referência e divide ao meio, N vezes. Mede o piso de
            ruído do instrumento sob trocabilidade.

            ⚠️  Este é o teste A/A como todo mundo pratica — e ele é
            PROVADAMENTE CEGO a contaminação da janela de referência. Se a sua
            referência contém dois regimes, a permutação distribui os dois
            igualmente entre as metades: as duas metades passam a ser amostras
            da MESMA mistura, e mistura é permutacionalmente trocável. Medido:
            o valor fica travado em ~1× o previsto de 0% a 50% de contaminação.

            Serve para: confirmar que o instrumento está sadio.
            Não serve para: descobrir que o gabarito está podre.

            2. SPLIT EM BLOCO (`modo="bloco"`)
            Divide a referência em blocos CONSECUTIVOS, na ordem em que ela
            está, e compara blocos adjacentes. Se a referência estiver ordenada
            pelo eixo suspeito — tempo, lote, turno, versão de firmware,
            tenant — este split detecta heterogeneidade interna.

            Medido (contaminação por um 2º regime a +1σ, no fim da janela):
                0%  → H ≈ 0.6      10% → H ≈ 2.8
                5%  → H ≈ 0.8      20% → H ≈ 9.6      50% → H ≈ 55.8
            Falso positivo sob H0 verdadeiro: 0.0% em 12 famílias de marginal.

            O índice H
            ----------
                H = piso medido / piso analítico

                H < 2   referência homogênea no eixo testado
                2 ≤ H < 5   suspeita — investigue antes de confiar no limiar
                H ≥ 5   CONTAMINADA: há mais de um regime dentro da referência.
                        Nenhum limiar te salva disso. O problema não é o limiar,
                        é o gabarito.

            ⚠️  ORDENE A REFERÊNCIA antes de chamar com `modo="bloco"`. Se a ordem
            das linhas for arbitrária, o split em bloco não tem poder nenhum — ele
            vira um split aleatório caro.
            """
        rng = np.random.default_rng(seed)
        n_total = len(self.referencia)
        sonda = self.com(psi_limiar=0.0, alpha=1.0, min_obs=20)

        # --- quantas features realmente entram na conta? ---------------------
        # O piso analítico descreve o MÁXIMO sobre k features numéricas.
        # Categóricas usam JS reescalada e ficam fora da forma fechada.
        _rel0 = sonda.report(self.referencia)
        k_num = int(((_rel0.tabela.tipo == "num") & _rel0.tabela.mensuravel).sum())
        k_num = max(k_num, 1)

        def _efeito(a: pd.DataFrame, b: pd.DataFrame) -> float:
            if min(len(a), len(b)) < 20:
                return np.nan
            return sonda.com(referencia=a).report(b).efeito_max

        out: dict[str, float | str] = {
            "n_referencia": n_total,
            "k_features_num": k_num,
            "bins": self.bins,
        }

        # =====================================================================
        #  1. split aleatório — o piso do instrumento
        # =====================================================================
        piso_aa = np.nan
        if modo in ("aleatorio", "ambos"):
            corte = int(n_total * frac)
            efeitos = []
            for _ in range(n_repeticoes):
                idx = rng.permutation(n_total)
                e = _efeito(self.referencia.iloc[idx[:corte]],
                            self.referencia.iloc[idx[corte:]])
                if np.isfinite(e):
                    efeitos.append(e)
            arr = np.asarray(efeitos)
            if arr.size:
                piso_aa = float(np.quantile(arr, quantil))
                out |= {
                    "piso_aa": round(piso_aa, 6),
                    "aa_mediana": round(float(np.median(arr)), 6),
                    "aa_max": round(float(arr.max()), 6),
                    "n_repeticoes": int(arr.size),
                }

        # =====================================================================
        #  2. split em bloco — o diagnóstico da referência
        # =====================================================================
        piso_bloco = np.nan
        if modo in ("bloco", "ambos"):
            tam = n_total // max(n_blocos, 2)
            pares = []
            for i in range(max(n_blocos, 2) - 1):
                a = self.referencia.iloc[i * tam:(i + 1) * tam]
                b = self.referencia.iloc[(i + 1) * tam:(i + 2) * tam]
                e = _efeito(a, b)
                if np.isfinite(e):
                    pares.append((i, e))
            if pares:
                piso_bloco = max(e for _, e in pares)
                pior = max(pares, key=lambda t: t[1])[0]
                out |= {
                    "piso_bloco": round(piso_bloco, 6),
                    "n_blocos": max(n_blocos, 2),
                    "bloco_mais_divergente": f"{pior}→{pior + 1}",
                }

        # =====================================================================
        #  3. o null, e as razões contra ele
        # =====================================================================
        n_meia = n_total // 2
        prev_aa = piso_analitico(n_meia, n_total - n_meia,
                                 bins=self.bins, k_features=k_num, quantil=quantil)
        tam_b = n_total // max(n_blocos, 2)
        prev_bl = piso_analitico(tam_b, tam_b,
                                 bins=self.bins, k_features=k_num, quantil=quantil)
        out["piso_analitico"] = round(prev_aa, 6)

        if np.isfinite(piso_aa) and prev_aa > 0:
            out["razao_aa"] = round(piso_aa / prev_aa, 2)

        H = np.nan
        if np.isfinite(piso_bloco) and prev_bl > 0:
            H = piso_bloco / prev_bl
            out["H"] = round(H, 2)
            out["referencia"] = ("homogenea" if H < 2 else
                                 "suspeita" if H < 5 else "CONTAMINADA")

        # =====================================================================
        #  4. o limiar sugerido
        # =====================================================================
        # Sem constante absoluta. O antigo `max(3*piso, 0.02)` impunha um piso
        # de 0.02 que, a n=40.000, é ~25× o ruído real — cegava o detector
        # exatamente onde havia mais poder estatístico disponível.
        #
        # O analítico entra como piso do piso: protege contra medição
        # subestimada (poucas repetições, marginal degenerada).
        base = np.nanmax([piso_aa, prev_aa])
        if np.isfinite(base):
            out["psi_alarme_sugerido"] = round(3.0 * base, 6)

        if np.isfinite(H) and H >= 5:
            out["aviso"] = (
                f"H={H:.1f}: a referência contém mais de um regime. O limiar "
                "sugerido acima é aritmeticamente correto e praticamente "
                "inútil — encurte ou segmente a referência e recalibre."
            )

        return out
