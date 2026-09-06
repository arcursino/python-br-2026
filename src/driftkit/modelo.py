"""
driftkit.modelo — o modelo sob monitoramento, e as métricas que sobrevivem a 0.58%.

Caminho no repositório: src/driftkit/modelo.py

Por que este módulo existe separado
-----------------------------------
No tutorial, o modelo é DELIBERADAMENTE simples e chato: uma regressão
logística com pré-processamento padrão. Não é o objeto de estudo — é o
paciente. Trocá-lo por um gradient boosting não mudaria nenhuma conclusão do
tutorial, e é exatamente esse o ponto: drift é um problema de SISTEMA, não de
escolha de estimador.

O que aqui NÃO é chato: as métricas.

Com prevalência de 0.58% (Bosch), a maior parte do instrumental habitual
quebra de formas silenciosas:

    acurácia   um modelo que prevê "nunca falha" acerta 99.42%. Inútil.
    F1@0.5     depende violentamente do limiar; com 0.58% quase sempre dá 0.
    ROC-AUC    otimista e insensível: a região que importa (topo do ranking)
               é uma fração minúscula da curva.
    RMSE       não se aplica a classificação.

    → PR-AUC (average precision) como métrica de RANKING
    → MCC no melhor limiar como métrica de DECISÃO (foi a métrica oficial
      da competição Bosch)
    → log-loss por amostra como SINAL CONTÍNUO para detectores sequenciais

Nota de honestidade para dizer em voz alta na aula: a ementa prometia
"F1 e RMSE". Eles estão implementados aqui (`metricas_completas`) para você
comparar, e o notebook 02 mostra em 2 minutos por que enganam. Prometer e
depois justificar um upgrade é diferente de prometer e omitir.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

__all__ = [
    "construir_modelo",
    "treinar",
    "metricas",
    "metricas_completas",
    "melhor_limiar_mcc",
    "logloss_por_amostra",
    "Desempenho",
]


def _one_hot() -> OneHotEncoder:
    """OneHotEncoder compatível com sklearn <1.2 e >=1.2.

    `sparse` virou `sparse_output` na 1.2. O Colab troca de versão sem avisar;
    esta função é o preço de não fixar `==` no pyproject.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def construir_modelo(
    num: list[str] | tuple[str, ...],
    cat: list[str] | tuple[str, ...] = (),
    *,
    C: float = 1.0,
    class_weight: str | dict | None = "balanced",
    max_iter: int = 2000,
    seed: int = 0,
) -> Pipeline:
    """Pipeline sklearn: imputação → escala/one-hot → logística.

    `class_weight="balanced"` é obrigatório aqui. Sem ele, com 0.58% de
    prevalência, a logística converge para o classificador trivial e a AUC
    fica ilusoriamente estável — mascarando exatamente o drift que queremos
    medir.
    """
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median", add_indicator=True)),
                ("sc", StandardScaler()),
            ]), list(num)),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", _one_hot()),
            ]), list(cat)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(
            C=C, class_weight=class_weight, max_iter=max_iter,
            random_state=seed, n_jobs=None,
        )),
    ])


def treinar(
    referencia: pd.DataFrame,
    *,
    num: list[str] | tuple[str, ...],
    cat: list[str] | tuple[str, ...] = (),
    alvo: str = "falha",
    **kw,
) -> Pipeline:
    """Treina o modelo na janela de referência."""
    mod = construir_modelo(num, cat, **kw)
    X = referencia[[*num, *cat]]
    y = referencia[alvo].to_numpy()
    if len(np.unique(y)) < 2:
        raise ValueError(
            f"a referência tem só uma classe (n={len(y)}, positivos={int(y.sum())}). "
            "Amplie a janela de referência."
        )
    return mod.fit(X, y)


@dataclass(frozen=True, slots=True)
class Desempenho:
    """Métricas de uma janela. `None` quando não estimável — nunca 0 silencioso."""

    n: int
    n_pos: int
    pr_auc: float | None
    roc_auc: float | None
    mcc: float | None
    logloss: float | None
    limiar: float | None

    @property
    def estimavel(self) -> bool:
        return self.pr_auc is not None

    def to_dict(self) -> dict:
        return {
            "n": self.n, "n_pos": self.n_pos,
            "pr_auc": self.pr_auc, "roc_auc": self.roc_auc,
            "mcc": self.mcc, "logloss": self.logloss, "limiar": self.limiar,
        }


MIN_POSITIVOS = 8


def metricas(
    modelo: Pipeline,
    janela_df: pd.DataFrame,
    *,
    num: list[str] | tuple[str, ...],
    cat: list[str] | tuple[str, ...] = (),
    alvo: str = "falha",
    limiar: float | None = None,
) -> Desempenho:
    """Avalia o modelo numa janela, devolvendo `None` quando não dá para estimar.

    Este `None` é uma decisão de projeto importante. A alternativa comum —
    devolver `np.nan` e seguir — produz o pior dos mundos: `nan < 0.7` é
    `False`, então a janela não estimável passa como se estivesse saudável.
    Com `None`, quem consome é obrigado a tratar.
    """
    y = janela_df[alvo].to_numpy()
    n, n_pos = len(y), int(y.sum())
    if n_pos < MIN_POSITIVOS or n_pos == n:
        return Desempenho(n, n_pos, None, None, None, None, None)

    p = modelo.predict_proba(janela_df[[*num, *cat]])[:, 1]
    lim = limiar if limiar is not None else melhor_limiar_mcc(y, p)[0]
    return Desempenho(
        n=n,
        n_pos=n_pos,
        pr_auc=round(float(average_precision_score(y, p)), 5),
        roc_auc=round(float(roc_auc_score(y, p)), 5),
        mcc=round(float(matthews_corrcoef(y, (p >= lim).astype(int))), 5),
        logloss=round(float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7))), 5),
        limiar=round(float(lim), 5),
    )


def metricas_completas(
    modelo: Pipeline,
    janela_df: pd.DataFrame,
    *,
    num: list[str] | tuple[str, ...],
    cat: list[str] | tuple[str, ...] = (),
    alvo: str = "falha",
) -> dict[str, float | None]:
    """Inclui acurácia e F1@0.5 — as métricas que a ementa prometia.

    Estão aqui para serem CONFRONTADAS, não usadas. Rode isto numa janela do
    Bosch e mostre para a sala: acurácia 0.994 com o modelo em colapso.
    É a demonstração mais rápida de por que trocamos de métrica.
    """
    y = janela_df[alvo].to_numpy()
    if len(np.unique(y)) < 2:
        return {"acuracia": None, "f1_at_05": None, "pr_auc": None, "mcc_otimo": None}
    p = modelo.predict_proba(janela_df[[*num, *cat]])[:, 1]
    trivial = np.zeros_like(y)
    lim, mcc = melhor_limiar_mcc(y, p)
    return {
        "prevalencia": round(float(y.mean()), 5),
        "acuracia_do_modelo": round(float(((p >= 0.5).astype(int) == y).mean()), 5),
        "acuracia_do_burro": round(float((trivial == y).mean()), 5),  # ← o argumento
        "f1_at_05": round(float(f1_score(y, (p >= 0.5).astype(int), zero_division=0)), 5),
        "pr_auc": round(float(average_precision_score(y, p)), 5),
        "mcc_otimo": round(float(mcc), 5),
        "limiar_otimo": round(float(lim), 5),
    }


def melhor_limiar_mcc(
    y: np.ndarray, p: np.ndarray, *, n_grade: int = 200
) -> tuple[float, float]:
    """Varre o limiar maximizando MCC. Devolve (limiar, mcc).

    Isto é a resposta ao "drift do limiar" — o modo de drift que quase todo
    mundo esquece. Quando P(Y) muda, o limiar ótimo se desloca MESMO COM
    P(Y|X) intacto. O MCC cai, alguém pede retreino, e a correção certa
    custava uma linha de código: mover τ.

    Sempre reporte o par (limiar, métrica). Métrica de decisão sem o limiar
    que a produziu não é reproduzível.
    """
    qs = np.unique(np.quantile(p, np.linspace(0.50, 0.9995, n_grade)))
    melhor_l, melhor_m = 0.5, -1.0
    for t in qs:
        m = matthews_corrcoef(y, (p >= t).astype(int))
        if m > melhor_m:
            melhor_l, melhor_m = float(t), float(m)
    return melhor_l, melhor_m


def logloss_por_amostra(
    modelo: Pipeline,
    janela_df: pd.DataFrame,
    *,
    num: list[str] | tuple[str, ...],
    cat: list[str] | tuple[str, ...] = (),
    alvo: str = "falha",
) -> np.ndarray:
    """Log-loss individual — o sinal para ADWIN / Page-Hinkley.

    A lição do Bloco IV em uma frase: quando o erro binário é ~0.6% e dominado
    pelo prior, DDM/EDDM param de funcionar. A solução não é trocar o
    detector sequencial; é TROCAR O SINAL DE ENTRADA por algo contínuo e
    sensível. Mesmo ADWIN, sinal diferente, resultado diferente.
    """
    y = janela_df[alvo].to_numpy()
    p = np.clip(modelo.predict_proba(janela_df[[*num, *cat]])[:, 1], 1e-7, 1 - 1e-7)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))
