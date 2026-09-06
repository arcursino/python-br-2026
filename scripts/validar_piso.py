#!/usr/bin/env python
"""
validar_piso.py — decide, em ~2 horas de execução, se o paper existe.

Caminho sugerido: scripts/validar_piso.py

O QUE ESTE SCRIPT TESTA
=======================
Três alegações, em ordem crescente de originalidade:

  A1  O piso de ruído do PSI sob H0 tem forma fechada:
          E[PSI] ≈ (B-1)·(1/n + 1/m)      p95 ≈ χ²_.95(B-1)·(1/n + 1/m)
      (PSI é a divergência de Jeffreys; sua expansão de 2ª ordem é um
      χ²_{B-1} escalado.)
      → Se A1 vale, "PSI > 0.1" não é um limiar de drift: é o piso de
        ruído de n≈200–340 com B=10.

  A2  O teste A/A com SPLIT ALEATÓRIO é cego a contaminação da janela de
      referência. Por construção: o split aleatório distribui a
      contaminação igualmente entre as metades, e uma mistura é
      permutacionalmente trocável.
      → Se A2 vale, o conselho padrão "rode um A/A na sua referência"
        está errado, e este script mostra o que fazer no lugar.

  A3  O split em BLOCO ao longo do eixo suspeito (tempo, lote, turno,
      tenant), comparado ao null analítico, DETECTA contaminação — com
      poder útil a partir de ~10% de mistura e taxa de falso positivo
      próxima de zero sob H0 verdadeiro.
      → A3 é a contribuição. Índice H = piso_medido / piso_analítico.

NOTA DE HONESTIDADE
===================
A1 provavelmente NÃO é nova. Antes de reivindicar qualquer coisa, procure:
  · Yurdakul, B. (2018) "Statistical Properties of Population Stability
    Index" — dissertação, Western Michigan University
  · Yurdakul & Naranjo, "Statistical properties of the population
    stability index" (Journal of Risk Model Validation)
Se eles derivam a distribuição assintótica, A1 é replicação — e a sua
contribuição passa a ser A2+A3 e a aplicação a monitoramento de ML.
Isso ainda é artigo. Mas o posicionamento muda.

USO
---
    python validar_piso.py --rapido           # ~3 min, só sintético
    python validar_piso.py                    # ~40 min, completo
    python validar_piso.py --dados meus.parquet --eixo data_referencia
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
#  Núcleo
# =============================================================================
def psi(ref, cur, bins: int = 10, eps: float = 1e-6) -> float:
    """PSI com bins por quantil DA REFERÊNCIA. Devolve nan se não mensurável."""
    r = np.asarray(ref, float); c = np.asarray(cur, float)
    r = r[np.isfinite(r)]; c = c[np.isfinite(c)]
    if r.size < 20 or c.size < 20:
        return np.nan
    b = np.unique(np.quantile(r, np.linspace(0, 1, bins + 1)))
    if b.size < 3:
        return np.nan
    b[0], b[-1] = -np.inf, np.inf
    p = np.clip(np.histogram(r, bins=b)[0] / r.size, eps, None)
    q = np.clip(np.histogram(c, bins=b)[0] / c.size, eps, None)
    return float(np.sum((q - p) * np.log(q / p)))


def piso_analitico(n: int, m: int, B: int = 10, q: float = 0.95) -> float:
    """O null em forma fechada. É contra ISTO que tudo é comparado."""
    return float(chi2.ppf(q, B - 1) * (1.0 / n + 1.0 / m))


def n_equivalente(limiar: float, B: int = 10, q: float = 0.95) -> float:
    """Qual tamanho de amostra faz `limiar` ser puro ruído? (n = m)"""
    return float(chi2.ppf(q, B - 1) * 2.0 / limiar)


def piso_aa_aleatorio(x, n: int, B: int = 10, reps: int = 150, rng=None) -> float:
    """A/A clássico: split ALEATÓRIO. (A2 diz que isto é cego a contaminação.)"""
    rng = rng or np.random.default_rng(0)
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size < 2 * n:
        return np.nan
    o = [psi(x[i[:n]], x[i[n:2 * n]], B) for i in
         (rng.permutation(x.size) for _ in range(reps))]
    return float(np.nanquantile(o, 0.95))


def indice_H(x, B: int = 10, modo: str = "bloco", reps: int = 150, rng=None) -> float:
    """H = piso medido / piso analítico.

    modo="bloco"    : 1ª metade vs 2ª metade NA ORDEM DADA (tempo/lote)
    modo="aleatorio": split permutado (controle negativo)

    H ≈ 1   → a janela é internamente homogênea no eixo testado
    H > 2   → suspeita de contaminação
    H > 5   → a janela contém mais de um regime. Seu gabarito está podre.
    """
    x = np.asarray(x, float)
    m = np.isfinite(x)
    if m.sum() < 200:
        return np.nan
    x = x[m]
    h = x.size // 2
    prev = piso_analitico(h, x.size - h, B)
    if modo == "bloco":
        v = psi(x[:h], x[h:], B)
    else:
        rng = rng or np.random.default_rng(0)
        o = [psi(x[i[:h]], x[i[h:]], B) for i in
             (rng.permutation(x.size) for _ in range(reps))]
        v = np.nanquantile(o, 0.95)
    return float(v / prev) if np.isfinite(v) and prev > 0 else np.nan


def H_dataframe(df: pd.DataFrame, colunas=None, B: int = 10,
                modo: str = "bloco") -> pd.DataFrame:
    """Aplica o índice H a cada coluna. Devolve tabela ordenada por H."""
    colunas = colunas or df.select_dtypes("number").columns.tolist()
    linhas = []
    for c in colunas:
        h = indice_H(df[c].to_numpy(), B=B, modo=modo)
        if np.isfinite(h):
            linhas.append({"feature": c, "H": round(h, 2),
                            "veredicto": ("homogenea" if h < 2 else
                                          "suspeita" if h < 5 else "CONTAMINADA")})
    return pd.DataFrame(linhas).sort_values("H", ascending=False)


# =============================================================================
#  Famílias de distribuição — cobrem as patologias de dado tabular real
# =============================================================================
def familias(rng, N: int = 200_000) -> dict:
    return {
        "normal":              rng.normal(0, 1, N),
        "lognormal":           rng.lognormal(0, 1, N),
        "cauchy":              rng.standard_cauchy(N),
        "exponencial":         rng.exponential(1, N),
        "bimodal":             np.r_[rng.normal(-3, 1, N // 2), rng.normal(3, 1, N // 2)],
        "uniforme":            rng.uniform(0, 1, N),
        "pareto_1.5":          rng.pareto(1.5, N) + 1,
        "arredondado_int":     np.round(rng.normal(0, 1, N)),
        "poisson_2":           rng.poisson(2, N).astype(float),
        "zero_inflado_70":     np.where(rng.random(N) < .7, 0., rng.normal(0, 1, N)),
        "zero_inflado_90":     np.where(rng.random(N) < .9, 0., rng.normal(0, 1, N)),
        "misto_pico_cauda":    np.where(rng.random(N) < .5, 0., rng.lognormal(0, 2, N)),
    }


# =============================================================================
#  E1 — o null analítico se sustenta?
# =============================================================================
def E1(rng, rapido: bool) -> pd.DataFrame:
    print("=" * 78)
    print("E1 · O PISO A/A MEDIDO SEGUE A PREDIÇÃO ANALÍTICA?")
    print("=" * 78)
    print("     previsto: p95 = χ²_.95(B-1)·(2/n)\n")
    fam = familias(rng, 60_000 if rapido else 200_000)
    ns = [1000, 8000] if rapido else [500, 1000, 4000, 8000, 40000]
    Bs = [10] if rapido else [5, 10, 20]
    reps = 60 if rapido else 150

    reg = []
    print(f"{'família':<20}{'n':>7}{'B':>4}{'previsto':>11}{'medido':>11}{'razão':>8}")
    print("-" * 78)
    for nome, x in fam.items():
        for n in ns:
            for B in Bs:
                if x.size < 2 * n:
                    continue
                prev = piso_analitico(n, n, B)
                med = piso_aa_aleatorio(x, n, B, reps, rng)
                if not np.isfinite(med) or med <= 0:
                    continue
                reg.append(dict(familia=nome, n=n, B=B, previsto=prev,
                                medido=med, razao=med / prev))
                if B == 10:
                    print(f"{nome:<20}{n:>7}{B:>4}{prev:>11.5f}{med:>11.5f}"
                          f"{med / prev:>8.2f}")
    R = pd.DataFrame(reg)
    r = R.razao.values
    print("-" * 78)
    print(f"razão mediana: {np.median(r):.2f}  ·  80% central: "
          f"[{np.quantile(r, .1):.2f}, {np.quantile(r, .9):.2f}]")
    print(f"fração em [0.3, 3.0]: {((r > .3) & (r < 3)).mean():.0%}")
    print(f"fração ACIMA de 3.0 : {(r > 3).mean():.1%}   ← se >5%, A1 está frágil")
    print("\nInterpretação: a predição é um TETO conservador. Distribuições")
    print("degeneradas (empates, zeros inflados) ficam ABAIXO, porque bins")
    print("colapsam e os graus de liberdade efetivos caem.")
    return R


# =============================================================================
#  E2 — a inversão do 0.1
# =============================================================================
def E2() -> pd.DataFrame:
    print("\n" + "=" * 78)
    print("E2 · A QUE TAMANHO DE AMOSTRA O LIMIAR 0.1 CORRESPONDE?")
    print("=" * 78)
    reg = []
    print(f"{'B':>5}{'n p/ E[PSI]=0.1':>18}{'n p/ p95=0.1':>15}"
          f"{'piso real @40k':>17}{'0.1 é X vezes':>15}")
    print("-" * 78)
    for B in [5, 10, 20, 50]:
        n_med = (B - 1) * 2 / 0.1
        n_p95 = n_equivalente(0.1, B)
        piso40k = piso_analitico(40000, 40000, B)
        reg.append(dict(B=B, n_media=n_med, n_p95=n_p95, piso_40k=piso40k,
                        folga=0.1 / piso40k))
        print(f"{B:>5}{n_med:>18.0f}{n_p95:>15.0f}{piso40k:>17.5f}"
              f"{0.1 / piso40k:>15.0f}×")
    print("-" * 78)
    print("Com B=10 (o padrão universal), 0.1 é o piso de ruído de n≈180–340.")
    print("É exatamente o regime de scorecard de crédito, de onde a regra veio.")
    print("\n>>> COROLÁRIO: um monitor com limiar FIXO fica progressivamente")
    print(">>> mais CEGO conforme o seu volume de dados cresce. Você paga por")
    print(">>> mais dados e joga fora todo o poder estatístico que eles compram.")
    return pd.DataFrame(reg)


# =============================================================================
#  E3 — a curva de poder: aleatório vs bloco
# =============================================================================
def E3(rng, rapido: bool) -> pd.DataFrame:
    print("\n" + "=" * 78)
    print("E3 · PODER DIAGNÓSTICO — SPLIT ALEATÓRIO vs SPLIT EM BLOCO")
    print("=" * 78)
    print("Referência de 4000 obs contaminada por um 2º regime no FIM da janela")
    print("(como na vida real: a mudança começou e ninguém percebeu).\n")
    n, B = 4000, 10
    n_feats = 8 if rapido else 16
    reps = 40 if rapido else 100
    deslocs = [1.0] if rapido else [0.5, 1.0, 2.0]

    reg = []
    for d in deslocs:
        print(f"deslocamento do 2º regime = {d:.1f} σ")
        print(f"{'contaminação':>14}{'H aleatório':>14}{'H bloco':>11}{'veredicto':>16}")
        print("-" * 60)
        for p in [0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50]:
            Ha, Hb = [], []
            for _ in range(n_feats):
                k = int(n * p)
                x = np.r_[rng.normal(0, 1, n - k), rng.normal(d, 1, k)]
                Ha.append(indice_H(x, B, "aleatorio", reps, rng))
                Hb.append(indice_H(x, B, "bloco"))
            ha, hb = np.nanmedian(Ha), np.nanmedian(Hb)
            ver = "homogenea" if hb < 2 else ("suspeita" if hb < 5 else "CONTAMINADA")
            reg.append(dict(desloc=d, contaminacao=p, H_aleatorio=ha, H_bloco=hb))
            print(f"{p:>13.0%}{ha:>14.1f}{hb:>11.1f}{ver:>16}")
        print()
    print("-" * 78)
    print(">>> O split ALEATÓRIO é constante em ~1 para toda contaminação: ele é")
    print(">>> CEGO, por construção. A permutação distribui a contaminação")
    print(">>> igualmente entre as metades, e uma mistura é trocável.")
    print(">>>")
    print(">>> O split em BLOCO cresce monotonicamente. É o teste que funciona.")
    return pd.DataFrame(reg)


# =============================================================================
#  E4 — tamanho do teste (falso positivo sob H0 verdadeiro)
# =============================================================================
def E4(rng, rapido: bool) -> pd.DataFrame:
    print("\n" + "=" * 78)
    print("E4 · CONTROLE — TAXA DE FALSO POSITIVO DO SPLIT EM BLOCO SOB H0")
    print("=" * 78)
    print("Dado genuinamente i.i.d.: o teste NÃO deve acusar contaminação.\n")
    reps = 150 if rapido else 500
    reg = []
    print(f"{'família':<22}{'H mediano':>11}{'P(H>2)':>9}{'P(H>3)':>9}{'P(H>5)':>9}")
    print("-" * 78)
    for nome, x in familias(rng, 30_000).items():
        vals = []
        for _ in range(reps):
            idx = rng.choice(x.size, 4000, replace=False)
            vals.append(indice_H(x[idx], 10, "bloco"))
        v = np.asarray(vals); v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        reg.append(dict(familia=nome, H_mediano=np.median(v),
                        fp2=np.mean(v > 2), fp3=np.mean(v > 3), fp5=np.mean(v > 5)))
        print(f"{nome:<22}{np.median(v):>11.2f}{np.mean(v > 2):>9.1%}"
              f"{np.mean(v > 3):>9.1%}{np.mean(v > 5):>9.1%}")
    R = pd.DataFrame(reg)
    print("-" * 78)
    print(f"P(H>2) máxima entre famílias: {R.fp2.max():.1%}"
          "   ← se >10%, o corte H>2 é frouxo demais")
    return R


# =============================================================================
#  E5 — dados reais do usuário
# =============================================================================
def E5(caminho: Path, eixo: str | None, B: int = 10) -> pd.DataFrame | None:
    print("\n" + "=" * 78)
    print("E5 · SEUS DADOS")
    print("=" * 78)
    df = (pd.read_parquet(caminho) if caminho.suffix in {".parquet", ".pq"}
          else pd.read_csv(caminho))
    print(f"{caminho.name}: {len(df):,} linhas × {df.shape[1]} colunas")

    if eixo and eixo in df.columns:
        df = df.sort_values(eixo, kind="stable")
        print(f"ordenado por `{eixo}` — o split em bloco respeita o eixo temporal")
    else:
        print("⚠️  SEM eixo temporal informado (--eixo). O split em bloco usa a")
        print("    ordem das linhas, que pode não significar nada. Se a ordem for")
        print("    arbitrária, E5 não tem poder — veja A2.")

    num = df.select_dtypes("number").columns.tolist()
    num = [c for c in num if c != eixo and df[c].notna().sum() > 500]
    if not num:
        print("nenhuma coluna numérica com dados suficientes.")
        return None

    tb_bloco = H_dataframe(df, num[:60], B, "bloco")
    tb_aleat = H_dataframe(df, num[:60], B, "aleatorio")
    tb = tb_bloco.merge(tb_aleat[["feature", "H"]], on="feature",
                        suffixes=("_bloco", "_aleatorio"))

    print(f"\n{'feature':<34}{'H bloco':>10}{'H aleat.':>10}{'veredicto':>16}")
    print("-" * 78)
    for _, r in tb.head(20).iterrows():
        print(f"{str(r.feature)[:33]:<34}{r.H_bloco:>10.2f}"
              f"{r.H_aleatorio:>10.2f}{r.veredicto:>16}")

    n_cont = (tb.H_bloco > 5).sum()
    print("-" * 78)
    print(f"features com H>5 (contaminadas): {n_cont}/{len(tb)}")
    print(f"H mediano no bloco: {tb.H_bloco.median():.2f}  ·  "
          f"no aleatório: {tb.H_aleatorio.median():.2f}")
    if n_cont > len(tb) * 0.2:
        print("\n⚠️  Mais de 20% das features acusam contaminação. Sua janela de")
        print("    referência provavelmente contém mudança de regime. NENHUM")
        print("    limiar te salva disso — o gabarito está podre. Encurte a")
        print("    referência ou segmente antes de calibrar.")

    piso = piso_analitico(len(df) // 2, len(df) // 2, B)
    print(f"\npiso analítico para n={len(df)//2:,}: {piso:.6f}")
    print(f"o limiar 0.1 é {0.1/piso:.0f}× esse piso  →  "
          f"corresponderia a n≈{n_equivalente(0.1, B):.0f}")
    return tb


# =============================================================================
#  Veredicto
# =============================================================================
def veredicto(r1, r2, r3, r4) -> dict:
    print("\n" + "█" * 78)
    print("VEREDICTO — O PAPER EXISTE?")
    print("█" * 78)

    razoes = r1.razao.values
    a1 = bool(np.median(razoes) > 0.4 and np.median(razoes) < 2.5
              and (razoes > 3).mean() < 0.05)

    aleat = r3.groupby("contaminacao").H_aleatorio.median()
    a2 = bool(aleat.max() / max(aleat.min(), 1e-9) < 2.0)

    bloco = r3.groupby("contaminacao").H_bloco.median()
    monot = bool(bloco.loc[0.5] > bloco.loc[0.0] * 5)
    poder = bool(bloco.loc[0.20] > 2)
    fp_ok = bool(r4.fp2.max() < 0.15)
    a3 = monot and poder and fp_ok

    for cod, ok, desc in [
        ("A1", a1, "o null analítico se sustenta em dados heterogêneos"),
        ("A2", a2, "o A/A com split aleatório é cego a contaminação"),
        ("A3", a3, "o split em bloco vs. o null detecta contaminação"),
    ]:
        print(f"  {'✅' if ok else '❌'}  {cod}  {desc}")

    print("\n  evidência:")
    print(f"     A1: razão mediana {np.median(razoes):.2f}, "
          f"{(razoes > 3).mean():.1%} acima de 3×  (n={len(razoes)} células)")
    print(f"     A2: H aleatório varia {aleat.min():.2f}→{aleat.max():.2f} "
          f"em 0–50% de contaminação")
    print(f"     A3: H bloco {bloco.loc[0.0]:.2f} (limpo) → "
          f"{bloco.loc[0.20]:.2f} (20%) → {bloco.loc[0.5]:.2f} (50%)")
    print(f"         falso positivo máx sob H0: {r4.fp2.max():.1%}")

    print("\n" + "-" * 78)
    if a1 and a2 and a3:
        print("  🟢 SIM — escreva o paper.")
        print("""
  Tese: o piso de ruído do PSI tem forma fechada; o limiar 0.1 corresponde
  a n≈200–340 e portanto fica ~100× solto em janelas modernas; e o teste
  A/A como praticado (split aleatório) é PROVADAMENTE cego a contaminação
  da referência — o split tem que ser ao longo do eixo suspeito.

  O terceiro ponto é o mais forte porque CORRIGE uma prática difundida,
  e correção de prática é aceita mais facilmente que método novo.

  ANTES DE ESCREVER — busca bibliográfica obrigatória:
    · Yurdakul (2018), propriedades estatísticas do PSI
    · Yurdakul & Naranjo, J. Risk Model Validation
    · "reference window selection" + "concept drift"
    · o que NannyML publicou sobre distinguir covariate de concept shift
  Se A1 já existe, reposicione: a contribuição é A2+A3.

  Alvos: KDMiLe ou BRACIS (nacionais, sérios, aceitam trabalho aplicado);
  workshop de MLOps em NeurIPS/ICML. NÃO journal de primeira viagem.""")
    elif a1 and (a2 or a3):
        print("  🟡 PARCIAL — artigo técnico longo sim, paper ainda não.")
        print("     Publique em português com o vídeo da palestra. Reveja os")
        print("     experimentos que falharam antes de tentar peer review.")
    else:
        print("  🔴 NÃO com esta evidência.")
        print("     Publique o artigo em português como material didático —")
        print("     que é valioso e tem lacuna real em pt-BR — e não reivindique")
        print("     contribuição metodológica.")
    print("-" * 78)
    return {"A1": a1, "A2": a2, "A3": a3,
            "razao_mediana": float(np.median(razoes)),
            "H_bloco_20pct": float(bloco.loc[0.20]),
            "fp_max": float(r4.fp2.max())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rapido", action="store_true", help="~3 min em vez de ~40")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dados", type=Path, default=None, help="parquet/csv seu (E5)")
    ap.add_argument("--eixo", type=str, default=None,
                    help="coluna temporal/de lote para o split em bloco")
    ap.add_argument("--saida", type=Path, default=Path("resultados_piso"))
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    a.saida.mkdir(parents=True, exist_ok=True)

    print(f"\nvalidar_piso.py · seed={a.seed} · modo="
          f"{'rápido' if a.rapido else 'completo'}\n")

    r1 = E1(rng, a.rapido)
    r2 = E2()
    r3 = E3(rng, a.rapido)
    r4 = E4(rng, a.rapido)
    r5 = E5(a.dados, a.eixo) if a.dados else None

    v = veredicto(r1, r2, r3, r4)

    for nome, tb in [("E1_null", r1), ("E2_inversao", r2),
                     ("E3_poder", r3), ("E4_falso_positivo", r4)]:
        tb.to_csv(a.saida / f"{nome}.csv", index=False)
    if r5 is not None:
        r5.to_csv(a.saida / "E5_seus_dados.csv", index=False)
    (a.saida / "veredicto.json").write_text(
        json.dumps(v, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nresultados em {a.saida}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
