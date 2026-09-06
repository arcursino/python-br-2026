#!/usr/bin/env python
"""
scripts/preprocess_bosch.py — pré-processamento do Bosch Production Line Performance.

Caminho no repositório: scripts/preprocess_bosch.py

VOCÊ (instrutor) roda isto UMA VEZ, em casa, e sobe os dois .parquet resultantes
no GitHub Releases. Os participantes NUNCA rodam isto em sala.

    14.3 GB de CSV  →  ~180 MB de parquet  →  ~25 min de aula economizados

Mas o código FICA no notebook, projetado e explicado, porque o padrão que ele
demonstra — "reduza por chunks, não carregue por chunks" — é a resposta certa
em entrevista de engenharia de dados e vale mais que o resultado.

Uso
---
    python scripts/preprocess_bosch.py --entrada ~/kaggle/bosch --saida data/
    python scripts/preprocess_bosch.py --entrada ~/kaggle/bosch --saida data/ --amostra 50000

Dados de origem (aceite as regras da competição antes):
    kaggle competitions download -c bosch-production-line-performance

    train_numeric.csv       2.1 GB   1.183.747 × 970   (Id, 968 features, Response)
    train_date.csv          2.9 GB   1.183.747 × 1157  (Id, 1156 timestamps)
    train_categorical.csv   2.7 GB   1.183.747 × 2141

Saída
-----
    meta_bosch.parquet          ~48 MB   1 linha por peça: eixo temporal + presença
    bosch_num_160feats.parquet  ~132 MB  Id + 160 features float32 + Response
"""

from __future__ import annotations

import argparse
import gc
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
#  Constantes
# =============================================================================
PADRAO_COLUNA = re.compile(r"^L(?P<linha>\d+)_S(?P<estacao>\d+)_[FD](?P<num>\d+)$")

CHUNK = 100_000
N_FEATURES_ALVO = 160
COBERTURA_MINIMA = 0.05   # feature presente em <5% das peças é ruído de rota
COMPRESSAO = "zstd"       # ~25% menor que snappy, leitura igualmente rápida

# O tempo no Bosch é dado em unidades de 0.01 semana (~100.8 min).
# Isto não está documentado na competição; foi inferido pela comunidade
# observando que t_max ≈ 1718 ≈ 17.18 semanas ≈ 4 meses de produção.
UNIDADE_TEMPO_SEMANAS = 0.01


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# =============================================================================
#  1. Topologia da fábrica — a convenção de nomes É o mapa
# =============================================================================
def parsear_topologia(colunas: list[str]) -> pd.DataFrame:
    """L3_S36_F3939 → linha 3, estação 36, feature 3939.

    Nenhum outro dataset público te dá a topologia de uma fábrica de graça.
    Aproveite: é ela que permite monitorar POR ESTAÇÃO em vez de por feature
    anônima — e monitorar por estação é o que transforma "a feature 3939
    driftou" em "a estação 36 driftou", que é uma frase acionável.
    """
    registros = []
    for c in colunas:
        if m := PADRAO_COLUNA.match(c):
            registros.append({
                "col": c,
                "linha": int(m["linha"]),
                "estacao": int(m["estacao"]),
                "num": int(m["num"]),
                "station": f"L{m['linha']}_S{m['estacao']}",
            })
    return pd.DataFrame(registros)


# =============================================================================
#  2. Eixo temporal + presença por estação  (a redução por chunks)
# =============================================================================
def construir_meta(
    f_date: Path,
    f_num: Path,
    *,
    chunk: int = CHUNK,
    limite_linhas: int | None = None,
) -> pd.DataFrame:
    """Varre train_date.csv em chunks e devolve UMA LINHA POR PEÇA.

    O padrão essencial: para cada chunk de 100k linhas × 1156 colunas
    (≈ 460 MB em float32), guardamos apenas ~130 números por peça e
    DESCARTAMOS o chunk. Pico de memória ~600 MB em vez de 5 GB.

    Colunas produzidas:
        Id, Response          identificação e rótulo
        t_min, t_max          primeiro e último timestamp da peça
        duracao               t_max - t_min (tempo de travessia)
        n_estacoes            quantas estações a peça visitou
        semana                t_min convertido em semana inteira → EIXO TEMPORAL
        vis_L?_S??            0/1 por estação → base do PRESENCE DRIFT

    ⚠️  Sobre o `Id`: lemos como int64 explicitamente. É tentador usar
    float32 no arquivo inteiro por economia, mas float32 tem 24 bits de
    mantissa (16.777.216) e os Ids do Bosch chegam a ~2.4 milhões. Passa —
    raspando. Num dataset 10× maior, o `Id` começaria a colidir SILENCIOSAMENTE.
    Não faça economia de tipo em chave primária.
    """
    log("lendo cabeçalho de train_date.csv...")
    cols_date = pd.read_csv(f_date, nrows=0).columns.tolist()
    date_cols = [c for c in cols_date if c != "Id"]
    topo = parsear_topologia(date_cols)
    log(f"  {len(date_cols)} colunas de timestamp em {topo.station.nunique()} estações")

    # --- índices precomputados: O(1) por chunk em vez de O(estações × colunas)
    # A versão ingênua ([c for c in ch.columns if MAPA[c] == est] dentro do
    # loop) faz 128 × 1156 comparações de string POR CHUNK. Isto é 8-15× mais
    # rápido, e a diferença aparece como 10 minutos de aula.
    idx_por_estacao: dict[str, list[int]] = defaultdict(list)
    mapa = dict(zip(topo["col"], topo["station"]))
    for i, c in enumerate(date_cols):
        if (est := mapa.get(c)) is not None:
            idx_por_estacao[est].append(i)
    estacoes = sorted(idx_por_estacao)
    indices = {e: np.asarray(idx_por_estacao[e]) for e in estacoes}

    partes: list[pd.DataFrame] = []
    total = 0
    t0 = time.perf_counter()

    leitor = pd.read_csv(
        f_date,
        chunksize=chunk,
        dtype={**{c: np.float32 for c in date_cols}, "Id": np.int64},
        nrows=limite_linhas,
    )
    for i, ch in enumerate(leitor):
        ts = ch[date_cols].to_numpy(dtype=np.float32, copy=False)
        visivel = ~np.isnan(ts)

        with np.errstate(all="ignore"):
            t_min = np.nanmin(ts, axis=1)
            t_max = np.nanmax(ts, axis=1)

        presenca = np.column_stack([visivel[:, indices[e]].any(axis=1) for e in estacoes])

        bloco = pd.DataFrame({
            "Id": ch["Id"].to_numpy(),
            "t_min": t_min.astype(np.float32),
            "t_max": t_max.astype(np.float32),
            "n_estacoes": presenca.sum(axis=1).astype(np.int16),
        })
        for j, e in enumerate(estacoes):
            bloco[f"vis_{e}"] = presenca[:, j].astype(np.int8)

        partes.append(bloco)
        total += len(ch)
        del ch, ts, visivel, presenca
        gc.collect()

        if i % 2 == 0:
            log(f"  chunk {i:>3} | {total:>9,} peças | {time.perf_counter() - t0:6.1f}s")

    meta = pd.concat(partes, ignore_index=True)
    del partes
    gc.collect()

    # --- rótulo: 10 MB, o "passo barato primeiro" -----------------------------
    log("lendo Id + Response de train_numeric.csv (~10 MB)...")
    resp = pd.read_csv(
        f_num, usecols=["Id", "Response"], dtype={"Id": np.int64, "Response": np.int8},
        nrows=limite_linhas,
    )
    meta = meta.merge(resp, on="Id", how="left", validate="one_to_one")

    # --- eixo temporal --------------------------------------------------------
    meta["duracao"] = (meta["t_max"] - meta["t_min"]).astype(np.float32)
    meta["semana"] = np.floor(meta["t_min"] * UNIDADE_TEMPO_SEMANAS).astype("Int32")
    meta["quinzena"] = (meta["semana"] // 2).astype("Int32")

    meta = meta.sort_values("t_min", kind="stable").reset_index(drop=True)

    log(f"\nmeta construída: {len(meta):,} peças × {meta.shape[1]} colunas")
    log(f"  taxa de falha : {meta['Response'].mean():.4%}")
    log(f"  t_min         : {meta['t_min'].min():.1f} → {meta['t_min'].max():.1f}")
    log(f"  semanas       : {meta['semana'].min()} → {meta['semana'].max()}")
    sem = meta.groupby("semana", observed=True).size()
    log(f"  peças/semana  : mediana {sem.median():,.0f}  (min {sem.min():,}  max {sem.max():,})")
    return meta


# =============================================================================
#  3. Seleção de features — duas passagens, nunca uma carga inteira
# =============================================================================
def selecionar_features(
    f_num: Path,
    *,
    n_alvo: int = N_FEATURES_ALVO,
    chunk: int = CHUNK,
    limite_linhas: int | None = None,
) -> list[str]:
    """Passagem 1: mede cobertura e variância. Devolve as `n_alvo` melhores.

    Critério, nesta ordem:
      1. cobertura ≥ 5%    (feature quase sempre NaN não é monitorável)
      2. variância > 0     (constante não tem distribuição para driftar)
      3. diversidade de estações — no máximo 6 features por estação

    O item 3 é o que diferencia esta seleção de um `nlargest(160)` ingênuo.
    Sem ele, as 160 features escolhidas se concentram em 12 estações e o
    monitor fica cego a metade da fábrica. Estamos escolhendo um SENSOR
    ESPACIAL, não um top-k de qualidade.
    """
    log("passagem 1/2 — cobertura e variância por feature...")
    cols = pd.read_csv(f_num, nrows=0).columns.tolist()
    feats = [c for c in cols if c not in ("Id", "Response")]

    n_total = 0
    n_validos = np.zeros(len(feats), dtype=np.int64)
    soma = np.zeros(len(feats))
    soma_q = np.zeros(len(feats))

    leitor = pd.read_csv(
        f_num, chunksize=chunk, usecols=feats,
        dtype={c: np.float32 for c in feats}, nrows=limite_linhas,
    )
    for i, ch in enumerate(leitor):
        arr = ch.to_numpy(dtype=np.float64, copy=False)
        val = ~np.isnan(arr)
        n_validos += val.sum(axis=0)
        soma += np.nansum(arr, axis=0)
        soma_q += np.nansum(arr**2, axis=0)
        n_total += len(ch)
        del ch, arr, val
        gc.collect()
        if i % 3 == 0:
            log(f"  {n_total:>9,} linhas")

    with np.errstate(all="ignore"):
        media = soma / np.maximum(n_validos, 1)
        var = soma_q / np.maximum(n_validos, 1) - media**2

    perfil = pd.DataFrame({
        "col": feats,
        "cobertura": n_validos / max(n_total, 1),
        "variancia": np.where(n_validos > 100, var, 0.0),
    })
    topo = parsear_topologia(feats)
    perfil = perfil.merge(topo[["col", "station", "linha", "estacao"]], on="col", how="left")

    elegiveis = perfil.query("cobertura >= @COBERTURA_MINIMA and variancia > 1e-12").copy()
    log(f"  {len(elegiveis):,} de {len(feats):,} features elegíveis")

    # diversidade: no máximo 6 por estação, priorizando cobertura
    elegiveis = elegiveis.sort_values(["cobertura", "variancia"], ascending=False)
    escolhidas = (
        elegiveis.groupby("station", observed=True, group_keys=False)
        .head(6)
        .head(n_alvo)["col"]
        .tolist()
    )
    if len(escolhidas) < n_alvo:  # completa se a cota por estação foi restritiva
        resto = [c for c in elegiveis["col"] if c not in set(escolhidas)]
        escolhidas += resto[: n_alvo - len(escolhidas)]

    n_est = perfil.query("col in @escolhidas")["station"].nunique()
    log(f"  {len(escolhidas)} features selecionadas, distribuídas em {n_est} estações")
    return escolhidas


def carregar_features(
    f_num: Path, features: list[str], *, chunk: int = CHUNK, limite_linhas: int | None = None
) -> pd.DataFrame:
    """Passagem 2: lê APENAS as colunas escolhidas. ~1.2 M × 160 float32 = 760 MB."""
    log("passagem 2/2 — carregando as features selecionadas...")
    usar = ["Id", *features, "Response"]
    tipos = {**{c: np.float32 for c in features}, "Id": np.int64, "Response": np.int8}
    partes = [
        ch for ch in pd.read_csv(
            f_num, usecols=usar, dtype=tipos, chunksize=chunk, nrows=limite_linhas
        )
    ]
    df = pd.concat(partes, ignore_index=True)
    log(f"  {df.shape[0]:,} × {df.shape[1]} | {df.memory_usage(deep=True).sum() / 1e6:.0f} MB")
    return df


# =============================================================================
#  4. Verificação — nunca publique um artefato que você não conferiu
# =============================================================================
def verificar(meta: pd.DataFrame, feats: pd.DataFrame) -> bool:
    """Checagens que, se falharem, invalidam tudo que vem depois."""
    log("\nverificando artefatos...")
    ok = True

    def chk(nome: str, cond: bool, obs: str = "") -> None:
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'✅' if cond else '❌'} {nome:<46} {obs}")

    chk("Id único em meta", meta["Id"].is_unique, f"{meta['Id'].nunique():,}")
    chk("Id único em features", feats["Id"].is_unique, f"{feats['Id'].nunique():,}")
    chk("mesmo conjunto de Ids", set(meta["Id"]) == set(feats["Id"]))
    chk("Response sem nulos", meta["Response"].notna().all())
    chk("taxa de falha ~0.58%", 0.004 < meta["Response"].mean() < 0.008,
        f"{meta['Response'].mean():.4%}")
    chk("t_min monotônico após sort", meta["t_min"].is_monotonic_increasing)
    chk("≥ 15 semanas de histórico", meta["semana"].nunique() >= 15,
        f"{meta['semana'].nunique()} semanas")
    chk("colunas vis_ presentes", sum(c.startswith("vis_") for c in meta.columns) > 50,
        f"{sum(c.startswith('vis_') for c in meta.columns)} estações")

    # a checagem que pega o bug mais caro: falha concentrada numa semana só
    por_sem = meta.groupby("semana", observed=True)["Response"].mean()
    chk("falha distribuída no tempo", por_sem.std() < por_sem.mean() * 2,
        f"cv={por_sem.std() / max(por_sem.mean(), 1e-9):.2f}")
    return ok


# =============================================================================
#  main
# =============================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entrada", type=Path, required=True, help="Pasta com os CSVs da Kaggle.")
    ap.add_argument("--saida", type=Path, default=Path("data"))
    ap.add_argument("--n-features", type=int, default=N_FEATURES_ALVO)
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--amostra", type=int, default=None,
                    help="Limita o nº de linhas — para testar o script em 2 min.")
    args = ap.parse_args()

    f_num = args.entrada / "train_numeric.csv"
    f_date = args.entrada / "train_date.csv"
    for f in (f_num, f_date):
        if not f.exists():
            print(f"❌ não encontrado: {f}\n"
                  "   kaggle competitions download -c bosch-production-line-performance",
                  file=sys.stderr)
            return 2
        log(f"{f.name:<24} {f.stat().st_size / 1e9:.2f} GB")

    args.saida.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    meta = construir_meta(f_date, f_num, chunk=args.chunk, limite_linhas=args.amostra)
    features = selecionar_features(f_num, n_alvo=args.n_features,
                                   chunk=args.chunk, limite_linhas=args.amostra)
    feats = carregar_features(f_num, features, chunk=args.chunk, limite_linhas=args.amostra)

    if not verificar(meta, feats):
        print("\n❌ verificação falhou. NÃO publique estes arquivos.", file=sys.stderr)
        return 1

    p_meta = args.saida / "meta_bosch.parquet"
    p_feat = args.saida / "bosch_num_160feats.parquet"
    log("\ngravando parquet...")
    # row_group menor = HTTP Range mais granular = leitura remota mais barata
    meta.to_parquet(p_meta, compression=COMPRESSAO, index=False, row_group_size=100_000)
    feats.to_parquet(p_feat, compression=COMPRESSAO, index=False, row_group_size=50_000)

    log("")
    log("=" * 58)
    for p in (p_meta, p_feat):
        log(f"  {p.name:<32} {p.stat().st_size / 1e6:>7.1f} MB")
    log(f"  tempo total: {(time.perf_counter() - t0) / 60:.1f} min")
    log("=" * 58)
    log("\nPróximo passo — publicar no GitHub Releases (NÃO no repo git):")
    log("  gh release create dados-v1 \\")
    log(f"     {p_meta} {p_feat} data/detector_config_referencia_v1.json \\")
    log('     --title "Dados derivados — PyBR 2026" \\')
    log('     --notes "Bosch pré-processado. Ver scripts/preprocess_bosch.py."')
    return 0


if __name__ == "__main__":
    sys.exit(main())
