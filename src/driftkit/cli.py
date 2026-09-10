"""
driftkit.cli — o artefato que transforma a política em infraestrutura.

Caminho no repositório: src/driftkit/cli.py

    O EXIT CODE É A POLÍTICA.

Esta é a ideia central deste arquivo. Enquanto a decisão de retreinar mora num
`if` dentro de uma célula de notebook, ela é uma opinião. Quando ela vira um
código de saída de processo, ela é infraestrutura: o GitHub Actions, o Airflow,
o cron, o Argo — qualquer orquestrador sabe ler um exit code, e nenhum deles
precisa saber o que é PSI.

    exit  0  → sem drift acionável ............. não faz nada
    exit 10  → retreino APROVADO ............... dispara o job de retreino
    exit 20  → retreino BLOQUEADO .............. abre ticket de engenharia
    exit 30  → dados insuficientes ............. não decide, não silencia

O `20` é o que não existe em nenhum pipeline de MLOps que eu conheça. Ele é a
diferença entre um sistema que retreina e um sistema que ENTENDE.

Uso
---
    driftkit calibrar --seed 7                  # gera o contrato v1 (Parte I)
    driftkit calibrar --modo bloco --eixo dia   # diagnostica a REFERÊNCIA
    driftkit check   data/janela.parquet        # mede. Não decide.
    driftkit decide  data/janela.parquet        # decide. Exit code é a resposta.
    driftkit retrain data/janela.parquet        # retreina — se autorizado.
    driftkit simular --de 30 --ate 130          # o pipeline no tempo
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from . import __version__

app = typer.Typer(
    name="driftkit",
    help="Detecção de drift com calibração — e um pipeline que sabe se recusar a retreinar.",
    no_args_is_help=True,
    add_completion=False,
)

ERRO = typer.colors.RED
OK = typer.colors.GREEN
AVISO = typer.colors.YELLOW


def _eco(msg: str = "", cor: str | None = None, negrito: bool = False) -> None:
    typer.secho(msg, fg=cor, bold=negrito)


def _carregar(caminho: Path) -> pd.DataFrame:
    if not caminho.exists():
        _eco(f"arquivo não encontrado: {caminho}", ERRO)
        raise typer.Exit(2)
    if caminho.suffix in {".parquet", ".pq"}:
        return pd.read_parquet(caminho)
    if caminho.suffix == ".csv":
        return pd.read_csv(caminho)
    _eco(f"formato não suportado: {caminho.suffix}", ERRO)
    raise typer.Exit(2)


# =============================================================================
@app.command()
def versao() -> None:
    """Mostra a versão."""
    _eco(f"driftkit {__version__}")


# =============================================================================
@app.command()
def calibrar(
    seed: int = typer.Option(7, help="Seed da fixture sintética."),
    saida: Path = typer.Option(
        Path("data/detector_config_referencia_v1.json"),
        help="Onde gravar o contrato. O nome default é o mesmo do asset do Release.",
    ),
    repeticoes: int = typer.Option(40, help="Repetições do teste A/A."),
    modo: str = typer.Option(
        "ambos",
        "--modo",
        help="aleatorio | bloco | ambos. 'bloco' é o que enxerga referência contaminada.",
        case_sensitive=False,
    ),
    n_blocos: int = typer.Option(4, "--n-blocos", help="Número de blocos no split em bloco."),
    eixo: str = typer.Option(
        "dia",
        "--eixo",
        help="Coluna que ordena a referência antes do split em bloco (tempo, lote, turno).",
    ),
) -> None:
    """Calibra o detector contra a fixture sintética E diagnostica a referência.

    É a Parte I inteira, executável sem notebook — o que torna a calibração
    reproduzível em CI. Nenhum limiar deste projeto foi escolhido a olho.

    \b
    Dois modos, duas perguntas diferentes:
      --modo aleatorio  → qual é o ruído do INSTRUMENTO?
      --modo bloco      → a minha REFERÊNCIA merece confiança?

    O segundo é o que importa mais, e é o que quase ninguém faz. O split
    aleatório é provadamente CEGO a uma referência que contém mais de um
    regime: a permutação distribui os regimes igualmente entre as metades,
    e mistura embaralhada é permutacionalmente trocável. Dá verde sempre.
    """
    from .detectors import DriftDetector, n_equivalente, piso_analitico
    from .fixtures import CAT, NUM, RANGES, gerar_fixture, validar_fixture
    from .state import Contrato, salvar_contrato

    modo = modo.lower()
    if modo not in {"aleatorio", "bloco", "ambos"}:
        _eco(f"--modo inválido: {modo!r}. Use aleatorio | bloco | ambos.", ERRO)
        raise typer.Exit(2)

    _eco("→ gerando fixture...", negrito=True)
    df = gerar_fixture(seed=seed)

    _eco("→ validando a FIXTURE (antes de acreditar em qualquer detector)...")
    checks = validar_fixture(df)
    falhas = checks.query("status == 'FAIL'")
    _eco(checks.to_string(index=False))
    if len(falhas):
        _eco("\n⛔ a fixture está errada. Nada abaixo disto teria significado.", ERRO)
        raise typer.Exit(1)

    ref = df.query("dia < 30")
    if modo in {"bloco", "ambos"}:
        if eixo not in ref.columns:
            _eco(
                f"\n⚠️  coluna --eixo {eixo!r} não existe na referência. "
                "Sem ordenação, o split em bloco vira um split aleatório caro.",
                AVISO,
            )
        else:
            ref = ref.sort_values(eixo)

    det = DriftDetector.from_reference(ref, num=list(NUM), cat=list(CAT), ranges=RANGES)

    _eco(f"\n→ calibrando o piso de ruído (modo={modo}, {repeticoes} repetições)...")
    piso = det.calibrar_piso(
        n_repeticoes=repeticoes, seed=seed, modo=modo, n_blocos=n_blocos
    )
    for k, v in piso.items():
        _eco(f"   {k:<26} {v}")

    # ---- o piso tem forma fechada: confira o medido contra o previsto -------
    n_meia = len(ref) // 2
    previsto = piso_analitico(n_meia, n_meia, bins=10)
    # Dois números com forma fechada aparecem nesta tela e NÃO são o mesmo:
    #   - este aqui: o piso de UMA feature isolada;
    #   - piso['piso_analitico']: o mesmo cálculo corrigido para as k features
    #     monitoradas simultaneamente (Bonferroni).
    # O rótulo diz qual é qual — a diferença entre eles É o preço estatístico
    # de monitorar seis features em vez de uma.
    _eco(
        f"\n   piso teórico POR FEATURE     {previsto:.6f}"
        f"   [χ²₀.₉₅(B-1)·(1/n+1/m)], sem Bonferroni"
    )
    _eco(
        f"   o contrato grava       {piso['piso_analitico']:.6f}"
        f"   — o mesmo cálculo corrigido para {piso['k_features_num']} features."
    )
    _eco(
        f"   o limiar 0.1 que você herdou é o piso de uma janela de "
        f"~{n_equivalente(0.10):.0f} observações."
    )
    _eco(
        f"\n   O 'PSI > 0.1' que você leu no blog não é lei da natureza.\n"
        f"   O piso DESTE detector, nesta referência, é {piso['piso_aa']:.5f}.",
        AVISO,
    )

    # ---- o diagnóstico que o split aleatório não consegue dar ---------------
    h = piso.get("H")
    if h is not None:
        if h < 2:
            _eco(f"\n   H = {h:.2f} → referência HOMOGÊNEA no eixo {eixo!r}. "
                 "Limiar confiável.", OK)
        elif h < 5:
            _eco(f"\n   H = {h:.2f} → SUSPEITA. Investigue antes de confiar no limiar.",
                 AVISO)
        else:
            _eco(
                f"\n   H = {h:.2f} → referência CONTAMINADA: ela contém mais de um "
                f"regime.\n   Bloco mais divergente: {piso.get('bloco_mais_divergente', '?')}\n"
                "   Nenhum limiar te salva disso. Encurte ou segmente a referência\n"
                "   ANTES de calibrar qualquer coisa.",
                ERRO,
            )

    contrato = Contrato(
        versao="v1-sintetico",
        origem=f"fixture Tire & Wheel, seed={seed}",
        features={"num": list(NUM), "cat": list(CAT)},
        limiares={
            "psi_piso_aa": piso["piso_aa"],
            "psi_piso_analitico": previsto,
            "psi_piso_bloco": piso.get("piso_bloco"),
            "H": h,
            "psi_alarme": piso["psi_alarme_sugerido"],
            "alpha": 0.05,
            "min_obs": 200,
            "bins": 10,
            "sigma_carta": 3.0,
        },
        guardas_retreino={
            "k_de_n": [3, 5],
            "cooldown_janelas": 14,
            "fator_piso": 3.0,
            "ganho_minimo": 0.01,
        },
        assinaturas_diagnosticas={
            "pipeline": "violação de range de engenharia > 10% dos registros",
            "hardware": "degradação localizada em segmento com P(X) calmo",
            "negocio": "P(X) em drift com performance preservada",
            "modelo": "drift persistente sem assinatura física",
        },
        ranges={k: list(v) for k, v in RANGES.items()},
        fixture={
            "features_monitoradas": len(NUM) + len(CAT),
            "amostras_por_janela": int(len(df) / df["dia"].nunique() * 3),
            "prevalencia": round(float(ref["falha"].mean()), 4),
            "seed": seed,
            "calibracao": {"modo": modo, "n_blocos": n_blocos, "eixo": eixo},
        },
        suite={"pass": len(checks), "total": len(checks)},
    )
    salvar_contrato(contrato, saida)
    _eco("\n" + contrato.resumo())
    _eco(f"\n✅ contrato gravado em {saida}", OK)


# =============================================================================
@app.command()
def check(
    janela: Path = typer.Argument(..., help="Parquet/CSV com a janela atual."),
    contrato: Optional[Path] = typer.Option(None, help="detector_config.json."),
    referencia: Optional[Path] = typer.Option(None, help="Parquet da referência."),
    json_out: bool = typer.Option(False, "--json", help="Saída em JSON."),
) -> None:
    """Mede drift numa janela. NÃO decide nada.

    Separação deliberada: medir e decidir são responsabilidades diferentes,
    e misturá-las é a razão de tantos monitores de drift virarem geradores de
    alarme que ninguém lê.
    """
    from .detectors import DriftDetector
    from .fixtures import CAT, NUM, gerar_fixture
    from .state import carregar_contrato

    cfg = carregar_contrato(contrato)
    ref = _carregar(referencia) if referencia else gerar_fixture(
        seed=cfg.get("fixture", {}).get("seed", 7)
    ).query("dia < 30")
    det = DriftDetector.from_contract(cfg, ref) if referencia is None else \
        DriftDetector.from_reference(ref, num=list(NUM), cat=list(CAT))

    rel = det.report(_carregar(janela))
    if json_out:
        typer.echo(json.dumps(rel.to_dict(), indent=2, ensure_ascii=False))
    else:
        _eco(rel.resumo())
    raise typer.Exit(0)


# =============================================================================
@app.command()
def decide(
    janela: Path = typer.Argument(..., help="Parquet/CSV com a janela atual."),
    contrato: Optional[Path] = typer.Option(None),
    referencia: Optional[Path] = typer.Option(None),
    dia: Optional[int] = typer.Option(None, help="Dia/período da janela (guardas 1 e 2b)."),
    segmento: Optional[str] = typer.Option(None, help="Ex.: CEL-02."),
    auc_global: Optional[float] = typer.Option(None),
    auc_segmento: Optional[float] = typer.Option(None),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Aplica as 5 guardas e devolve a decisão COMO EXIT CODE.

    \b
    0  nada a fazer
    10 retreino APROVADO
    20 retreino BLOQUEADO (causa não-ML)
    30 dados insuficientes
    """
    from .detectors import DriftDetector
    from .fixtures import CAT, NUM, RANGES, gerar_fixture
    from .policy import Acao, PoliticaRetreino, tabela_das_quatro_guardas
    from .state import carregar_contrato, historico, registrar_decisao

    cfg = carregar_contrato(contrato)
    ref = _carregar(referencia) if referencia else gerar_fixture(
        seed=cfg.get("fixture", {}).get("seed", 7)
    ).query("dia < 30")
    det = DriftDetector.from_reference(
        ref, num=list(NUM), cat=list(CAT),
        psi_limiar=cfg["limiares"]["psi_alarme"],
        alpha=cfg["limiares"]["alpha"],
        min_obs=cfg["limiares"]["min_obs"],
        ranges=RANGES,
    )
    pol = PoliticaRetreino.from_contract(cfg, det)
    # reidrata a memória: guardas 1 e 2b dependem do passado
    pol._historico = [h["evidencias"].get("n_drift", 0) > 0 for h in historico()[-10:]]

    d = pol.avaliar(
        _carregar(janela), dia=dia, segmento=segmento,
        auc_global=auc_global, auc_segmento=auc_segmento,
    )
    registrar_decisao(d)

    if json_out:
        typer.echo(json.dumps(d.to_dict(), indent=2, ensure_ascii=False))
    elif not quiet:
        cor = {Acao.NADA: OK, Acao.RETREINAR: typer.colors.CYAN,
               Acao.BLOQUEAR: ERRO, Acao.INSUFICIENTE: AVISO}[d.acao]
        _eco(str(d), cor)
        if d.acao is Acao.BLOQUEAR:
            _eco("\n   As guardas clássicas, uma a uma:", negrito=True)
            _eco(tabela_das_quatro_guardas(d).to_string(index=False))
            _eco(
                "\n   Isto NÃO é um bug do pipeline. É a guarda 5 funcionando.\n"
                "   Nenhuma métrica de ML distinguiria este caso de um retreino legítimo.",
                AVISO,
            )
    raise typer.Exit(d.exit_code)


# =============================================================================
@app.command()
def retrain(
    janela: Path = typer.Argument(...),
    contrato: Optional[Path] = typer.Option(None),
    force: bool = typer.Option(False, "--force", help="Ignora a última decisão. Use com culpa."),
    saida: Path = typer.Option(Path("models/challenger.joblib")),
) -> None:
    """Retreina — e se RECUSA se `decide` não autorizou.

    O `--force` existe porque toda operação precisa de escape hatch. Mas ele
    grava no histórico que foi forçado: auditoria é o que separa exceção
    justificada de hábito.
    """
    import joblib

    from .fixtures import CAT, NUM
    from .modelo import metricas, treinar
    from .state import carregar_contrato, ultima_decisao

    ultima = ultima_decisao()
    if not force:
        if ultima is None:
            _eco("nenhuma decisão registrada. Rode `driftkit decide` antes.", ERRO)
            raise typer.Exit(2)
        if ultima["acao"] != "RETREINAR":
            _eco(f"⛔ retreino RECUSADO — última decisão foi {ultima['acao']}", ERRO)
            _eco(f"   motivo: {ultima['motivo']}")
            raise typer.Exit(20)
    else:
        _eco("⚠️  --force: ignorando a política. Isto fica registrado.", AVISO)

    cfg = carregar_contrato(contrato)
    dados = _carregar(janela)

    _eco("→ treinando challenger...")
    challenger = treinar(dados, num=NUM, cat=CAT)

    # ---- guarda 4 de verdade: shadow validation antes de promover -----------
    champion_path = saida.parent / "champion.joblib"
    if champion_path.exists():
        champion = joblib.load(champion_path)
        m_new = metricas(challenger, dados, num=NUM, cat=CAT)
        m_old = metricas(champion, dados, num=NUM, cat=CAT)
        if m_new.pr_auc is None or m_old.pr_auc is None:
            _eco("janela não estimável — promoção adiada.", AVISO)
            raise typer.Exit(30)
        ganho = m_new.pr_auc - m_old.pr_auc
        minimo = cfg["guardas_retreino"].get("ganho_minimo", 0.01)
        _eco(f"   champion PR-AUC={m_old.pr_auc:.4f} | challenger={m_new.pr_auc:.4f} "
             f"| Δ={ganho:+.4f} (mínimo {minimo})")
        if ganho < minimo:
            _eco("→ challenger não superou o champion. Modelo atual MANTIDO.", AVISO)
            raise typer.Exit(0)

    saida.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(challenger, saida)
    joblib.dump(challenger, champion_path)
    _eco(f"✅ challenger promovido a champion → {saida}", OK)
    raise typer.Exit(0)


# =============================================================================
@app.command()
def simular(
    de: int = typer.Option(30, help="Dia inicial."),
    ate: int = typer.Option(130, help="Dia final."),
    passo: int = typer.Option(5),
    seed: int = typer.Option(7),
    contrato: Optional[Path] = typer.Option(None),
) -> None:
    """Roda o pipeline dia a dia sobre a fixture — o 'cron' do tutorial.

    É o clímax do Bloco V: você vê o mesmo pipeline decidir NADA no TC-1,
    NADA no TC-2 (drift real, causa de negócio), e BLOQUEAR no TC-3 —
    exatamente onde um pipeline convencional retreinaria e destruiria o
    processo.
    """
    from .detectors import DriftDetector
    from .fixtures import CAT, NUM, RANGES, gabarito, gerar_fixture, janela as jan
    from .modelo import metricas, treinar
    from .policy import Acao, PoliticaRetreino
    from .state import carregar_contrato

    cfg = carregar_contrato(contrato)
    df = gerar_fixture(seed=seed)
    ref = df.query("dia < 30")

    det = DriftDetector.from_reference(
        ref, num=list(NUM), cat=list(CAT),
        psi_limiar=cfg["limiares"]["psi_alarme"],
        alpha=cfg["limiares"]["alpha"],
        min_obs=cfg["limiares"]["min_obs"],
        ranges=RANGES,
    )
    pol = PoliticaRetreino.from_contract(cfg, det)
    modelo = treinar(ref, num=NUM, cat=CAT)

    _eco(f"{'dia':>4} {'evento':<7} {'exit':>4}  {'ação':<13} {'causa':<10} motivo", negrito=True)
    _eco("-" * 104)
    icones = {Acao.NADA: "🟢", Acao.RETREINAR: "🔁", Acao.BLOQUEAR: "⛔", Acao.INSUFICIENTE: "⚪"}

    for dia in range(de, ate + 1, passo):
        w = jan(df, dia)
        if w.empty:
            continue
        seg = w.query("equipamento == 'CEL-02'")
        m_g = metricas(modelo, w, num=NUM, cat=CAT)
        m_s = metricas(modelo, seg, num=NUM, cat=CAT) if len(seg) > 50 else None

        d = pol.avaliar(
            w, dia=dia, segmento="CEL-02",
            auc_global=m_g.roc_auc,
            auc_segmento=m_s.roc_auc if m_s and m_s.estimavel else None,
        )
        ev = gabarito(dia)
        _eco(
            f"{dia:>4} {(ev.tc if ev else '—'):<7} {d.exit_code:>4}  "
            f"{icones[d.acao]} {d.acao.name:<11} {d.causa:<10} {d.motivo[:52]}",
            {Acao.BLOQUEAR: ERRO, Acao.RETREINAR: typer.colors.CYAN}.get(d.acao),
        )

    _eco(
        "\nLeia a coluna `exit`. É isso que o orquestrador consome.\n"
        "O `20` no TC-3 é o tutorial inteiro em um número.",
        negrito=True,
    )


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
