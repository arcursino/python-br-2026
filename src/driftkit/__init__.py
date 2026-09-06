"""
driftkit — detecção e mitigação de drift com calibração por dataset sintético.

Caminho no repositório: src/driftkit/__init__.py

Tutorial "Rumo ao Desconhecido: Tratando Drift em Machine Learning" — PyBR 2026.

A tese do pacote, em três linhas
--------------------------------
1. Um detector de drift é um INSTRUMENTO DE MEDIÇÃO, e instrumento se calibra
   contra um padrão conhecido antes de medir o desconhecido.
2. O drift que mais move as distribuições é frequentemente o que menos
   machuca; o que mais machuca costuma ser quase invisível nelas.
3. Um pipeline que sabe se RECUSAR a retreinar é mais maduro que um que
   retreina rápido.

Mapa dos módulos
----------------
    fixtures.py   o dataset sintético — a FIXTURE, com drift plantado
    detectors.py  os instrumentos: psi, js_cat, wasserstein, DriftDetector
    modelo.py     o modelo sob monitoramento + métricas robustas a 0.58%
    policy.py     as 5 guardas de retreino (a 5ª é a que falta no mercado)
    state.py      contrato versionado (v1→v2) e histórico de decisões
    data.py       carga do Bosch com cache local / leitura por URL
    cli.py        `driftkit check | decide | retrain | simular` (exit codes)
    notebook.py   utilitários de sala de aula: checkpoint, requer, driftkit()
    testing.py    gabaritos dos exercícios "você implementa"

Uso rápido
----------
>>> from driftkit.fixtures import gerar_fixture, janela
>>> from driftkit.detectors import DriftDetector
>>> df = gerar_fixture(seed=7)
>>> det = DriftDetector.from_reference(
...     df.query("dia < 30"), num=["torque_medido"], cat=["aro"]
... )
>>> det.report(janela(df, 55)).n_drift        # TC-1: controle negativo
0
"""

from __future__ import annotations

__version__ = "1.0.0"
__author__ = "Seu Nome"

from .detectors import DriftDetector, RelatorioDrift, js_cat, psi, viola_range, wasserstein
from .fixtures import EVENTOS, FEATS, RANGES, cpk, gerar_fixture, janela, ppm_fora_spec
from .policy import Acao, Causa, Decisao, PoliticaRetreino

__all__ = [
    "__version__",
    # detectores
    "DriftDetector",
    "RelatorioDrift",
    "psi",
    "js_cat",
    "wasserstein",
    "viola_range",
    # fixture
    "gerar_fixture",
    "janela",
    "cpk",
    "ppm_fora_spec",
    "EVENTOS",
    "FEATS",
    "RANGES",
    # política
    "PoliticaRetreino",
    "Decisao",
    "Acao",
    "Causa",
]