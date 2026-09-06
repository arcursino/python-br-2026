"""
tests/conftest.py — fixtures compartilhadas da suíte.

Este arquivo é, ele próprio, material didático. Ele demonstra o ponto que o
notebook `01` argumenta em prosa:

    O detector é o PRODUTO. O dataset sintético é o TESTE UNITÁRIO.

Repare no `scope="session"` das fixtures caras. Gerar 117 mil linhas a cada
teste tornaria a suíte lenta o bastante para você parar de rodá-la — e uma
suíte que você não roda é pior que nenhuma, porque dá falsa segurança.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from driftkit.detectors import DriftDetector
from driftkit.fixtures import CAT, NUM, RANGES, gerar_fixture
from driftkit.modelo import treinar

SEED_PADRAO = 7
SEEDS_ROBUSTEZ = (7, 11, 23, 42, 101, 202, 314, 999)


def pytest_configure(config: pytest.Config) -> None:
    np.random.seed(SEED_PADRAO)  # determinismo de qualquer legado que ainda use o RNG global


# =============================================================================
#  dados
# =============================================================================
@pytest.fixture(scope="session")
def df() -> pd.DataFrame:
    """A fixture completa, seed 7. Gerada UMA vez por sessão."""
    return gerar_fixture(seed=SEED_PADRAO)


@pytest.fixture(scope="session")
def referencia(df: pd.DataFrame) -> pd.DataFrame:
    """Janela-base: dias 1–29, antes de qualquer evento plantado."""
    return df.query("dia < 30")


@pytest.fixture(scope="session")
def detector(referencia: pd.DataFrame) -> DriftDetector:
    """Detector com limiar CALIBRADO — não com o 0.1 do folclore.

    O limiar sai do teste A/A sobre a própria referência. É a diferença entre
    um número medido e um número herdado de um blog post.
    """
    base = DriftDetector.from_reference(
        referencia, num=list(NUM), cat=list(CAT), ranges=RANGES
    )
    piso = base.calibrar_piso(n_repeticoes=25, seed=SEED_PADRAO)
    return base.com(psi_limiar=piso["psi_alarme_sugerido"])


@pytest.fixture(scope="session")
def piso_aa(referencia: pd.DataFrame) -> float:
    base = DriftDetector.from_reference(referencia, num=list(NUM), cat=list(CAT))
    return base.calibrar_piso(n_repeticoes=25, seed=SEED_PADRAO)["piso_aa"]


@pytest.fixture(scope="session")
def modelo(referencia: pd.DataFrame):
    """Modelo treinado na referência. É o paciente, não o objeto de estudo."""
    return treinar(referencia, num=NUM, cat=CAT)


# =============================================================================
#  parametrização compartilhada
# =============================================================================
@pytest.fixture(params=SEEDS_ROBUSTEZ, ids=lambda s: f"seed{s}")
def seed_robustez(request: pytest.FixtureRequest) -> int:
    """Um seed que passa NÃO é um teste que passa.

    Esta fixture força os testes de controle negativo a rodarem em 8 fixtures
    independentes. Se o silêncio do TC-1 só acontece com seed 7, ele é
    coincidência, não propriedade do detector.
    """
    return request.param
