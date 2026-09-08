"""
driftkit.state — contrato versionado do detector e histórico de decisões.

Caminho no repositório: src/driftkit/state.py

    detector_config.json é o artefato que atravessa a Parte I e a Parte II.

O que ele carrega, e por que a distinção é o ponto pedagógico da ponte:

    MÉTODO (transfere sem alteração)          PARÂMETRO (é remedido)
    ----------------------------------        ------------------------------
    o predicado `p < α/m E efeito > piso`     o valor do piso
    as 5 guardas e sua ordem                  k de n, cooldown
    as assinaturas diagnósticas               limiar de violação de range
    a exigência de teste A/A                  min_obs

Um limiar calibrado com 9 features, 2.700 amostras/janela e prevalência de 29%
NÃO se aplica a 160 features, 40 mil amostras/janela e prevalência de 0.58%.
Quem transporta o número em vez do método está fazendo chute com aparência de
rigor — e o `fator_amplificacao` registrado aqui mede exatamente o tamanho
desse erro.
"""

from __future__ import annotations

import json
import numpy as np
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .policy import Decisao

__all__ = [
    "Contrato",
    "salvar_contrato",
    "carregar_contrato",
    "registrar_decisao",
    "historico",
    "ultima_decisao",
    "fator_amplificacao",
]

NOME_PADRAO = "detector_config.json"
NOME_HISTORICO = "historico_decisoes.jsonl"


def _agora() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ambiente() -> dict[str, str]:
    """Sem isto, 'reproduzível' é uma palavra bonita."""
    import numpy
    import pandas
    import sklearn

    return {
        "python": sys.version.split()[0],
        "plataforma": platform.platform(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "sklearn": sklearn.__version__,
    }


@dataclass
class Contrato:
    """O contrato calibrado do detector, serializável para JSON."""

    versao: str
    origem: str
    features: dict[str, list[str]]
    limiares: dict[str, float | int]
    guardas_retreino: dict[str, Any] = field(default_factory=dict)
    assinaturas_diagnosticas: dict[str, str] = field(default_factory=dict)
    ranges: dict[str, list[float]] = field(default_factory=dict)
    fixture: dict[str, Any] = field(default_factory=dict)
    suite: dict[str, int] = field(default_factory=dict)
    linhagem: dict[str, Any] = field(default_factory=dict)
    criado_em: str = field(default_factory=_agora)
    ambiente: dict[str, str] = field(default_factory=_ambiente)

    # -- validação -----------------------------------------------------------
    def __post_init__(self) -> None:
        obrig = {"psi_piso_aa", "psi_alarme", "alpha", "min_obs"}
        if faltando := obrig - set(self.limiares):
            raise ValueError(f"contrato incompleto: faltam limiares {sorted(faltando)}")
        if self.limiares["psi_alarme"] <= self.limiares["psi_piso_aa"]:
            raise ValueError(
                f"psi_alarme ({self.limiares['psi_alarme']}) <= "
                f"psi_piso_aa ({self.limiares['psi_piso_aa']}). "
                "Um limiar dentro do ruído do próprio instrumento produz "
                "alarme perpétuo. Recalibre."
            )
        if self.suite and self.suite.get("pass", 0) < self.suite.get("total", 0):
            raise ValueError(
                f"suíte com falhas ({self.suite['pass']}/{self.suite['total']}). "
                "Um detector com teste vermelho não vai para produção. "
                "Este é o ponto inteiro do tutorial."
            )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent, ensure_ascii=False)

    def resumo(self) -> str:
        f = self.features
        return (
            f"contrato {self.versao}  (origem: {self.origem})\n"
            f"  features   : {len(f.get('num', []))} num + {len(f.get('cat', []))} cat\n"
            f"  piso A/A   : {self.limiares['psi_piso_aa']:.5f}\n"
            f"  psi_alarme : {self.limiares['psi_alarme']:.5f}"
            f"  ({self.limiares['psi_alarme'] / max(self.limiares['psi_piso_aa'], 1e-9):.1f}× o piso)\n"
            f"  min_obs    : {self.limiares['min_obs']}\n"
            f"  suíte      : {self.suite.get('pass', '?')}/{self.suite.get('total', '?')} verdes\n"
            f"  criado em  : {self.criado_em}"
        )


# =============================================================================
#  persistência
# =============================================================================
def _dir() -> Path:
    from .data import dir_dados

    return dir_dados()


def salvar_contrato(c: Contrato, caminho: str | Path | None = None) -> Path:
    p = Path(caminho) if caminho else _dir() / NOME_PADRAO
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(c.to_json(), encoding="utf-8")
    print(f"✅ contrato {c.versao} salvo em {p.name}")
    return p


def carregar_contrato(caminho: str | Path | None = None) -> dict:
    """Carrega o contrato. Se ausente, cai no fallback distribuído.

    Degradação graciosa é requisito de sala de aula, não luxo: em 3h30 alguém
    VAI perder a sessão do Colab entre os blocos. `raise` cru aqui trava a
    pessoa e ela perde o resto do tutorial.
    """
    if caminho is not None:
        p = Path(caminho)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    proprio = _dir() / NOME_PADRAO
    if proprio.exists():
        return json.loads(proprio.read_text(encoding="utf-8"))

    from .data import carregar_contrato_v1

    return carregar_contrato_v1()


# =============================================================================
#  histórico de decisões — é o que dá memória ao pipeline
# =============================================================================
def registrar_decisao(d: Decisao, caminho: str | Path | None = None) -> Path:
    """Anexa a decisão a um JSONL.

    Sem histórico persistente, as guardas 1 (persistência k de n) e 2b
    (cooldown) são impossíveis: ambas dependem de saber o que aconteceu nas
    janelas anteriores. É por isso que um `for` de notebook não é um pipeline —
    ele perde a memória ao fim da célula.
    """
    p = Path(caminho) if caminho else _dir() / NOME_HISTORICO
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": _agora(), **d.to_dict()}, ensure_ascii=False) + "\n")
    return p


def historico(caminho: str | Path | None = None) -> list[dict]:
    p = Path(caminho) if caminho else _dir() / NOME_HISTORICO
    if not p.exists():
        return []
    return [json.loads(li) for li in p.read_text(encoding="utf-8").splitlines() if li.strip()]


def ultima_decisao(caminho: str | Path | None = None) -> dict | None:
    h = historico(caminho)
    return h[-1] if h else None


# =============================================================================
#  o entregável conceitual da ponte
# =============================================================================
def comparar_regimes(
    v1: dict,
    v2: dict,
    *,
    nome_v1: str = "sintético",
    nome_v2: str = "real",
) -> dict:
    """O que acontece quando você transporta um limiar entre regimes.

    Substitui `fator_amplificacao()`, que partia de uma premissa falsa: a de
    que o piso de ruído CRESCE do sintético para o real. Ele não cresce — ele
    é previsível, e cai com o tamanho da janela:

        piso ∝ (B-1)·(1/n + 1/m)

    A janela do Bosch é ~15× maior que a da fixture. O ganho de n domina
    folgadamente o efeito de ter 160 features em vez de 9. O piso real é
    MENOR, não maior.

    E isso torna a lição mais forte, não mais fraca:

        Um limiar fixo não fica ruidoso quando o volume cresce.
        Ele fica CEGO. Você paga por mais dados e joga fora
        todo o poder estatístico que eles compram.

    Parameters
    ----------
    v1, v2 : dict
        Saídas de `DriftDetector.calibrar_piso()` nos dois regimes.
    """
    def _extrai(v: dict) -> dict:
        return {
            "n": v.get("n_referencia"),
            "k": v.get("k_features_num"),
            "bins": v.get("bins"),
            "previsto": v.get("piso_analitico", np.nan),
            "medido": v.get("piso_aa", np.nan),
            "limiar": v.get("psi_alarme_sugerido", np.nan),
            "H": v.get("H", np.nan),
            "referencia": v.get("referencia", "?"),
        }

    a, b = _extrai(v1), _extrai(v2)
    r = {f"{nome_v1}_{k}": val for k, val in a.items()}
    r |= {f"{nome_v2}_{k}": val for k, val in b.items()}

    # o número que importa: o limiar do regime 1, medido em pisos do regime 2
    folga = a["limiar"] / b["previsto"] if b["previsto"] > 0 else np.nan
    r["limiar_v1_em_pisos_de_v2"] = round(folga, 1) if np.isfinite(folga) else None
    r["razao_dos_pisos"] = (round(b["previsto"] / a["previsto"], 3)
                            if a["previsto"] > 0 else None)

    if not np.isfinite(folga):
        r["veredicto"] = "não avaliável"
    elif folga > 10:
        r["veredicto"] = (
            f"o limiar de {nome_v1} é {folga:.0f}× o piso de {nome_v2}. "
            f"Transportá-lo torna o detector CEGO: só drift catastrófico "
            f"cruzaria. Você tem {b['n']:,} observações por janela e está "
            f"usando um limiar calibrado para ~{n_equivalente(a['limiar']):.0f}."
        )
    elif folga > 3:
        r["veredicto"] = (
            f"o limiar de {nome_v1} é {folga:.0f}× o piso de {nome_v2}: "
            "conservador, perde drift moderado. Recalibre e ganhe sensibilidade."
        )
    elif folga > 1:
        r["veredicto"] = (
            "o limiar transfere sem catástrofe — coincidência dos tamanhos de "
            "janela, não propriedade do método. Recalibre mesmo assim."
        )
    else:
        r["veredicto"] = (
            f"o limiar de {nome_v1} está ABAIXO do piso de {nome_v2}: "
            "alarme perpétuo. Em duas semanas ninguém olha o dashboard."
        )

    # o diagnóstico que o fator antigo não tinha
    ruins = [n for n, v in ((nome_v1, a), (nome_v2, b))
             if v["referencia"] == "CONTAMINADA"]
    if ruins:
        r["aviso"] = (
            f"referência CONTAMINADA em: {', '.join(ruins)}. "
            "Antes de discutir limiar, conserte o gabarito."
        )
    return r


def fator_amplificacao(*_args, **_kw):  # pragma: no cover
    raise NotImplementedError(
        "fator_amplificacao() foi removida: a premissa estava errada.\n"
        "O piso de ruído do PSI é ~(B-1)(1/n+1/m) — ele CAI quando a janela\n"
        "cresce, então ele não 'amplifica' do sintético para o real.\n"
        "Use comparar_regimes(v1, v2). Ver driftkit.detectors.piso_analitico."
    )
