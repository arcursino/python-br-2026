# Rumo ao Desconhecido: Tratando Drift em Machine Learning

**Tutorial · Python Brasil 2026 · 3h30**

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SEU_USUARIO/pybr2026-drift/blob/main/notebooks/00_ambiente.ipynb)
[![CI](https://github.com/SEU_USUARIO/pybr2026-drift/actions/workflows/ci.yml/badge.svg)](https://github.com/SEU_USUARIO/pybr2026-drift/actions)

> Um detector de drift é um **instrumento de medição**. E ninguém valida um
> instrumento contra uma amostra de valor desconhecido — você calibra a balança
> com um peso padrão de 1 kg, e só depois pesa o que não conhece.

---

## ⚡ Comece aqui (2 minutos, faça ANTES do evento)

1. Clique no badge **Abrir no Colab** acima.
2. Rode a única célula do notebook `00_ambiente`.
3. Se aparecerem só ✅, você está pronto. Se aparecer ❌, me avise por e-mail.

**Faça isso em casa.**

> 💡 **Não precisa instalar nada.** Não precisa baixar os 14 GB da Kaggle.
> Não precisa de conta paga. Só de um navegador e uma conta Google.
>
> ⚠️ Se sua conta for institucional (`@universidade.br`), teste antes: algumas
> organizações bloqueiam o Colab por política. Nesse caso, use uma conta pessoal.

<details>
<summary><b>Prefere rodar local?</b> (opcional, para quem traz o próprio notebook)</summary>

```bash
git clone https://github.com/SEU_USUARIO/pybr2026-drift.git
cd pybr2026-drift
make setup      # uv sync, ~30s
make test       # se ficar verde, você está pronto
make demo       # o argumento do tutorial inteiro em 2 minutos
```
</details>

---

## O que você leva para casa

Um pacote Python instalável, testado e com CI — não uma pasta de notebooks.

```bash
driftkit calibrar                        # calibra o detector E diagnostica a referência
driftkit check   data/janela.parquet     # mede drift. NÃO decide nada.
driftkit decide  data/janela.parquet     # decide. O EXIT CODE é a resposta.
driftkit retrain data/janela.parquet     # retreina — se autorizado.
driftkit simular --de 30 --ate 130       # o pipeline decidindo, dia após dia.
```

O contrato com a esteira de CI/CD:

| exit | significado | o que o orquestrador faz |
|:---:|---|---|
| `0` | sem drift acionável | nada |
| `10` | retreino **aprovado** | dispara o job de retreino |
| `20` | retreino **bloqueado** | abre ticket para engenharia de processo |
| `30` | dados insuficientes | registra — e **não** confunde com "está tudo bem" |

O `20` é o que não existe em nenhum pipeline de MLOps convencional.

---

## A tese, em três linhas

1. O drift que mais **move as distribuições** é frequentemente o que **menos machuca**.
   O que mais machuca costuma ser quase invisível nelas.
2. O piso de ruído do seu detector é **previsível** — e o `PSI > 0.1` que você
   herdou corresponde a uma janela de ~340 observações, não à sua.
3. Um pipeline que sabe **se recusar** a retreinar é mais maduro que um que retreina rápido.

---

## 📐 O piso de ruído tem forma fechada

Este é o resultado que reorganizou o tutorial. O PSI é a divergência de
Jeffreys; sob a hipótese nula — duas amostras da **mesma** distribuição — ele é
um qui-quadrado escalado:

$$\mathrm{E}[\text{PSI}] = (B-1)\left(\tfrac{1}{n}+\tfrac{1}{m}\right)
\qquad
p_{95} = \chi^2_{0{,}95}(B-1)\left(\tfrac{1}{n}+\tfrac{1}{m}\right)$$

Verificado em 12 famílias de marginal (normal, lognormal, cauchy, pareto,
bimodal, contagem, zero-inflada): razão medido/previsto com **mediana 0,93** e
91% das células em `[0.3, 3.0]`. Distribuições degeneradas ficam *abaixo* da
previsão — bins colapsam e os graus de liberdade efetivos caem. **A fórmula é um
teto conservador.**

### Onde vive o `0.1`

Resolvendo $$p_{95} = 0{,}1$$ com $$B = 10$$:

| B (bins) | n em que `0.1` é puro ruído |
|---:|---:|
| 5 | 190 |
| **10** | **338** |
| 20 | 603 |

> O `PSI > 0.1` nunca foi um limiar de drift. É o piso de ruído de uma janela de
> ~300 observações — exatamente o regime de *scorecard* de crédito, de onde a
> regra veio.

E o corolário contraintuitivo:

> **Um monitor com limiar fixo fica progressivamente mais cego conforme o seu
> volume de dados cresce.** A 40.000 observações por janela o piso real é
> `0.0008`, e o `0.1` está **118× solto**. Você paga por mais dados e joga fora
> todo o poder estatístico que eles compram.

```python
from driftkit.detectors import piso_analitico, n_equivalente

piso_analitico(40_000, 40_000, bins=10)   # 0.00085
n_equivalente(0.10, bins=10)              # 338
```

---

## ⚠️ O teste A/A que todo mundo faz não testa o que você pensa

O conselho padrão — inclusive o que **este repositório dava até a v0.2** — é
"embaralhe sua referência, divida ao meio e meça". Isso mede o ruído do
instrumento, e é útil. Mas é **provadamente cego** ao problema mais grave que
uma referência pode ter: conter mais de um regime.

> Se a referência é uma mistura de dois regimes, a permutação distribui os dois
> **igualmente** entre as metades. As duas metades viram amostras da **mesma**
> mistura — e mistura embaralhada é permutacionalmente trocável. O teste dá
> verde. Sempre.

Medido (contaminação por um 2º regime a +1σ, no fim da janela):

| contaminação | split **aleatório** | split **em bloco** |
|---:|---:|---:|
| 0% | 1,0 | 0,6 |
| 5% | 1,0 | 0,8 |
| 10% | 1,0 | **2,8** |
| 20% | 1,0 | **9,6** |
| 50% | 1,0 | **55,8** |

Falso positivo sob $$H_0$$ verdadeiro: **0,0%** nas 12 famílias.

### O índice H

$$H = \frac{\text{piso medido no bloco}}{\text{piso analítico previsto}}$$

| H | leitura |
|---|---|
| `< 2` | referência homogênea no eixo testado — limiar confiável |
| `2 – 5` | suspeita — investigue antes de confiar |
| `≥ 5` | **CONTAMINADA**: mais de um regime dentro da referência |

```python
det.calibrar_piso(modo="ambos", n_blocos=4)
# {'piso_aa': ..., 'piso_analitico': ..., 'razao_aa': 0.94,
#  'piso_bloco': ..., 'H': 1.3, 'referencia': 'homogenea',
#  'bloco_mais_divergente': '2→3', 'psi_alarme_sugerido': ...}
```

⚠️ **Ordene a referência pelo eixo suspeito** (tempo, lote, turno, tenant) antes
de chamar com `modo="bloco"`. Se a ordem das linhas for arbitrária, o split em
bloco vira um split aleatório caro.

Reprodução completa dos experimentos:

```bash
python scripts/validar_piso.py --rapido                        # ~3 min
python scripts/validar_piso.py --dados seus.parquet --eixo data
```

---

## Os dois datasets, e por que a ordem não é negociável

| | Parte I — Tire & Wheel (sintético) | Parte II — Bosch Production Line |
|---|---|---|
| Papel | **teste unitário** do detector | **teste de campo** |
| Ground truth | conhecido (nós plantamos) | inexistente |
| Pergunta | *meu detector funciona?* | *e quando ninguém me diz a resposta?* |
| Entregável | limiares calibrados + suíte verde | alertas com hipótese de causa |

A ordem é **metodológica**, não didática. A Parte I é o peso padrão.

E o que atravessa entre elas é o **método**, nunca o número:

| transfere sem alteração | é remedido |
|---|---|
| o predicado `p < α/m` **E** `efeito > piso` | o valor do piso |
| as 5 guardas e sua ordem | `k de n`, cooldown |
| as assinaturas diagnósticas | limiar de violação de range |
| a exigência de calibração contra o null | `min_obs`, `psi_alarme` |

```python
from driftkit.state import comparar_regimes
comparar_regimes(cal_v1, cal_v2, nome_v1="sintético", nome_v2="Bosch")
# 'limiar_v1_em_pisos_de_v2': 30.4
# 'veredicto': 'o limiar de sintético é 30× o piso de Bosch. Transportá-lo
#               torna o detector CEGO: só drift catastrófico cruzaria.'
```

---

## Mapa da ementa → onde está no repositório

| Ementa | Onde |
|---|---|
| I. Tipos de drift e impacto | `notebooks/01_sintetico.ipynb`, Bloco I |
| II. Ambiente de monitoramento | `01`, Bloco II-A · `evidently` + `alibi-detect` |
| III. Detecção de data drift (KS, χ², PSI) | `notebooks/02_real.ipynb`, Bloco III |
| IV. Concept drift, janelas, detectores sequenciais | `02`, Bloco IV · `river` |
| V. Mitigação, gatilhos, versionamento | `src/driftkit/cli.py` + `.github/workflows/drift.yml` |
| VI. Q&A e recursos | este README |

**Desvios declarados** (explicados em aula, não escondidos):

- **F1/RMSE → MCC/PR-AUC.** Com prevalência de 0.58%, F1@0.5 não significa nada
  e RMSE não se aplica a classificação. As duas estão implementadas em
  `modelo.metricas_completas()` para você comparar — e ver a acurácia de 0.994
  com o modelo em colapso.
- **Bônus não prometido:** um dataset real (Bosch) além do sintético, e o
  resultado analítico sobre o piso de ruído.

---

## Estrutura

```
pybr2026-drift/
├── setup_colab.py                 # bootstrap idempotente (a Célula 0)
├── src/driftkit/
│   ├── fixtures.py                # a fixture sintética, com drift plantado
│   ├── detectors.py               # PSI, JS, Wasserstein, DriftDetector,
│   │                              #   piso_analitico, n_equivalente
│   ├── modelo.py                  # o modelo + métricas que sobrevivem a 0.58%
│   ├── policy.py                  # as 5 guardas de retreino
│   ├── state.py                   # contrato v1 → v2 · comparar_regimes
│   ├── data.py                    # carga do Bosch: cache local / URL
│   ├── cli.py                     # os exit codes
│   ├── notebook.py                # checkpoint, requer, aposte
│   └── testing.py                 # gabaritos dos exercícios
├── tests/
│   ├── conftest.py
│   ├── test_detectors.py          # TC-1 a TC-4, prova cruzada, null analítico
│   └── test_limitacoes.py         # ⭐ o que o detector NÃO faz, via xfail(strict)
├── scripts/
│   ├── preprocess_bosch.py        # 14 GB → 180 MB (o instrutor roda uma vez)
│   └── gerar_solucoes.py          # gera os *_SOLUCAO.ipynb a partir das tags
├── notebooks/
│   ├── 00_ambiente.ipynb          # rode em casa
│   ├── 01_sintetico.ipynb         # com TODOs
│   ├── 02_real.ipynb              # com TODOs
│   ├── *_SOLUCAO.ipynb            # gerados: make solucoes
│   └── executados/                # com todas as saídas — modo espectador
└── .github/workflows/drift.yml    # o monitor agendado
```

Os notebooks de solução **não são versionados à mão**:

```bash
make solucoes             # regenera a partir das tags exercicio-N
make verificar-solucoes   # falha o CI se estiverem dessincronizados
```

---

## ⭐ O arquivo que você deveria ler primeiro

`tests/test_limitacoes.py`.

A maioria das suítes documenta o que o sistema **faz**. Essa documenta o que ele
**não faz**, usando `@pytest.mark.xfail(strict=True)` — que quebra o CI se
alguém "consertar" uma limitação sem atualizar a documentação.

```python
@pytest.mark.xfail(strict=True, reason=(
    "LIMITAÇÃO ESTRUTURAL, não bug: concept drift é mudança em P(Y|X). "
    "Nenhuma vigilância sobre P(X) tem obrigação matemática de detectá-lo."
))
def test_monitor_de_distribuicao_detecta_concept_drift(detector, df):
    ...


@pytest.mark.xfail(strict=True, reason=(
    "O split ALEATÓRIO distribui a contaminação igualmente entre as metades. "
    "A mistura resultante é permutacionalmente trocável — o A/A dá verde "
    "para qualquer nível de contaminação. Use modo='bloco'."
))
def test_aa_aleatorio_detecta_referencia_contaminada(detector):
    ...
```

---

## Dados

Os derivados vivem em **GitHub Releases**, não no repositório git.

| arquivo | tamanho | conteúdo |
|---|---:|---|
| `meta_bosch.parquet` | 48 MB | 1 linha/peça: eixo temporal + presença por estação |
| `bosch_num_160feats.parquet` | 132 MB | 160 features float32 selecionadas por cobertura |
| `detector_config_referencia_v1.json` | 3 KB | contrato de referência (fallback) |

```python
from driftkit.data import baixar_bosch, carregar_meta_bosch
baixar_bosch()                    # uma vez, ~180 MB
meta = carregar_meta_bosch()      # do cache local
```

Leitura direta por URL também funciona, e com HTTP Range o pyarrow transfere
só as colunas pedidas:

```python
import pandas as pd
from driftkit.data import caminho_ou_url
pd.read_parquet(caminho_ou_url("bosch_num_160feats.parquet"), columns=["Id"])
```

Para reconstruir do zero (exige aceitar as regras da competição na Kaggle):

```bash
kaggle competitions download -c bosch-production-line-performance
make dados
```

---

## Durante o tutorial

**Cartões de sinalização** — pegue um na entrada:

| 🟢 | 🟡 | 🔴 |
|---|---|---|
| rodou, acompanhando | rodou mas não entendi | travado, preciso de ajuda |

**Cinco checkpoints.** Cada um é uma célula que imprime ✅ ou ❌. Se der ❌,
levante o 🔴 — não avance sozinho.

**Terminou antes?** Escolha um:

| | desafio | bloco |
|---|---|---|
| ⭐ | Baixe `psi_limiar` para 0.001 e capture o TC-1 ficando vermelho | II-A |
| ⭐ | Rode `calibrar_piso` nos dois modos. Explique a diferença em duas frases | III |
| ⭐⭐ | Qual o **menor offset** (Nm) que a suíte ainda detecta? | II-A |
| ⭐⭐ | Troque KS por Wasserstein. O TC-3 muda de resposta? | III |
| ⭐⭐ | Suba `n_blocos` para 8. O `bloco_mais_divergente` aponta sempre a mesma transição? | III |
| ⭐⭐⭐ | Faça CEL-02 **e** CEL-03 derivarem juntas — o monitor ainda distingue? | II-A |
| ⭐⭐⭐ | Implemente a **guarda 5b** (desgaste de ferramenta) e escreva o teste | V |
| 🏆 | Construa o eixo temporal do Bosch a partir do `Id` e documente o que quebra | II-B |

---

## Segunda-feira, 15 minutos

> **Divida sua referência pela metade no tempo — não aleatoriamente.**
> Compare o PSI com $$\chi^2_{0{,}95}(B-1)\cdot(2/n)$$.
>
> Se der mais que o dobro, sua janela de referência contém mais de um regime —
> e **nenhum limiar te salva disso**.

```python
from driftkit.detectors import DriftDetector

ref = sua_referencia.sort_values("data")        # ⚠️ ORDENE PRIMEIRO
det = DriftDetector.from_reference(ref)
print(det.calibrar_piso(modo="ambos", n_blocos=4))
```

Leia o `H`. Se ele passar de 5, o problema não é o seu limiar — é o seu
**gabarito**. Encurte ou segmente a referência antes de calibrar qualquer coisa.

---

## Referências

- Gama, J. et al. (2014). *A survey on concept drift adaptation.* ACM Computing Surveys.
- Sculley, D. et al. (2015). *Hidden technical debt in machine learning systems.* NeurIPS.
- Widmer, G. & Kubat, M. (1996). *Learning in the presence of concept drift.* Machine Learning.
- Bifet, A. & Gavaldà, R. (2007). *Learning from time-changing data with adaptive windowing.* SDM.
- Yurdakul, B. (2018). *Statistical Properties of Population Stability Index.* Western Michigan University.
- Bosch Production Line Performance — Kaggle, 2016.

**Ferramentas de mercado** (tudo aqui existe em biblioteca; implementamos à mão
para entender o instrumento antes de comprá-lo):
[Evidently](https://github.com/evidentlyai/evidently) ·
[alibi-detect](https://github.com/SeldonIO/alibi-detect) ·
[NannyML](https://github.com/NannyML/nannyml) ·
[river](https://github.com/online-ml/river) ·
[Deepchecks](https://github.com/deepchecks/deepchecks)

Nenhuma delas sabe qual é o **seu** piso de ruído. O default é sempre `0.1` ou
`p < 0.05`. Use a biblioteca — mas calibre.

---

MIT. Slides e gravação: em breve.
