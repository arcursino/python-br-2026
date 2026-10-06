# Glossário e mapa do `driftkit`

> Consulta rápida: os conceitos estatísticos e as funções usadas nos notebooks.
> Para a tese completa e os experimentos, veja o [README](../README.md).

## A ideia central em 5 frases

1. Modelo degradado não dá erro: devolve `200 OK` com a previsão errada.
2. Mudança em P(X) (data drift) faz o alarme disparar e muitas vezes não machuca o modelo. Mudança em P(Y|X) (concept drift) sempre machuca e quase não aparece no alarme.
3. Um detector é um instrumento: calibre contra um gabarito (o dataset sintético com drift plantado) antes de usar.
4. Não existe limiar universal. O piso de ruído depende do tamanho da janela, e o `PSI > 0.1` corresponde a ~340 observações.
5. Se a causa é física (sensor) ou de pipeline, bloqueie o retreino (`exit 20`).

---

## Conceitos

| Termo | Definição curta |
|---|---|
| **Data drift** (covariate shift) | Muda P(X); a relação X→Y continua válida |
| **Concept drift** | Muda P(Y\|X); o modelo aprendeu uma regra que deixou de valer |
| **Label shift** | Muda P(Y), por exemplo um novo critério de inspeção |
| **Data quality / training-serving skew** | Schema, unidade ou range quebrados no pipeline |
| **KS** | Maior distância entre duas CDFs empíricas. Com n grande o p-valor vai a zero, por isso olhe a magnitude. |
| **χ²** | Teste para variáveis categóricas |
| **PSI** | Population Stability Index: quanto a distribuição mudou entre referência e produção. Ver [seção PSI](#psi-passo-a-passo) |
| **Jensen–Shannon** | Divergência simétrica e limitada, usada para categóricas |
| **Wasserstein** | Distância na unidade física (Nm, psi): a métrica que a engenharia entende |
| **Bonferroni** | Usa α/m quando há m features. Sem isso, 160 features × α=0,05 dão ~8 falsos alarmes por janela. |
| **Teste A/A** | Comparar a referência com ela mesma. O que alarmar é ruído do instrumento. |
| **Split aleatório × em bloco** | O aleatório mede o ruído. O em bloco (na ordem temporal) detecta uma referência com mais de um regime. |
| **Piso analítico** | E[PSI] ≈ (B−1)(1/n + 1/m). Diminui com n: um limiar fixo fica cego com muito dado. |
| **Índice H** | piso medido em bloco ÷ piso previsto. <2 homogênea · 2–5 suspeita · ≥5 contaminada |
| **PR-AUC** | Área sob a curva precisão–recall. Informativa com prevalência baixa (0,58%). |
| **MCC** | Correlação de Matthews, robusta a desbalanceamento. Depende do limiar τ: reporte o par (τ, MCC). |
| **Cpk / PPM** | Capacidade do processo / peças por milhão fora de especificação |

---
## PSI passo a passo

**PSI = Population Stability Index** (Índice de Estabilidade da População).
Mede o quanto a distribuição de uma feature mudou entre dois momentos.

1. Divida a feature em **faixas** (bins). O padrão é 10 faixas.
2. Para cada faixa, calcule duas porcentagens:
   - **Esperado:** % dos registros da **referência** que caem nessa faixa
   - **Observado:** % dos registros da **produção atual** que caem nessa faixa
3. Some a contribuição de todas as faixas:

$$PSI = \sum_{\text{faixas}} (\text{Observado} - \text{Esperado}) \times \ln\left(\frac{\text{Observado}}{\text{Esperado}}\right)$$

**Leitura:** cada faixa contribui com um valor positivo, e o PSI é zero só se
as duas distribuições forem idênticas. Quanto maior, mais a distribuição mudou.

⚠️ **Mesmo sem drift, o PSI não dá zero** (ruído de amostragem). Esse valor é o
**piso de ruído**, e diminui quando a janela aumenta. O famoso limiar `0.1`
é o piso para janelas de ~340 observações. Por isso o limiar precisa ser
calibrado para o seu tamanho de janela (veja `calibrar_piso()`).

---

## Dataset sintético: linha do tempo

| Dias | Evento | P(X) | Modelo | Ação esperada |
|---|---|---|---|---|
| 1–29 | Referência | | | |
| 30–59 | **TC-1** estável | silêncio | ok | nenhuma (exige zero falsos alarmes) |
| 60–89 | **TC-2** campanha aro R20 | alarme alto | ok | registrar, **não** retreinar |
| 90–119 | **TC-3** sensor descalibrado (CEL-02) | quase calmo | colapsa | **bloquear** e acionar metrologia |
| 120–124 | TC-3' recalibrado | | | |
| ~125–130 | **TC-4** unidade kPa/psi | violação de range | | corrigir o pipeline |

---

## Mapa dos módulos

### `driftkit.fixtures`
| Função | Descrição |
|---|---|
| `gerar_fixture(seed=7)` | 130 dias × 4 células, com drifts plantados |
| `janela(df, dia, w=3)` | Últimos `w` dias até `dia` |
| `gabarito(dia)` | O evento plantado naquele dia |
| `validar_fixture(df)` | Confere se a fixture contém o que promete |
| `cpk`, `ppm_fora_spec` | Métricas de processo |

### `driftkit.detectors`
| Função / método | Descrição |
|---|---|
| `psi`, `js_cat`, `wasserstein` | Métricas de distância entre distribuições |
| `viola_range(cur, ranges)` | Fração fora do range de engenharia, por feature |
| `alarma(p, efeito, alpha_corr, limiar)` | Predicado de alarme: significância **e** magnitude |
| `piso_analitico(n, m, bins)` | Piso de ruído previsto |
| `n_equivalente(limiar, bins)` | Tamanho de janela em que o limiar é puro ruído |
| `DriftDetector.from_reference(...)` | Constrói o detector (imutável) |
| `.report(janela)` → `RelatorioDrift` | `.n_drift`, `.efeito_max`, `.top_features`, `.resumo()` |
| `.calibrar_piso(modo=...)` | Teste A/A: `"aleatorio"`, `"bloco"` ou `"ambos"` |
| `.com(**mudancas)` | Cópia com parâmetros alterados |

### `driftkit.modelo`
| Função | Descrição |
|---|---|
| `treinar(...)` | Treina o modelo base |
| `metricas(...)` | ROC-AUC, PR-AUC, MCC e τ. Devolve `None` quando a métrica não é estimável. |
| `metricas_completas(...)` | Inclui acurácia e F1@0.5, para comparação |
| `melhor_limiar_mcc`, `logloss_por_amostra` | τ ótimo; sinal contínuo para detectores sequenciais |

### `driftkit.policy`
| Item | Descrição |
|---|---|
| `PoliticaRetreino(...).avaliar(...)` → `Decisao` | `.acao`, `.causa`, `.exit_code`, `.guardas` |
| `.diagnosticar(...)` | Causa por assinatura: PIPELINE → NEGÓCIO → HARDWARE → MODELO (residual) |
| `tabela_das_quatro_guardas(d)` | As 4 guardas clássicas aplicadas a uma decisão |

**Ordem das guardas:** 0 mensurável → 1 persistência (k de n) → 3 magnitude > fator × piso → diagnóstico → 2 contexto → **5 causa é ML?** → 1b causa MODELO confirmada → 4 shadow → retreinar.

| exit | Significado |
|:---:|---|
| 0 | sem drift acionável |
| 10 | retreino aprovado |
| 20 | retreino **bloqueado** (causa não-ML) |
| 30 | dados insuficientes (≠ "sem drift") |

### `driftkit.state` / `data` / `notebook`
| Item | Descrição |
|---|---|
| `Contrato`, `salvar_contrato`, `carregar_contrato` | Limiares e guardas em JSON (v1 sintético → v2 Bosch) |
| `comparar_regimes(v1, v2)` | Mostra por que um limiar não se transporta entre regimes |
| `baixar_bosch`, `carregar_*_bosch` | Dados Bosch pré-processados |
| `checkpoint`, `requer`, `aposte`, `resposta`, `cli` | Utilitários de sala de aula. `cli()` devolve o exit code (com `!`, ele se perde). |
