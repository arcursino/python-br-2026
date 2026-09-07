# =============================================================================
#  Makefile — python-br-2026
#
#  Para quem trouxe o próprio notebook. Quem estiver no laboratório usa Colab
#  e não precisa de nada disto.
#
#      make setup     instala tudo com uv (30s)
#      make test      suíte rápida
#      make ci        o que o CI roda
#      make demo      o tutorial inteiro em 2 minutos, sem notebook
# =============================================================================
.DEFAULT_GOAL := ajuda
.PHONY: ajuda setup setup-pip test test-tudo ci lint fmt lab demo calibrar simular dados limpar

UV := $(shell command -v uv 2>/dev/null)
RUN := $(if $(UV),uv run,python -m)

ajuda:  ## mostra esta ajuda
	@echo ""
	@echo "  python-br-2026 — Rumo ao Desconhecido: Tratando Drift em ML"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
	@echo ""

# -----------------------------------------------------------------------------
setup:  ## instala o ambiente com uv (recomendado)
ifndef UV
	@echo "uv não encontrado. Instale com:"
	@echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
	@echo "Ou use: make setup-pip"
	@exit 1
endif
	uv sync --extra dev --extra mercado
	@echo ""
	@echo "✅ pronto. Valide com: make test"

setup-pip:  ## instala com pip + venv (alternativa sem uv)
	python -m venv .venv
	. .venv/bin/activate && pip install -U pip && pip install -e ".[dev,mercado]"
	@echo "✅ pronto. Ative com: source .venv/bin/activate"

# -----------------------------------------------------------------------------
test:  ## suíte rápida (o que você roda a cada mudança)
	$(RUN) pytest -q -m "not lento and not requer_bosch and not mercado"

test-tudo:  ## suíte completa, incluindo testes lentos
	$(RUN) pytest -v --cov=driftkit --cov-report=term-missing

ci: lint test  ## o portão de qualidade: lint + suíte
	@echo "✅ CI local verde"

lint:  ## ruff check
	$(RUN) ruff check src tests scripts

fmt:  ## ruff format + fix
	$(RUN) ruff format src tests scripts
	$(RUN) ruff check --fix src tests scripts

# -----------------------------------------------------------------------------
lab:  ## abre o Jupyter Lab
	$(RUN) jupyter lab notebooks/

calibrar:  ## gera o contrato v1 a partir da fixture sintética (Parte I sem notebook)
	$(RUN) driftkit calibrar --seed 7 --saida data/detector_config.json

simular:  ## roda o pipeline dia a dia — o clímax do Bloco V
	$(RUN) driftkit simular --de 30 --ate 130 --passo 5

demo: calibrar simular  ## o argumento do tutorial inteiro, em 2 minutos
	@echo ""
	@echo "  Leia a coluna 'exit'. O 20 no TC-3 é o tutorial inteiro em um número."

# -----------------------------------------------------------------------------
dados:  ## pré-processa o Bosch (instrutor, uma vez; exige os CSVs da Kaggle)
	$(RUN) python scripts/preprocess_bosch.py --entrada $${BOSCH_DIR:-~/kaggle/bosch} --saida data/

dados-teste:  ## testa o script de pré-processamento com 50k linhas (2 min)
	$(RUN) python scripts/preprocess_bosch.py --entrada $${BOSCH_DIR:-~/kaggle/bosch} \
		--saida /tmp/bosch-teste --amostra 50000

release:  ## publica os dados derivados no GitHub Releases
	gh release create dados-v1 \
		data/meta_bosch.parquet \
		data/bosch_num_160feats.parquet \
		data/detector_config_referencia_v1.json \
		--title "Dados derivados — PyBR 2026" \
		--notes "Bosch pré-processado. Ver scripts/preprocess_bosch.py."

limpar:  ## remove artefatos gerados (preserva os dados baixados)
	rm -rf .pytest_cache .ruff_cache **/__pycache__ .coverage htmlcov models/
	rm -f data/detector_config.json data/historico_decisoes.jsonl
