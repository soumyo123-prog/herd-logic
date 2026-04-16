.PHONY: setup install test test-llm test-data test-yfinance test-nsepython test-technical test-technical-unit redis-start redis-stop clean

PYENV_PYTHON := $(HOME)/.pyenv/versions/3.12.12/bin/python
VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

# Create virtualenv and install dependencies
setup: $(VENV)/bin/activate

$(VENV)/bin/activate:
	$(PYENV_PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo ""
	@echo "Virtual environment ready. Activate with:"
	@echo "  source $(VENV)/bin/activate"

# Install/update dependencies (if venv already exists)
install: setup
	$(PIP) install -r requirements.txt

# Test OpenRouter connectivity and structured JSON output
test-llm: setup
	$(PYTHON) tests/test_openrouter.py

# Test BaseDataProvider — caching, retry, graceful degradation
test-data: setup
	$(PYTHON) tests/test_data_provider.py

# Test YFinanceProvider — real network calls against Yahoo Finance
test-yfinance: setup
	$(PYTHON) tests/test_yfinance_provider.py

# Test NSEPythonProvider — real network calls against NSE India
test-nsepython: setup
	$(PYTHON) tests/test_nsepython_provider.py

# Test Technical agent — unit tests only (no network, no LLM)
test-technical-unit: setup
	$(PYTHON) tests/test_indicators.py
	$(PYTHON) tests/test_levels.py
	$(PYTHON) tests/test_features.py
	$(PYTHON) tests/test_base_agent.py

# Full Technical agent suite: unit + pipeline (real yfinance) + end-to-end (LLM)
test-technical: setup test-technical-unit
	$(PYTHON) tests/test_technical_pipeline.py
	$(PYTHON) tests/test_technical_agent.py

# Run every test file in tests/
test: test-llm test-data test-yfinance test-nsepython test-technical

# Redis — local cache backend. Uses Docker so the host stays clean.
redis-start:
	docker run -d --name herd-redis -p 6379:6379 redis:7-alpine
	@echo "Redis running at localhost:6379. Stop with: make redis-stop"

redis-stop:
	docker rm -f herd-redis

# Vault — store/retrieve API keys from macOS Keychain
vault-set: setup
	@read -p "Key name (e.g. OPENROUTER_API_KEY): " key; \
	read -sp "Value: " value; echo; \
	$(PYTHON) -m scripts.vault_setup set $$key $$value

vault-list: setup
	$(PYTHON) -m scripts.vault_setup list

# Remove virtualenv
clean:
	rm -rf $(VENV)
	@echo "Virtual environment removed."
