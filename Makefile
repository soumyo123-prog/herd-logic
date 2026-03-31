.PHONY: setup install test-llm clean

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
	$(PYTHON) scripts/test_openrouter.py

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
