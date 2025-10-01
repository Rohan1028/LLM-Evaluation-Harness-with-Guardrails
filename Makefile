.PHONY: setup lint typecheck test ingest eval adv report regress clean

PYTHON ?= python
VENV ?= .venv

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/Scripts/python -m pip install -U pip
	$(VENV)/Scripts/python -m pip install -r requirements.txt
	$(VENV)/Scripts/python -m pre-commit install

lint:
	$(VENV)/Scripts/python -m ruff check .

typecheck:
	$(VENV)/Scripts/python -m mypy src

test:
	$(VENV)/Scripts/python -m pytest -q

ingest:
	$(VENV)/Scripts/python -m evalguard.cli ingest --corpus ./data/corpus --collection demo

eval:
	$(VENV)/Scripts/python -m evalguard.cli run --suite demo --models mock:deterministic --k 4 --out ./reports/demo_run

adv:
	$(VENV)/Scripts/python -m evalguard.cli adversarial --suite all --models mock:deterministic --out ./reports/adv_run

report:
	$(VENV)/Scripts/python -m evalguard.cli report --input ./reports/demo_run/aggregate.json --html ./reports/index.html

regress:
	$(VENV)/Scripts/python -m evalguard.cli regress --current ./reports/demo_run/aggregate.json --baseline ./reports/baseline_metrics.json

clean:
	rm -rf .pytest_cache .mypy_cache htmlcov .coverage* $(VENV)
