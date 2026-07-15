# Convenience targets over the markovmps package. The package itself never
# requires make; every target names the plain command it runs.

PYTHON ?= python3

install:
	$(PYTHON) -m pip install -e .[dev]

test:
	$(PYTHON) -m pytest tests -q

run:
	$(PYTHON) run.py --problem all

hero:
	$(PYTHON) tools/make_hero.py

.PHONY: install test run hero
