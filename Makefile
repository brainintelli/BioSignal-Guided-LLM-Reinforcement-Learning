PYTHON ?= python
.PHONY: smoke reproduce test figures

smoke:
	$(PYTHON) -m src.reproduce smoke

reproduce:
	$(PYTHON) -m src.reproduce all

test:
	$(PYTHON) -m unittest discover -s tests -v

figures:
	$(PYTHON) figures/draw.py --all --output runs/figures
