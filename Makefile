.PHONY: help venv install wheelhouse clean

help:
	@echo "Available commands:"
	@echo "  make venv        # create a Python virtual environment"
	@echo "  make install     # install pinned dependencies into .venv"
	@echo "  make wheelhouse  # build wheelhouse for offline installation"
	@echo "  make clean       # remove generated environment and caches"

venv:
	python3 -m venv .venv
	. .venv/bin/activate && python -m pip install --upgrade pip setuptools wheel

install: venv
	. .venv/bin/activate && python -m pip install --no-cache-dir -r requirements.txt

wheelhouse:
	python3 -m venv .wheelenv
	. .wheelenv/bin/activate && python -m pip install --upgrade pip setuptools wheel
	. .wheelenv/bin/activate && python -m pip wheel -r requirements.txt -w wheelhouse

clean:
	rm -rf .venv .wheelenv wheelhouse __pycache__
