.PHONY: help venv install clean

help:
	@echo "Available commands:"
	@echo "  make venv      # create a Python virtual environment"
	@echo "  make install   # install pinned dependencies into .venv"
	@echo "  make clean     # remove generated environment and caches"

venv:
	python3 -m venv .venv
	. .venv/bin/activate && python -m pip install --upgrade pip setuptools wheel

install: venv
	. .venv/bin/activate && python -m pip install --no-cache-dir -r requirements.txt

clean:
	rm -rf .venv __pycache__
