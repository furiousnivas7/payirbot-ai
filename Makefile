.PHONY: help venv install wheelhouse test clean

help:
	@echo "Available commands:"
	@echo "  make venv             # create a Python virtual environment"
	@echo "  make install          # install pinned dependencies into .venv"
	@echo "  make install-direct   # install dependencies without proxy settings"
	@echo "  make install-proxy    # install dependencies through a proxy"
	@echo "  make wheelhouse       # build wheelhouse for offline installation"
	@echo "  make test             # validate repository imports and syntax"
	@echo "  make clean            # remove generated environment and caches"

venv:
	python3 -m venv .venv
	. .venv/bin/activate && python -m pip install --upgrade pip setuptools wheel

install: venv
	. .venv/bin/activate && python -m pip install --no-cache-dir -r requirements.txt

install-direct: venv
	. .venv/bin/activate && env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
	  python -m pip install --no-cache-dir -r requirements.txt

install-proxy: venv
	@if [ -z "$(PROXY)" ]; then echo "Please set PROXY=http://host:port"; exit 1; fi
	. .venv/bin/activate && env \
	  http_proxy=$(PROXY) https_proxy=$(PROXY) HTTP_PROXY=$(PROXY) HTTPS_PROXY=$(PROXY) \
	  PIP_TRUSTED_HOST="pypi.org files.pythonhosted.org" \
	  python -m pip install --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt

wheelhouse:
	python3 -m venv .wheelenv
	. .wheelenv/bin/activate && python -m pip install --upgrade pip setuptools wheel
	. .wheelenv/bin/activate && python -m pip wheel -r requirements.txt -w wheelhouse

test: venv
	. .venv/bin/activate && python -m py_compile src/config.py src/data_loader.py src/evaluate.py src/train.py src/prepare_data.py src/convert_to_tflite.py src/dashboard.py

clean:
	rm -rf .venv .wheelenv wheelhouse __pycache__
