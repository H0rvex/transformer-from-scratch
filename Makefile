PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

.PHONY: install dev lint test bench train-clf train-gpt docker-build docker-cpu ablate

install:
	$(PIP) install -U pip
	$(PIP) install -e ".[dev]"

dev: install

lint:
	ruff check src tests scripts app
	ruff format --check src tests scripts app
	mypy src/transformer

test:
	$(PYTHON) -m pytest -q -m "not gpu" tests/

bench:
	$(PYTHON) scripts/benchmark.py
	$(PYTHON) scripts/bench_kernels.py
	$(PYTHON) scripts/bench_inference.py --kind clf

ablate:
	$(PYTHON) scripts/ablate.py

train-clf:
	$(PYTHON) scripts/train_classifier.py

train-gpt:
	$(PYTHON) scripts/train_gpt.py

docker-build:
	docker build -t transformer-fs:cuda -f Dockerfile .

docker-cpu:
	docker build -t transformer-fs:cpu -f Dockerfile.cpu .
