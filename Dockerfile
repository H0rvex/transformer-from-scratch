# CUDA runtime for GPU training/benchmarks; CI uses CPU-only pytest on host.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY tests ./tests
COPY scripts ./scripts
COPY configs ./configs
COPY app ./app
COPY docs ./docs

RUN python3.10 -m pip install --no-cache-dir -U pip setuptools wheel \
    && pip install --no-cache-dir -e ".[dev]"

CMD ["python3.10", "-m", "pytest", "-q", "-m", "not gpu", "tests/"]
