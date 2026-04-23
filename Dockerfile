# CUDA runtime for GPU training/benchmarks; install torch from CUDA wheel index (CPU-only wheels break GPU).
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip git \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY tests ./tests
COPY scripts ./scripts
COPY configs ./configs
COPY app ./app
COPY docs ./docs

RUN python3.10 -m pip install --no-cache-dir -U pip setuptools wheel \
    && pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir -e ".[dev]"

RUN chown -R appuser:appgroup /app

USER appuser

CMD ["python3.10", "-m", "pytest", "-q", "-m", "not gpu", "tests/"]
