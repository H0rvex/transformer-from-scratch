#!/usr/bin/env bash
# One ordered pass: train (optional) → metrics/docs tables → ONNX → inference/kernel benches → sample generations.
# Run from repo root:  bash scripts/portfolio_pipeline.sh
# Env: PYTHON=python3  SKIP_TRAIN=1  PORTFOLIO_CLF_DIR=...  PORTFOLIO_GPT_DIR=...  HYDRA_GPT_EXTRA="train.amp_dtype=fp16 train.compile=false"

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_CLF="${SKIP_CLF:-0}"
SKIP_GPT="${SKIP_GPT:-0}"
PORTFOLIO_CLF_DIR="${PORTFOLIO_CLF_DIR:-outputs/portfolio/clf}"
PORTFOLIO_GPT_DIR="${PORTFOLIO_GPT_DIR:-outputs/portfolio/gpt}"
HYDRA_CLF_EXTRA="${HYDRA_CLF_EXTRA:-}"
HYDRA_GPT_EXTRA="${HYDRA_GPT_EXTRA:-}"

echo "== Portfolio pipeline (repo: $ROOT) =="

if [[ "$SKIP_TRAIN" == "1" ]]; then
  SKIP_CLF=1
  SKIP_GPT=1
  echo ">> SKIP_TRAIN=1 — using existing checkpoints under outputs/portfolio/"
fi

if [[ "$SKIP_CLF" != "1" ]]; then
  echo ">> Train IMDB classifier → $PORTFOLIO_CLF_DIR"
  # shellcheck disable=SC2086
  "$PYTHON" scripts/train_classifier.py hydra.run.dir="$PORTFOLIO_CLF_DIR" $HYDRA_CLF_EXTRA
else
  echo ">> SKIP_CLF=1 — reusing $PORTFOLIO_CLF_DIR"
fi

if [[ "$SKIP_GPT" != "1" ]]; then
  echo ">> Train GPT (TinyShakespeare) → $PORTFOLIO_GPT_DIR"
  # shellcheck disable=SC2086
  "$PYTHON" scripts/train_gpt.py hydra.run.dir="$PORTFOLIO_GPT_DIR" $HYDRA_GPT_EXTRA
else
  echo ">> SKIP_GPT=1 — reusing $PORTFOLIO_GPT_DIR"
fi

if [[ ! -f "$PORTFOLIO_GPT_DIR/best_model.pt" ]]; then
  echo "ERROR: missing $PORTFOLIO_GPT_DIR/best_model.pt (train first or set PORTFOLIO_GPT_DIR)" >&2
  exit 1
fi

echo ">> Attention micro-bench → docs/BENCHMARKS.md"
"$PYTHON" scripts/benchmark.py --repeats 50 --warmup 15

echo ">> LM train-step sweep (meaningful on CUDA) → docs/BENCHMARKS_TRAIN_STEP.md"
"$PYTHON" scripts/benchmark.py --train-step --batch 8 --seq 128 --d-model 256 --heads 8 --layers 2 --vocab 512 --repeats 30 --warmup 10

echo ">> Synthetic ablations → docs/ABLATIONS.md"
"$PYTHON" scripts/ablate.py

echo ">> Attention heatmaps (synthetic encoder block) → docs/assets/attention/"
"$PYTHON" scripts/viz_attention.py

echo ">> ONNX export + numeric check → docs/assets/onnx/"
"$PYTHON" scripts/export_onnx.py

echo ">> Inference benches (writes clf + gpt tables; avoid overwriting with two outs)"
"$PYTHON" scripts/bench_inference.py --kind clf --out docs/INFERENCE_BENCHMARKS.md
"$PYTHON" scripts/bench_inference.py --kind gpt --out docs/INFERENCE_BENCHMARKS_GPT.md

echo ">> Kernel micro-bench (Triton path when CUDA available) → docs/KERNELS.md"
"$PYTHON" scripts/bench_kernels.py

if command -v trtexec >/dev/null 2>&1; then
  echo ">> TensorRT engines (trtexec found)"
  mkdir -p docs/assets/trt
  "$PYTHON" scripts/build_trt_engine.py --onnx docs/assets/onnx/classifier.onnx --out docs/assets/trt/classifier_fp16.plan --fp16 || true
  "$PYTHON" scripts/build_trt_engine.py --onnx docs/assets/onnx/gpt.onnx --out docs/assets/trt/gpt_fp16.plan --fp16 || true
else
  echo ">> Skipping TensorRT build (trtexec not on PATH)"
fi

echo ">> Sample generations → docs/assets/generations.md"
"$PYTHON" scripts/generate.py --checkpoint "$PORTFOLIO_GPT_DIR/best_model.pt" --out docs/assets/generations.md

echo "== Done. Highlights =="
echo "  Classifier metrics/plots: $PORTFOLIO_CLF_DIR/ (metrics.csv, Hydra logs)"
echo "  GPT checkpoint:           $PORTFOLIO_GPT_DIR/best_model.pt"
echo "  Tables: docs/BENCHMARKS.md, docs/BENCHMARKS_TRAIN_STEP.md, docs/ABLATIONS.md"
echo "  Inference: docs/INFERENCE_BENCHMARKS.md, docs/INFERENCE_BENCHMARKS_GPT.md"
echo "  Generations: docs/assets/generations.md"
