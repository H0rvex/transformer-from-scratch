# Portfolio data pipeline

This doc matches the “run it, populate real numbers” intent in the portfolio plan (Phase 6): training produces **Hydra `metrics.csv` + checkpoints**; scripts regenerate **benchmark and doc tables** under `docs/`.

## Is a local GPU enough?

**For portfolio credibility (code + systems story): yes.** Completing training, ONNX export, kernel and inference benches, and committing real artifacts shows end-to-end ownership. Reviewers care that the stack works and that you can interpret numbers.

**For the plan’s “NVIDIA-shaped” headline tables:** the plan explicitly targets **Ampere-class** runs (bf16, `torch.compile`, TRT on a well-supported stack) and short wall-clock. A **GTX 1060** can still run the same pipeline; use `train.amp_dtype=fp16` and often `train.compile=false` if bf16/compile misbehave. Throughput and VRAM rows will differ from an L4/A10—that is honest, not weak.

**Renting an L4/A10** is optional polish when you want README/model-card numbers that match common datacenter GPUs and faster iteration.

## One-shot script (recommended)

From the repo root, after `pip install -e ".[dev]"` (and optional `[app]`, GPU extras for ORT CUDA):

```bash
bash scripts/portfolio_pipeline.sh
```

Checkpoints and CSVs land under fixed Hydra dirs so downstream steps stay predictable:

- Classifier: `outputs/portfolio/clf/` (`metrics.csv`, `best_model.pt`, plots if enabled)
- GPT: `outputs/portfolio/gpt/` (`metrics.csv`, `best_model.pt`)

**Re-run benches only** (skip all training):

```bash
SKIP_TRAIN=1 bash scripts/portfolio_pipeline.sh
```

**Re-run only one leg** (classifier-only or GPT-only):

```bash
SKIP_CLF=1 bash scripts/portfolio_pipeline.sh   # keep existing classifier checkpoint; train GPT + benches
SKIP_GPT=1 bash scripts/portfolio_pipeline.sh   # keep existing GPT checkpoint; train classifier + benches
```

**Older / Pascal GPU** (e.g. GTX 1060, sm_61) for GPT training.

Pascal has **no native bf16** (and weak fp16), and the stock config is tuned for short Ampere runs, so out of the box TinyShakespeare plateaus at loss ≈ 6 / PPL ≈ 400 — the model barely gets past the unigram prior. Two fixes:

1. `train.amp_dtype=fp16` (or `train.amp=false`) — avoid the bf16 path.
2. Enough optimizer steps — default is only `25 × 28 ≈ 700` steps, and with `warmup_steps=200` + cosine decay, effective learning is under 500 steps. Bump to ~5k steps.

Recommended override on a 1060 (≈60–75 min; the classifier leg is unchanged):

```bash
SKIP_CLF=1 \
HYDRA_GPT_EXTRA='train.amp_dtype=fp16 train.epochs=200 train.warmup_steps=500 train.lr=3e-4 model.dropout=0.1' \
bash scripts/portfolio_pipeline.sh
```

**Target numbers (BPE-2048 TinyShakespeare, small 6-layer / d=384 GPT).** The tokenizer is **byte-level BPE**, not char-level, so loss/PPL are **per BPE token** and are not directly comparable to nanoGPT char-level numbers.

| Regime | Val loss (nats) | Val PPL |
|---|---:|---:|
| Uniform over `vocab=2048` | 7.62 | 2048 |
| Unigram / frequency prior (what the broken 25-epoch run hits) | ≈ 5.8–6.0 | ≈ 300–400 |
| **Portfolio target for this repo** | **4.0–5.0** | **55–150** |
| Stretch (risks overfit on ~300k BPE tokens) | ~3.5 | ~30 |

Commit the resulting `outputs/portfolio/gpt/metrics.csv` and `docs/assets/generations.md` as your Phase 6 GPT artifacts, and populate the model card / README rows from this CSV.

**Custom Python:**

```bash
PYTHON=python3.11 bash scripts/portfolio_pipeline.sh
```

## Same steps as explicit commands

1. **Train** (Hydra; metrics + checkpoint in run dir):

   ```bash
   python scripts/train_classifier.py hydra.run.dir=outputs/portfolio/clf
   python scripts/train_gpt.py hydra.run.dir=outputs/portfolio/gpt
   ```

2. **Attention table** → `docs/BENCHMARKS.md`:

   ```bash
   python scripts/benchmark.py --repeats 50 --warmup 15
   ```

3. **Training-step sweep** (CUDA exercises fp16/bf16 × compile) → `docs/BENCHMARKS_TRAIN_STEP.md`:

   ```bash
   python scripts/benchmark.py --train-step --batch 8 --seq 128 --d-model 256 --heads 8 --layers 2 --vocab 512 --repeats 30 --warmup 10
   ```

   To align the attention table with **GPT small** width for README copy-paste, override `--batch/--seq/--d-model/--heads` to match `configs/model/gpt_small.yaml` and your batch size.

4. **Ablations** → `docs/ABLATIONS.md`:

   ```bash
   python scripts/ablate.py
   ```

5. **Attention viz** (synthetic encoder) → `docs/assets/attention/`:

   ```bash
   python scripts/viz_attention.py
   ```

6. **ONNX** → `docs/assets/onnx/`:

   ```bash
   python scripts/export_onnx.py
   ```

7. **Inference benches** (after ONNX; uses toy arch matching export):

   ```bash
   python scripts/bench_inference.py --kind clf --out docs/INFERENCE_BENCHMARKS.md
   python scripts/bench_inference.py --kind gpt --out docs/INFERENCE_BENCHMARKS_GPT.md
   ```

   Merge or copy rows into a single narrative doc if you prefer one file.

8. **Triton RMSNorm bench** → `docs/KERNELS.md`:

   ```bash
   python scripts/bench_kernels.py
   ```

9. **TensorRT** (optional; requires `trtexec` on `PATH`):

   ```bash
   python scripts/build_trt_engine.py --onnx docs/assets/onnx/classifier.onnx --out docs/assets/trt/classifier_fp16.plan --fp16
   python scripts/build_trt_engine.py --onnx docs/assets/onnx/gpt.onnx --out docs/assets/trt/gpt_fp16.plan --fp16
   ```

10. **Generations** (trained GPT checkpoint):

    ```bash
    python scripts/generate.py --checkpoint outputs/portfolio/gpt/best_model.pt --out docs/assets/generations.md
    ```

## Makefile

- `make train-clf` / `make train-gpt` — Hydra default run dirs (timestamped under `outputs/clf/` and `outputs/gpt/`).
- `make bench` — attention + kernel + classifier inference bench only (see `Makefile`).
- `make portfolio-pipeline` — runs `scripts/portfolio_pipeline.sh`.

## CI parity (local)

```bash
ruff check src tests scripts app && ruff format --check src tests scripts app
mypy src/transformer
python -m pytest -q -m "not gpu"
```

## What to commit for Phase 6

Per plan: populated `docs/*.md` tables, `docs/assets/*` (plots, ONNX, generations), and README/model-card numbers sourced from **your** `metrics.csv` and benchmark outputs—whether from a rented GPU or your local card.
