# Contributing

- **Lint / format:** `ruff check src tests scripts app` and `ruff format --check src tests scripts app` (or `ruff format` to write).
- **Types:** `mypy src/transformer`
- **Tests:** `pytest -q -m "not gpu"` (add `-k` to filter; use a real `python3` on `PATH` if your `python` is not CPython).
- **Pre-commit (optional):** `pre-commit install` then `pre-commit run -a`

Open a PR with a short description of the change and any new scripts or config fields.
