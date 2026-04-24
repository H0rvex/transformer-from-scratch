# Contributing

- **Lint / format:** `ruff check src tests scripts app` and `ruff format --check src tests scripts app` (or `ruff format` to write).
- **Types:** `mypy src/transformer`
- **Tests:** `python -m pytest -q -m "not gpu"` (add `-k` to filter). Prefer the same interpreter you installed the project with (e.g. `venv/bin/python -m pytest …`); `python -m` avoids a wrong `pytest` on `PATH` (some setups alias it to a Node shim).
- **Pre-commit (optional):** `pre-commit install` then `pre-commit run -a`

Open a PR with a short description of the change and any new scripts or config fields.
