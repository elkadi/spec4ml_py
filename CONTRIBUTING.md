# Contributing to spec4ml_py

Thanks for your interest!

1. Fork and create a feature branch.
2. Setup dev env:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -e .[dev]
   pre-commit install
   ```
3. Run checks:
   ```bash
   ruff check . && black --check . && mypy spec4ml_py && pytest -q
   ```
4. Open a PR.
