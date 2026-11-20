# AGENTS.md

## Setup / verification

- Python is pinned to 3.14 (`.python-version`; both workflows use 3.14; root `requires-python = ">=3.14"`).
- Install the tested subpackages explicitly; `pip install -e .` at the root exposes no importable modules (root `pyproject.toml` declares no packages and no dependencies):
  ```
  python -m pip install fairlearn scikit-learn pandas numpy huggingface_hub pytest pytest-cov ./src/model_card_generator ./src/bias_detection ./src/audit_trail
  ```
  `requirements.txt` is not consumed by CI; `.github/workflows/governance.yml` has the effective install command.
- Verify with the CI command: `python -m pytest tests/test_governance.py -v --tb=short`. Use `-k <expr>` to focus a single test/class.
- `pytest.ini` always adds `--cov=src --cov-report=term-missing --cov-report=html`; use `--no-cov` for quick runs. Coverage currently reports ~0% for `src/` even when tests pass, because tests import installed wheels rather than `src/` sources.

## Package / import layout

- `src/` contains six independent setuptools packages, each with its own `pyproject.toml` and `package-dir = "."`; installing `./src/<name>` gives top-level import `<name>`.
- Tests import installed top-level modules (`from model_card_generator...`), while `examples/*.py` import `src.*` and work from the repo root without installing. Changing one import style requires updating the CI install step.
- CI installs/tests only `model_card_generator`, `bias_detection`, and `audit_trail`. `human_in_the_loop` imports `audit_trail` but is not installed by the workflow; `datasheet_generator` and `llm_bias_test` are also outside governance tests.

## Testing / code gotchas

- `tests/test_governance.py` seeds numpy with 42 and trains a `GradientBoostingClassifier`; bias assertions (`disparate_impact_ratio >= 0.8`, demographic parity diff `<= 0.15`) depend on that synthetic data fixture.
- `data/sample_loan_data.csv` is not used by the tests.
- Edit `src/<module>/<module>.py`; `src/*/build/lib/...` are generated copies. Root `build/`, `dist/`, `htmlcov/`, and `*.egg-info/` are ignored artifacts.

## CI / release

- PRs run `.github/workflows/governance.yml`. Tags `v*` and manual dispatch run `.github/workflows/build.yml` (`python -m build`, upload `dist/*`); it builds the root project only, not per-submodule wheels.
