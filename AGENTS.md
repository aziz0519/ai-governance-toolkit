# AGENTS.md

## Setup / verification

- Python is pinned to 3.14 (`.python-version`; both workflows use 3.14; root `requires-python = ">=3.14"`).
- Install the tested subpackages **editable**; `pip install -e .` at the root exposes no importable modules (root `pyproject.toml` declares no packages and no dependencies):
  ```
  python -m pip install fairlearn scikit-learn pandas numpy huggingface_hub pytest pytest-cov -e ./src/model_card_generator -e ./src/bias_detection -e ./src/audit_trail
  ```
  Non-editable installs silently test site-packages copies instead of `src/`. `requirements.txt` is not consumed by CI; `.github/workflows/governance.yml` has the effective install command and verifies imports resolve under `src/`.
- Verify with the CI command: `python -m pytest tests/test_governance.py -v --tb=short`. Use `-k <expr>` to focus a single test/class.
- `pytest.ini` always adds `--cov=src --cov-report=term-missing --cov-report=html`; use `--no-cov` for quick runs. With editable installs, coverage reflects `src/` (about 31% total: only 3 of 6 packages are tested); with non-editable installs it reports 0% despite passing tests. No `--cov-fail-under` gate is configured.

## Package / import layout

- `src/` contains six independent setuptools packages, each with its own `pyproject.toml` and `package-dir = "."`; installing `./src/<name>` gives top-level import `<name>`.
- Tests import installed top-level modules (`from model_card_generator...`), while `examples/*.py` import `src.*`; those imports need the repo root on `sys.path` (e.g. Jupyter or `python -m` started at root). Plain `python examples/foo.py` does not add the root. Changing one import style requires updating the CI install step.
- CI installs/tests only `model_card_generator`, `bias_detection`, and `audit_trail`. `human_in_the_loop` imports `audit_trail` but is not installed by the workflow; `datasheet_generator` and `llm_bias_test` are also outside governance tests.

## Testing / code gotchas

- `tests/test_governance.py` seeds numpy with 42 and trains a `GradientBoostingClassifier`; bias assertions (`disparate_impact_ratio >= 0.8`, demographic parity diff `<= 0.15`) depend on that synthetic data fixture.
- `data/sample_loan_data.csv` is not used by the tests.
- Edit `src/<module>/<module>.py`; `src/*/build/lib/...` are generated copies. Root `build/`, `dist/`, `htmlcov/`, and `*.egg-info/` are ignored artifacts.

## CI / release

- PRs run `.github/workflows/governance.yml`. Tags `v*` and manual dispatch run `.github/workflows/build.yml` (`python -m build`, upload `dist/*`); it builds the root project only, not per-submodule wheels.
