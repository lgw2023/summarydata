# Repository Guidelines

## Project Structure & Module Organization

This Python data pipeline supports sports and health QA. Reusable code lives in `src/`: `data_loader/` reads inputs, `generators/` calls model endpoints, and `data_clean/` parses and aggregates personal-health records. Supporting logic is in `analysis/`, `config/`, and `utils/`. Workflows are in `scripts/`: `run_pipeline.py` generates candidates, `visualize_data_app.py` provides Streamlit review, and `run_trainningdata.py` exports SFT/KTO data. YAML definitions belong in `configs/`; prompt templates in `prompts/`. Generated JSONL files are grouped under `data/<experiment>/{intermediate,processed}/`. Tests sit beside data-cleaning modules as `test_*.py` files.

## Setup, Test, and Development Commands

Create an isolated environment before running scripts:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

Run a small generation smoke test with:

```bash
python scripts/run_pipeline.py --config configs/summary_train_v36.yaml \
  --raw-data summary_train_v36.xlsx --stage generate --max-rows 3
```

Launch manual review with `streamlit run scripts/visualize_data_app.py`. Export scored examples with `python scripts/run_trainningdata.py --raw-data <dataset>.xlsx`. The bundled KTO judge is historical; do not treat it as the current production scorer without confirming the intended evaluation workflow.

## Coding Style & Naming Conventions

Use four-space indentation and standard PEP 8 naming: `snake_case` for modules, functions, and variables; `PascalCase` for classes; `UPPER_SNAKE_CASE` for constants. Add type hints to new reusable functions and keep CLI work inside `main()` guarded by `if __name__ == "__main__":`. Prefer shared helpers in `src/` over duplicating logic in scripts. No formatter or linter is enforced, so keep imports grouped and avoid unrelated reformatting.

## Testing Guidelines

Use pytest. Name files `test_<behavior>.py` and tests `test_<expected_behavior>`. Place focused unit tests near the corresponding `src/data_clean` code. Run `pytest -q` before submitting; for targeted work, use `pytest -q src/data_clean/test_aggregate_time.py`. There is no configured coverage threshold, but bug fixes should include a regression test.

## Commit & Pull Request Guidelines

Recent commits use short, imperative summaries such as `Improve data aggregation and output formatting`; follow that style and keep each commit scoped. Pull requests should explain the changed pipeline stage, list validation commands, identify affected configs and output schemas, and link an issue when available. Include screenshots for Streamlit UI changes and small before/after schema examples for data-format changes—never real health text.

## Security & Data Handling

Keep credentials in `.env`; never print or commit API keys. Treat root-level workbooks and generated health records as sensitive. Use synthetic or redacted fixtures in tests and reviews, and avoid committing large generated artifacts unless they are intentional, documented deliverables.
