# Contributing Guidelines

Thank you for considering a contribution to **llm-eval-guardrails**! We welcome enhancements, bug fixes, and documentation improvements.

## Getting Started
1. Clone the repository and create a virtual environment: `make setup`
2. Install pre-commit hooks: `pre-commit install` (already handled by `make setup`)
3. Run the demo pipeline to confirm everything works locally: `make ingest && make eval`

## Development Workflow
- Branch from `main`.
- Keep pull requests focused and include tests for new logic.
- Run the full quality gate before opening a PR:
  ```bash
  make lint
  make typecheck
  make test
  ```
- Ensure `reports/demo_run` artifacts regenerate successfully if your change touches evaluation or reporting.

## Testing
- Unit tests live in `tests/`.
- Mock providers ensure tests run without external API calls—please keep them fast and deterministic.
- Add regression fixtures or YAML prompts when fixing adversarial gaps.

## Coding Standards
- Python 3.11, type hints everywhere, docstrings for public functions/classes.
- Formatting via Black, linting via Ruff, typing via MyPy strict mode.
- Use Rich logging and Typer’s help text conventions for CLI changes.

## Documentation
- Update `README.md` for user-facing changes.
- Capture new CLI flags or configuration fields.
- Keep the sample report and baseline metrics current when you adjust evaluations.

## Reporting Issues
Please include:
- Reproduction steps
- Expected vs. actual behavior
- Environment information (OS, Python version, dependencies)
- Relevant logs or report snippets

## Code of Conduct
We follow the [Contributor Covenant](https://www.contributor-covenant.org/). Be respectful, collaborative, and helpful.

Happy hacking!
