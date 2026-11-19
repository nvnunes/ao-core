# Development

## Bootstrap

From a Git checkout, run the bootstrap script with Conda on `PATH`.

```bash
./scripts/bootstrap.sh
```

The script creates `./.conda` with Python 3.12, installs the package with its
development and documentation dependencies, enables the versioned Git hook,
and runs the checks in [Testing](testing.md).

## Daily Workflow

Refresh the editable install after dependency or package metadata changes.

```bash
./.conda/bin/python -m pip install -e ".[dev,docs]"
```

Serve the documentation locally with `./.conda/bin/mkdocs serve`.

## Git Hook

Bootstrap sets `core.hooksPath` to `.githooks`. The pre-commit hook runs
tests, Ruff and a strict documentation build.
