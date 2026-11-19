# Testing

Use the repository-local Conda environment described in
[Development](development.md).

## Canonical Checks

```bash
./.conda/bin/python -m pytest -q
./.conda/bin/ruff check src tests
./.conda/bin/mkdocs build --strict
./.conda/bin/python -m build
```

The test suite checks the public import and its version against the installed
distribution metadata. The strict site build checks navigation and generates
the Python reference from the package.

For packaging changes, inspect the source distribution and wheel, then install
the wheel and import `ao_core` in an isolated environment outside the source
checkout. Check that the wheel includes `py.typed` and that the source
distribution includes the project docs and bootstrap script.
