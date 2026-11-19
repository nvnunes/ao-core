#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda is required to bootstrap AO Core." >&2
  exit 1
fi

env_prefix="$repo_root/.conda"
if [ ! -d "$env_prefix/conda-meta" ]; then
  conda create -y --prefix "$env_prefix" python=3.12 pip
fi

"$env_prefix/bin/python" -m pip install -e ".[dev,docs]"
git config core.hooksPath .githooks

"$env_prefix/bin/python" -m pytest -q
"$env_prefix/bin/ruff" check src tests
"$env_prefix/bin/mkdocs" build --strict
"$env_prefix/bin/python" -m build
