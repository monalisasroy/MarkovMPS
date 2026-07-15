#!/usr/bin/env bash
# One-command bootstrap: a local virtual environment with the package and
# its test dependencies installed editably. Plain pip works equally well;
# this script only saves retyping.
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .[dev]
echo "environment ready: activate with 'source .venv/bin/activate'"
echo "run the test suite with 'python -m pytest tests -q'"
