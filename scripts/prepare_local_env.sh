#!/usr/bin/env bash
set -euo pipefail

python3.11 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -e .

cat <<'MSG'
Local environment ready.

Optional groups, install only when needed:
  pip install -e '.[analysis,monitoring,clients,dev]'

No model was downloaded.
MSG
