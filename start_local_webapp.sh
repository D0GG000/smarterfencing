#!/usr/bin/env bash
# Start SmarterFencing locally (same /demo UI + pipeline as production v262).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export WORKSPACE_ROOT="${WORKSPACE_ROOT:-$ROOT/local_workspace}"
export LOCAL_WEBAPP_PORT="${LOCAL_WEBAPP_PORT:-5000}"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    PYTHON_BIN="python"
  fi
fi

exec "$PYTHON_BIN" run_local_webapp.py
