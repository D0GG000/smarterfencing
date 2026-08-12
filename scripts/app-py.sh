#!/usr/bin/env bash
# Run a Python script with the same interpreter as gunicorn (mmpose-env).
set -euo pipefail
exec /opt/conda/envs/mmpose-env/bin/python3 "$@"
