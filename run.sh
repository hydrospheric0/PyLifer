#!/usr/bin/env bash
# run.sh — thin venv wrapper for PyLifer.py
# Usage: ./run.sh [PyLifer.py arguments...]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
PYTHON_BIN="$VENV_DIR/bin/python"
PIP_BIN="$VENV_DIR/bin/pip"
REQ_FILE="$SCRIPT_DIR/requirements.txt"
REQ_STAMP="$VENV_DIR/.requirements.sha256"

# Create venv if missing
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "[run.sh] Creating virtual environment ..."
    python3 -m venv "$VENV_DIR"
fi

# Reinstall only when requirements.txt changes
current_hash="$(sha256sum "$REQ_FILE" | awk '{print $1}')"
saved_hash="$(cat "$REQ_STAMP" 2>/dev/null || true)"
if [[ "$current_hash" != "$saved_hash" ]]; then
    echo "[run.sh] Installing dependencies ..."
    "$PIP_BIN" install --upgrade pip -q
    "$PIP_BIN" install -r "$REQ_FILE"
    echo "$current_hash" > "$REQ_STAMP"
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/PyLifer.py" "$@"
