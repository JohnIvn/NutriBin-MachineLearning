#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
APP_DIR="$REPO_ROOT/yolo/scripts"
APP_FILE="$APP_DIR/desktop_app.py"
VENV_PYTHON="$REPO_ROOT/venv/bin/python"

export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"

if [[ ! -f "$APP_FILE" ]]; then
    echo "Desktop app not found: $APP_FILE" >&2
    exit 1
fi

if [[ -x "$VENV_PYTHON" ]]; then
    PYTHON="$VENV_PYTHON"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
else
    echo "python3 not found" >&2
    exit 1
fi

cd "$APP_DIR"
exec "$PYTHON" "$APP_FILE"