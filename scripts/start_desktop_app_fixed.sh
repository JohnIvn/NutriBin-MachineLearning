#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/admin/NutriBin-MachineLearning"
APP_SCRIPT="$REPO_ROOT/yolo/scripts/desktop_app.py"

PYTHON_BIN="$REPO_ROOT/venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-/home/admin/.Xauthority}"
export PYTHONUNBUFFERED=1
export NUTRIBIN_AUTO_LIVE=1
export NUTRIBIN_ESP32_ENABLE=1
export NUTRIBIN_ESP32_IP="${NUTRIBIN_ESP32_IP:-192.168.4.1}"
export NUTRIBIN_LIVE_CAMERA="${NUTRIBIN_LIVE_CAMERA:-0}"
export NUTRIBIN_LIVE_MODEL="${NUTRIBIN_LIVE_MODEL:-best.pt}"
export NUTRIBIN_LIVE_MODE="${NUTRIBIN_LIVE_MODE:-pytorch}"
export NUTRIBIN_LIVE_CONF="${NUTRIBIN_LIVE_CONF:-0.25}"
export NUTRIBIN_LIVE_IMGSZ="${NUTRIBIN_LIVE_IMGSZ:-640}"
export NUTRIBIN_LIVE_DEVICE="${NUTRIBIN_LIVE_DEVICE:-auto}"
export NUTRIBIN_STARTUP_DELAY_MS="${NUTRIBIN_STARTUP_DELAY_MS:-2000}"

for _ in $(seq 1 60); do
  if [[ -S /tmp/.X11-unix/X0 && -f "$XAUTHORITY" ]]; then
    break
  fi
  sleep 2
done

cd "$REPO_ROOT"
exec "$PYTHON_BIN" "$APP_SCRIPT" \
  --auto-live-detection \
  --esp32-enable \
  --esp32-ip "$NUTRIBIN_ESP32_IP" \
  --camera "$NUTRIBIN_LIVE_CAMERA" \
  --model "$NUTRIBIN_LIVE_MODEL" \
  --mode "$NUTRIBIN_LIVE_MODE" \
  --conf "$NUTRIBIN_LIVE_CONF" \
  --imgsz "$NUTRIBIN_LIVE_IMGSZ" \
  --device "$NUTRIBIN_LIVE_DEVICE" \
  --startup-delay-ms "$NUTRIBIN_STARTUP_DELAY_MS"
