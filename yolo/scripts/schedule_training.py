"""Schedule training runs for the YOLO project.

Usage examples:
  python schedule_training.py --time 23:30 --epochs 50 --imgsz 640 --batch 8 --repeat-daily
  python schedule_training.py --time 07:00 --epochs 10 --imgsz 320 --batch 16 --once

This script will wait until the specified time (local system time) and then
invoke the project's `train_model.py` using the same interpreter. It supports
common training options so you can schedule runs with the same flags you'd
normally pass to `train_model.py`.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional


def build_train_command(yolo_root: Path, params: Dict[str, str]) -> list:
    """Build the python command to run train_model.py with provided params."""
    script_path = yolo_root / 'scripts' / 'train_model.py'
    cmd = [sys.executable, str(script_path)]

    # Basic training flags
    if params.get('epochs') is not None:
        cmd.extend(['--epochs', str(params['epochs'])])
    if params.get('imgsz') is not None:
        cmd.extend(['--imgsz', str(params['imgsz'])])
    if params.get('batch') is not None:
        cmd.extend(['--batch', str(params['batch'])])
    if params.get('device') is not None:
        cmd.extend(['--device', str(params['device'])])

    # Optional advanced flags
    if params.get('base_weights'):
        cmd.extend(['--base-weights', str(params['base_weights'])])
    if params.get('auto_create'):
        # Accept boolean-like values
        val = str(params['auto_create']).lower()
        if val in ('1', 'true', 'yes', 'y'):
            cmd.append('--auto-create')

    return cmd


def wait_until(target_dt: datetime):
    """Sleep until the target datetime (local time). Handles long waits.

    This is robust to system time drift in that it sleeps in chunks.
    """
    while True:
        now = datetime.now()
        if now >= target_dt:
            return
        # Sleep no longer than 60 seconds at a time so KeyboardInterrupt works
        delta = (target_dt - now).total_seconds()
        time.sleep(min(60, max(0.5, delta)))


def schedule_training(time_str: str,
                      params: Dict[str, str],
                      repeat_daily: bool = True,
                      once: bool = False) -> None:
    """Schedule and run training at the given time.

    time_str: 'HH:MM' in 24h format (local time).
    params: dictionary of training parameters (see build_train_command).
    repeat_daily: if True, run every day at the time. If False and not once,
                  it will run only the next occurrence.
    once: if True, run only once and exit.
    """
    yolo_root = Path(__file__).resolve().parent.parent

    # Parse time string
    try:
        hour, minute = [int(x) for x in time_str.split(':')]
    except Exception:
        raise ValueError("Time must be in HH:MM format")

    while True:
        now = datetime.now()
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)

        print(f"Next scheduled training: {target.strftime('%Y-%m-%d %H:%M:%S')}")
        wait_until(target)

        # Build and run command
        cmd = build_train_command(yolo_root, params)

        print(f"Starting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("Executing:", ' '.join(cmd))

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                cwd=str(yolo_root)
            )

            # Stream output to stdout
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line.rstrip())

            returncode = proc.wait()
            if returncode == 0:
                print(f"Training completed successfully at {datetime.now()}")
            else:
                print(f"Training failed with exit code {returncode} at {datetime.now()}")

        except KeyboardInterrupt:
            print("Interrupted by user — terminating training process if running.")
            try:
                proc.terminate()
            except Exception:
                pass
            break
        except Exception as e:
            print(f"Error running training: {e}")

        if once:
            break
        if not repeat_daily:
            break


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Schedule YOLO training runs")
    p.add_argument('--time', required=True, help='Time to run in HH:MM (24h)')
    p.add_argument('--repeat-daily', action='store_true', help='Run every day at the time')
    p.add_argument('--once', action='store_true', help='Run only once at the next occurrence')

    # Training options that mirror train_model.py
    p.add_argument('--epochs', type=int, help='Number of training epochs')
    p.add_argument('--imgsz', type=int, help='Image size for training')
    p.add_argument('--batch', type=int, help='Batch size')
    p.add_argument('--device', type=str, help='Device to use (cpu/cuda/auto)')
    p.add_argument('--base-weights', type=str, help='Path to base weights')
    p.add_argument('--auto-create', action='store_true', help='Enable auto-create behavior')

    return p.parse_args()


def main() -> None:
    args = _parse_args()

    params: Dict[str, Optional[str]] = {}
    if args.epochs is not None:
        params['epochs'] = str(args.epochs)
    if args.imgsz is not None:
        params['imgsz'] = str(args.imgsz)
    if args.batch is not None:
        params['batch'] = str(args.batch)
    if args.device is not None:
        params['device'] = args.device
    if args.base_weights is not None:
        params['base_weights'] = args.base_weights
    if args.auto_create:
        params['auto_create'] = 'true'

    try:
        schedule_training(args.time, params, repeat_daily=args.repeat_daily, once=args.once)
    except Exception as e:
        print(f"Fatal: {e}")


if __name__ == '__main__':
    main()
