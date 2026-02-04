"""Light shim for scheduled training used by the desktop app.

This module provides a `ScheduledTraining` class to avoid import errors
from `desktop_app.py`. The heavy lifting is done by
`schedule_training.py` which is invoked as a subprocess by the app.
"""
from __future__ import annotations

class ScheduledTraining:
    """Placeholder shim for scheduled training integration.

    The desktop app imports this symbol; the actual scheduling is handled
    by `schedule_training.py` which the app will call as a subprocess.
    """
    def __init__(self):
        pass

    def start(self, *args, **kwargs):
        raise NotImplementedError("Use schedule_training.py via subprocess")
import threading
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
import time


class ScheduledTraining:
    """Run a Python script at a scheduled time (daily optionally).

    Usage:
        st = ScheduledTraining(time_str="02:00", script="/path/to/train_model.py",
                               args=["--epochs", "10", "--imgsz", "640"], repeat_daily=True)
        st.start(background=True)

    Behavior:
    - time_str: 'HH:MM' in 24-hour format (local time). If invalid, starts immediately.
    - script: path to python script to run (string).
    - args: list of extra args to pass to the script (e.g. ['--epochs','20']).
    - repeat_daily: if True, runs every day at that time. If False, runs once.
    """

    def __init__(self, time_str: str, script: str, args=None, repeat_daily: bool = True):
        self.time_str = time_str
        self.script = str(script)
        self.args = list(args) if args else []
        self.repeat_daily = bool(repeat_daily)

        self._thread = None
        self._stop_event = threading.Event()
        self._process = None
        self.is_running = False

    def _parse_time(self):
        try:
            parts = self.time_str.split(":")
            hour = int(parts[0])
            minute = int(parts[1]) if len(parts) > 1 else 0
            now = datetime.now()
            target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if target <= now:
                target += timedelta(days=1)
            return target
        except Exception:
            # If parsing fails, run immediately
            return datetime.now()

    def _run_loop(self):
        self.is_running = True
        try:
            while not self._stop_event.is_set():
                next_run = self._parse_time()
                wait_seconds = (next_run - datetime.now()).total_seconds()
                # If negative for some reason, set to 0
                if wait_seconds > 0:
                    # Sleep in small increments to be responsive to stop
                    end = time.time() + wait_seconds
                    while time.time() < end and not self._stop_event.is_set():
                        time.sleep(min(1.0, end - time.time()))

                if self._stop_event.is_set():
                    break

                # Launch the training script
                try:
                    cmd = [sys.executable, self.script] + self.args
                    cwd = str(Path(self.script).resolve().parent)
                    # Use Popen so parent can continue; collect output in real usage if needed
                    self._process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        universal_newlines=True,
                        encoding='utf-8',
                        cwd=cwd,
                    )

                    # Stream output until process exits or stop requested
                    for line in self._process.stdout or []:
                        # minimal default behavior: print to stdout so UI can pick it up if desired
                        try:
                            print(line.rstrip())
                        except Exception:
                            pass
                        if self._stop_event.is_set():
                            break

                    # Wait for termination (or kill on stop)
                    if not self._stop_event.is_set():
                        self._process.wait()
                    else:
                        try:
                            self._process.terminate()
                            self._process.wait(timeout=5)
                        except Exception:
                            pass

                except Exception as e:
                    # Print errors so calling environment can catch them
                    try:
                        print(f"ScheduledTraining error: {e}")
                    except Exception:
                        pass
                finally:
                    self._process = None

                if not self.repeat_daily:
                    break

                # Sleep a short while before computing next run to avoid tight loop
                for _ in range(0, 5):
                    if self._stop_event.is_set():
                        break
                    time.sleep(1)

        finally:
            self.is_running = False

    def start(self, background: bool = True):
        """Start the scheduled trainer. If background=True, runs in a daemon thread."""
        if self.is_running:
            return
        self._stop_event.clear()
        if background:
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
        else:
            # Blocking run
            self._run_loop()

    def stop(self):
        """Stop the scheduled trainer and terminate any running training process."""
        self._stop_event.set()
        # Terminate running process if any
        try:
            if self._process and self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except Exception:
                    try:
                        self._process.kill()
                    except Exception:
                        pass
        except Exception:
            pass

        # Join thread if it's running
        try:
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=2)
        except Exception:
            pass

    @property
    def next_run_time(self):
        """Return the next scheduled run as a datetime (approximate)."""
        return self._parse_time()
