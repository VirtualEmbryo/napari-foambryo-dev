"""Library-call stand-ins for the runner's tests: slow, failing, or with a process of their own.

The runner imports its target by `"module:function"` in a spawned child, so these live in an importable module.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path


def sleep_with_a_grandchild(pid_file: str, seconds: float = 600.0) -> str:
    """Start a process of its own (as a seed ensemble's workers are), write both PIDs, then sleep."""
    grandchild = subprocess.Popen([sys.executable, "-c", f"import time; time.sleep({seconds})"])
    Path(pid_file).write_text(f"{os.getpid()} {grandchild.pid}")
    time.sleep(seconds)
    return "finished"


def refuse(message: str) -> None:
    """Raise as the library does when it refuses."""
    raise ValueError(message)


def report_progress(n: int, progress=None) -> int:
    """Call `progress(done, total, item)` n times, as `infer_tensions_over_seeds` does per seed."""
    for done in range(1, n + 1):
        progress(done, n, done - 1)
    return n
