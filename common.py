"""Common boilerplate shared by code/a1..a4.

Provides:
* sys.path / cwd setup so each script can be run from any directory
* a tiny ``log_result(...)`` helper that appends a JSON line to results/aN.jsonl
* per-aspect result dir creation
"""
import json
import os
import sys
import time
from typing import Dict


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def setup_paths():
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    os.chdir(ROOT)


def results_path(aspect: str, suffix: str = "results.jsonl") -> str:
    d = os.path.join(ROOT, "results", aspect)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, suffix)


def log_result(aspect: str, record: Dict) -> None:
    p = results_path(aspect)
    record = {**record, "ts": time.time()}
    with open(p, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"  -> appended to {p}", flush=True)


def hb(msg: str) -> None:
    import datetime
    line = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open(os.path.join(ROOT, "results", "_progress.log"), "a") as f:
            f.write(line + "\n")
    except Exception:
        pass
