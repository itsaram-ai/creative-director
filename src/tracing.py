"""Trace logging: every agent decision written to traces/<trace_id>.jsonl.

Why it matters: in a production content system you must attribute downstream
performance (did the post work?) to upstream decisions (which angle, which
prompt version, which retrieved evidence). Trace IDs are how you join those.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

TRACES_DIR = Path(__file__).resolve().parent.parent / "traces"


class Tracer:
    def __init__(self, trace_id: str):
        self.trace_id = trace_id
        TRACES_DIR.mkdir(exist_ok=True)
        self.path = TRACES_DIR / f"{trace_id}.jsonl"

    def log(self, step: str, *, input: Any, output: Any,  # noqa: A002
            prompt_version: str | None = None) -> None:
        record = {
            "trace_id": self.trace_id,
            "ts": time.time(),
            "step": step,
            "prompt_version": prompt_version,
            "input": input,
            "output": output,
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
