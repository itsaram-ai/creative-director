"""LLM client with two interchangeable backends.

Backend selection (automatic):
  1. If ANTHROPIC_API_KEY is set  -> Anthropic Python SDK (recommended)
  2. Otherwise                    -> Claude Code headless mode (`claude -p`),
                                     billed to your Claude subscription.

Every call returns plain text. Structured-output parsing/validation happens
one layer up (see agents.py) so the client stays dumb and swappable.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass

DEFAULT_MODEL = os.environ.get("CD_MODEL", "claude-haiku-4-5-20251001")
MAX_TOKENS = int(os.environ.get("CD_MAX_TOKENS", "1500"))


@dataclass
class LLMResponse:
    text: str
    model: str
    backend: str
    latency_s: float


class LLMClient:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        if self.api_key:
            self.backend = "anthropic_sdk"
            from anthropic import Anthropic  # imported lazily on purpose

            self._client = Anthropic()
        elif shutil.which("claude"):
            self.backend = "claude_cli"
            self._client = None
        else:
            raise RuntimeError(
                "No backend available. Either set ANTHROPIC_API_KEY or install "
                "Claude Code (https://docs.claude.com/en/docs/claude-code/overview) "
                "and run `claude login`."
            )

    def complete(self, system: str, user: str) -> LLMResponse:
        """Single-turn completion. Retries transient failures with backoff."""
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                start = time.perf_counter()
                if self.backend == "anthropic_sdk":
                    text = self._complete_sdk(system, user)
                else:
                    text = self._complete_cli(system, user)
                latency = time.perf_counter() - start
                return LLMResponse(text=text, model=self.model,
                                   backend=self.backend, latency_s=latency)
            except Exception as err:  # noqa: BLE001 - log-and-retry boundary
                last_err = err
                time.sleep(2**attempt)  # 1s, 2s, 4s
        raise RuntimeError(f"LLM call failed after 3 attempts: {last_err}")

    # ------------------------------------------------------------------ #

    def _complete_sdk(self, system: str, user: str) -> str:
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(
            block.text for block in msg.content if block.type == "text"
        )

    def _complete_cli(self, system: str, user: str) -> str:
        """Shell out to `claude -p` (Claude Code headless mode).

        Uses subscription billing - no API key needed. We ask for JSON
        envelope output and pull the `result` field, falling back to raw
        stdout if the envelope shape ever changes.
        """
        prompt = f"{system}\n\n---\n\n{user}"
        cmd = [
            "claude", "-p", prompt,
            "--output-format", "json",
            "--model", self.model,
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
            check=True,
        )
        try:
            envelope = json.loads(proc.stdout)
            return envelope.get("result", proc.stdout)
        except json.JSONDecodeError:
            return proc.stdout.strip()
