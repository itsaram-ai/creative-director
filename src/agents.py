"""Planner, writer, and critic adapters for the content pipeline.

Each agent loads a versioned prompt, formats its task context, and delegates
JSON generation and Pydantic validation to `generate_structured()`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from .llm_client import LLMClient
from .schemas import Brief, Critique, Draft, Plan

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
T = TypeVar("T", bound=BaseModel)


def load_prompt(name: str) -> str:
    """Prompts are versioned files (planner_v1.md), not inline strings.

    This is the 'prompts as first-class engineering artifacts' principle:
    they get diffed, reviewed, and tested like code.
    """
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


def generate_structured(
    client: LLMClient,
    system_prompt: str,
    user_prompt: str,
    schema: type[T],
    max_attempts: int = 3,
) -> T:
    """Call the LLM and validate its JSON output against a pydantic schema.

    On failure (bad JSON or schema violation) the error is fed back to the
    model and it retries - the simplest reliable structured-generation loop.
    """
    feedback = ""
    for _ in range(max_attempts):
        response = client.complete(system_prompt, user_prompt + feedback)
        raw = _strip_code_fences(response.text)
        try:
            return schema.model_validate(json.loads(raw))
        except (json.JSONDecodeError, ValidationError) as err:
            feedback = (
                "\n\nYour previous reply was invalid. Return ONLY valid JSON "
                f"matching the schema. Validation error:\n{err}"
            )
    raise RuntimeError(f"Failed to produce valid {schema.__name__} "
                       f"after {max_attempts} attempts")


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    return text.strip()


def run_planner(client: LLMClient, brief: Brief, chunks) -> Plan:
    """Create a grounded content plan from a brief and retrieved chunks."""
    system_prompt = load_prompt("planner_v1.md")

    evidence_parts = []

    for chunk in chunks:
        formatted_chunk = f"[{chunk.id}]\n{chunk.text}"
        evidence_parts.append(formatted_chunk)

    evidence_text = "\n\n".join(evidence_parts)

    brief_json = json.dumps(brief.model_dump(), indent=2)
    user_prompt = f"""BRIEF:
    {brief_json}

    RETRIEVED KNOWLEDGE:
    {evidence_text}
    """
    return generate_structured(client, system_prompt, user_prompt, Plan)


def run_writer(
    client: LLMClient,
    brief: Brief,
    plan: Plan,
    feedback: str | None = None,
) -> Draft:
    """Execute a plan, optionally incorporating revision feedback."""
    system_prompt = load_prompt("writer_v2.md")
    
    brief_json = json.dumps(brief.model_dump(), indent=2)
    plan_json = json.dumps(plan.model_dump(), indent=2)

    user_prompt = f"""BRIEF:
    {brief_json}

    PLAN:
    {plan_json}
    """

    if feedback is not None:
        user_prompt += f"\n\nREVISION NOTES FROM CRITIC:\n{feedback}"

    return generate_structured(client, system_prompt, user_prompt, Draft)


def run_critic(
    client: LLMClient,
    brief: Brief,
    plan: Plan,
    draft: Draft,
    chunks,
) -> Critique:
    """Score a draft against the brief, plan, and retrieved evidence."""
    system_prompt = load_prompt("critic_v1.md")

    evidence_parts = []

    for chunk in chunks:
        formatted_chunk = f"[{chunk.id}]\n{chunk.text}"
        evidence_parts.append(formatted_chunk)

    evidence_text = "\n\n".join(evidence_parts)

    brief_json = json.dumps(brief.model_dump(), indent=2)
    plan_json = json.dumps(plan.model_dump(), indent=2)
    draft_json = json.dumps(draft.model_dump(), indent=2)

    user_prompt = f"""BRIEF:
    {brief_json}

    PLAN:
    {plan_json}

    DRAFT:
    {draft_json}

    RETRIEVED KNOWLEDGE:
    {evidence_text}
    """
    
    return generate_structured(client, system_prompt, user_prompt, Critique)
