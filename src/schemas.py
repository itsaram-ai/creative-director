"""Pydantic schemas: the structured-generation contract.

Every agent must emit JSON matching one of these models. Validation failures
trigger a retry with the validation error fed back to the model - this is
the simplest robust pattern for structured generation.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class Brief(BaseModel):
    """What the user asks the pipeline to do."""

    topic: str
    audience: str
    platform: str  # e.g. "tiktok", "linkedin"
    goal: str  # e.g. "drive signups", "grow followers"


class Plan(BaseModel):
    """Planner output: the WHAT-to-make decision."""

    angle: str = Field(description="The creative angle chosen and why it fits")
    format: str = Field(description="Content format, e.g. 'listicle', 'hot take'")
    hook_strategy: str = Field(description="How the first line earns attention")
    key_points: list[str] = Field(min_length=2, max_length=5)
    retrieved_evidence: list[str] = Field(
        description="Chunk IDs from the knowledge base that informed this plan"
    )


class Draft(BaseModel):
    """Writer output: the actual content, fully structured."""

    headline: str = Field(max_length=120)
    body: str
    hashtags: list[str] = Field(min_length=2, max_length=8)
    rationale: str = Field(description="Why this execution serves the plan")

    @field_validator("hashtags")
    @classmethod
    def hashtags_start_with_hash(cls, v: list[str]) -> list[str]:
        return [h if h.startswith("#") else f"#{h}" for h in v]


class Critique(BaseModel):
    """Critic output: scored against a rubric, with actionable feedback."""

    relevance: int = Field(ge=1, le=5)
    hook_strength: int = Field(ge=1, le=5)
    audience_fit: int = Field(ge=1, le=5)
    groundedness: int = Field(
        ge=1, le=5,
        description="Does the draft reflect the retrieved knowledge, "
                    "or did the writer ignore/contradict it?",
    )
    verdict: str = Field(pattern="^(pass|revise)$")
    feedback: str = Field(description="Specific, actionable revision notes")

    @property
    def mean_score(self) -> float:
        return (self.relevance + self.hook_strength
                + self.audience_fit + self.groundedness) / 4


class PipelineResult(BaseModel):
    """Everything the pipeline produces for one brief."""

    brief: Brief
    plan: Plan
    draft: Draft
    critique: Critique
    revisions: int
    trace_id: str
