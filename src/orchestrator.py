"""Pipeline orchestrator: plan -> write -> critique -> (revise) -> result.

This module is complete - study it. It shows the agent loop pattern that
36 Labs' JD calls "closing the loop": the critic's feedback drives revision,
and every decision is traced so outcomes can be attributed to the upstream
choices that produced them.
"""

from __future__ import annotations

import uuid

from . import retrieval
from .agents import run_critic, run_planner, run_writer
from .llm_client import LLMClient
from .schemas import Brief, PipelineResult
from .tracing import Tracer

MAX_REVISIONS = 2


def run_pipeline(brief: Brief, knowledge_dir: str = "knowledge",
                 model: str | None = None) -> PipelineResult:
    trace_id = uuid.uuid4().hex[:12]
    tracer = Tracer(trace_id)
    client = LLMClient(model) if model else LLMClient()

    # 1. RAG: ground the planner in the knowledge base
    chunks_index = retrieval.build_index(retrieval.load_chunks(knowledge_dir))
    query = f"{brief.topic} {brief.audience} {brief.platform}"
    retrieved = retrieval.top_k(chunks_index, query, k=4)
    tracer.log("retrieval", input=query,
               output=[(c.id, round(s, 4)) for c, s in retrieved])
    chunks = [c for c, _ in retrieved]

    # 2. Planner decides WHAT to make
    plan = run_planner(client, brief, chunks)
    tracer.log("planner", input=brief.model_dump(), output=plan.model_dump(),
               prompt_version="planner_v1")

    # 3. Writer -> Critic loop, with critic feedback driving revision
    feedback = None
    revisions = 0
    while True:
        draft = run_writer(client, brief, plan, feedback=feedback)
        tracer.log("writer", input={"feedback": feedback},
                   output=draft.model_dump(), prompt_version="writer_v2")

        critique = run_critic(client, brief, plan, draft, chunks)
        tracer.log("critic", input=draft.headline,
                   output=critique.model_dump(), prompt_version="critic_v1")

        if critique.verdict == "pass" or revisions >= MAX_REVISIONS:
            break
        feedback = critique.feedback
        revisions += 1

    result = PipelineResult(brief=brief, plan=plan, draft=draft,
                            critique=critique, revisions=revisions,
                            trace_id=trace_id)
    tracer.log("pipeline_done", input=None,
               output={"revisions": revisions,
                       "mean_score": critique.mean_score})
    return result
