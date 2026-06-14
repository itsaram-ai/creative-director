# creative-director

A multi-agent LLM pipeline that decides **what** content to make — then makes
it, critiques it, and revises it. Built as a focused study of the agentic
patterns behind production LLM systems: orchestration, RAG, structured
generation, evals, and decision tracing.

> Generation is solved; judgement isn't. Models can make almost anything you
> ask for — the harder question is what to make. This project puts that
> decision (the planner) at the front of the pipeline and grounds it in
> retrieved knowledge.

## Architecture

```
brief (topic, audience, platform, goal)
        │
        ▼
  ┌─────────────┐     TF-IDF retrieval over markdown
  │   PLANNER    │◄─── knowledge base (top-k chunks,
  │ decides WHAT │     cited by chunk_id in the plan)
  └──────┬──────┘
         ▼
  ┌─────────────┐     structured JSON output,
  │    WRITER    │     pydantic-validated with
  │ executes plan│     validation-error retry
  └──────┬──────┘
         ▼
  ┌─────────────┐     4-dimension rubric incl.
  │    CRITIC    │     groundedness vs retrieved
  │ scores draft │     evidence
  └──────┬──────┘
         │ verdict == "revise"? → feedback loops back
         │ to the writer (max 2 revisions)
         ▼
  result + full trace (traces/<id>.jsonl — every agent
  decision logged with prompt versions and evidence IDs)
```

## Design decisions

- **Prompts are versioned files** (`prompts/planner_v1.md`), not inline
  strings — diffed, reviewed, and regression-tested like code.
- **Structured generation via validate-and-retry**: every agent must emit
  JSON matching a pydantic schema; validation errors are fed back to the
  model. Simple, robust, model-agnostic.
- **TF-IDF retrieval, implemented from scratch**: the corpus is tiny, so
  neural embeddings would be unjustified complexity — and building it by
  hand proves the mechanics. Swapping in sentence-transformers is a
  one-module change (see Future work).
- **Groundedness is enforced twice**: the planner must cite chunk IDs, and
  the critic scores the draft against the actual retrieved evidence.
- **Every decision is traced** with prompt versions and evidence IDs, so
  downstream outcomes can be attributed to upstream choices — the
  prerequisite for any closed-loop improvement system.
- **Dual LLM backend**: uses the Anthropic SDK when `ANTHROPIC_API_KEY` is
  set, otherwise falls back to Claude Code headless mode (`claude -p`) on
  subscription billing.

## Setup

```bash
python -m venv .venv

# macOS/Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate

python -m pip install -r requirements.txt

# Backend A (recommended): Anthropic API
export ANTHROPIC_API_KEY=sk-ant-...

# Backend B (no key needed): Claude Code logged in via `claude login`
```

## Usage

```bash
python main.py --topic "why most students misuse AI tools" \
               --audience "university students" \
               --platform tiktok \
               --goal "grow followers"
```

## Sample output

Abbreviated output from a TikTok brief:

```json
{
  "plan": {
    "angle": "Confessional narrative: how getting caught using ChatGPT backfired",
    "format": "Fast-cut video with on-screen text",
    "retrieved_evidence": [
      "platform_tiktok.md:0",
      "platform_tiktok.md:1",
      "hooks_and_headlines.md:1",
      "audience_gen_z.md:0"
    ]
  },
  "draft": {
    "headline": "i got caught using chatgpt and froze (here's why that backfired)",
    "hashtags": [
      "#AIacademicintegrity",
      "#collegeconfessions",
      "#studentlife"
    ]
  },
  "critique": {
    "relevance": 4,
    "hook_strength": 4,
    "audience_fit": 4,
    "groundedness": 4,
    "verdict": "pass"
  },
  "revisions": 1
}
```

The full run also writes a JSONL decision trace to `traces/<trace_id>.jsonl`.

## Evals

```bash
python -m pytest evals/test_retrieval.py -v  # retrieval unit tests
python -m evals.run_evals                  # full-pipeline golden-set evals
```

The golden set (`evals/golden_briefs.json`) checks schema validity,
platform constraints (e.g. hashtag limits), evidence citation,
critic groundedness, platform voice, and a minimum rubric score —
run after every prompt or model change to catch regressions.

Current golden-set result: **18/18 checks passing**.

## Future work

- Swap TF-IDF for sentence-transformer embeddings behind the same
  `top_k` interface
- Online loop: feed real engagement metrics back into planner context
- A/B prompt versions (`planner_v2`) compared on the golden set
- Parallel candidate generation with critic-as-selector
