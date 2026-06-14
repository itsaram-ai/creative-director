"""Eval harness: run the full pipeline over a golden brief set and check
output properties. This is regression testing for an LLM system - run it
after every prompt or model change and compare pass rates.

Usage:  python -m evals.run_evals
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.orchestrator import run_pipeline  # noqa: E402
from src.schemas import Brief, PipelineResult  # noqa: E402

GOLDEN = Path(__file__).resolve().parent / "golden_briefs.json"


def check_case(result: PipelineResult, expect: dict) -> list[tuple[str, bool, str]]:
    """Each check returns (name, passed, detail)."""
    checks: list[tuple[str, bool, str]] = []

    # 1. Schema validity is implicitly proven by PipelineResult existing,
    #    but we assert the structured fields are populated.
    checks.append(("schema_populated",
                   bool(result.draft.headline and result.draft.body),
                   "headline/body non-empty"))

    # 2. Platform constraint: hashtag count
    n_tags = len(result.draft.hashtags)
    checks.append(("hashtag_count",
                   n_tags <= expect["max_hashtags"],
                   f"{n_tags} tags (max {expect['max_hashtags']})"))

    # 3. Groundedness proxy: the plan must cite retrieved evidence,
    #    and the critic's groundedness score must be >= 3.
    checks.append(("evidence_cited",
                   len(result.plan.retrieved_evidence) > 0,
                   f"cited {result.plan.retrieved_evidence}"))
    checks.append(("critic_groundedness",
                   result.critique.groundedness >= 3,
                   f"score {result.critique.groundedness}"))

    # 4. Platform voice proxy: output mentions platform-native concepts
    text = (result.plan.hook_strategy + " " + result.draft.body
            + " " + result.draft.rationale).lower()
    hit = any(term.lower() in text for term in expect["platform_terms_any"])
    checks.append(("platform_voice", hit,
                   f"looked for any of {expect['platform_terms_any']}"))

    # 5. Quality bar: critic mean score
    checks.append(("mean_score",
                   result.critique.mean_score >= expect["min_mean_score"],
                   f"{result.critique.mean_score:.2f} "
                   f"(min {expect['min_mean_score']})"))
    return checks


def main() -> None:
    cases = json.loads(GOLDEN.read_text())
    total = passed = 0
    for i, case in enumerate(cases, 1):
        brief = Brief.model_validate(case["brief"])
        print(f"\n=== Case {i}: {brief.topic[:60]} [{brief.platform}] ===")
        result = run_pipeline(brief)
        for name, ok, detail in check_case(result, case["expect"]):
            total += 1
            passed += ok
            print(f"  {'PASS' if ok else 'FAIL'}  {name:20s} {detail}")
        print(f"  trace: traces/{result.trace_id}.jsonl  "
              f"revisions: {result.revisions}")
    print(f"\n{passed}/{total} checks passed "
          f"({passed / total:.0%})" if total else "no cases")


if __name__ == "__main__":
    main()
