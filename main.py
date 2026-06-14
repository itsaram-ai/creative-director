"""CLI entry point.

Usage:
  python main.py --topic "why most students misuse AI tools" \
                 --audience "university students" \
                 --platform tiktok \
                 --goal "grow followers"
"""

from __future__ import annotations

import argparse
import json

from src.orchestrator import run_pipeline
from src.schemas import Brief


def main() -> None:
    parser = argparse.ArgumentParser(description="creative-director pipeline")
    parser.add_argument("--topic", required=True)
    parser.add_argument("--audience", required=True)
    parser.add_argument("--platform", required=True,
                        choices=["tiktok", "linkedin"])
    parser.add_argument("--goal", required=True)
    parser.add_argument("--model", default=None,
                        help="override model id (optional)")
    args = parser.parse_args()

    brief = Brief(topic=args.topic, audience=args.audience,
                  platform=args.platform, goal=args.goal)
    result = run_pipeline(brief, model=args.model)

    print("\n" + "=" * 60)
    print("PLAN")
    print(json.dumps(result.plan.model_dump(), indent=2))
    print("\nDRAFT")
    print(json.dumps(result.draft.model_dump(), indent=2))
    print("\nCRITIQUE")
    print(json.dumps(result.critique.model_dump(), indent=2))
    print(f"\nrevisions: {result.revisions}   "
          f"trace: traces/{result.trace_id}.jsonl")


if __name__ == "__main__":
    main()
