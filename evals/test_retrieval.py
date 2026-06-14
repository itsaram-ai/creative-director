"""Acceptance tests for YOUR retrieval implementation.

Run: pytest evals/test_retrieval.py -v
All four must pass before you wire retrieval into the pipeline.
"""

from pathlib import Path

import pytest

from src import retrieval

KNOWLEDGE = Path(__file__).resolve().parent.parent / "knowledge"


@pytest.fixture(scope="module")
def index():
    chunks = retrieval.load_chunks(KNOWLEDGE)
    assert len(chunks) >= 8, "expected multiple chunks per file"
    assert all(c.id and c.text and c.source_file for c in chunks)
    return retrieval.build_index(chunks)


def test_tokenize_basics():
    tokens = retrieval.tokenize("The First-Line of ANY post, decides 80%!")
    assert "first" in " ".join(tokens) or "first-line" in tokens or "line" in tokens
    assert "the" not in tokens, "stopwords should be removed"
    assert all(len(t) >= 3 for t in tokens)


def test_topical_query_hits_right_file(index):
    results = retrieval.top_k(index, "tiktok hashtags algorithm", k=3)
    assert results, "no results returned"
    top_chunk, top_score = results[0]
    assert top_chunk.source_file == "platform_tiktok.md"
    assert top_score > 0


def test_scores_sorted_descending(index):
    results = retrieval.top_k(index, "hook first line attention", k=4)
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_irrelevant_query_low_scores(index):
    relevant = retrieval.top_k(index, "linkedin dwell time hook", k=1)
    irrelevant = retrieval.top_k(index, "quantum chromodynamics zebra", k=1)
    if irrelevant:  # may legitimately return nothing
        assert irrelevant[0][1] < relevant[0][1]
