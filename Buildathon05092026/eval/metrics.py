"""Stage 1 — deterministic retrieval metrics. No API calls, ₹0, reproducible.

Responsibility
--------------
Compare the ``chunk_id``s a retriever returned against the labelled ground-truth
``chunk_id``s in ``eval/golden_set.jsonl`` and report:

    Hit Rate / Recall@k    did any / how many gold chunks appear in the top-k?
    MRR                    how high did the first gold chunk rank?
    Context precision      what fraction of returned chunks were actually gold?

Hard rules (CLAUDE.md §When Editing the Eval)
---------------------------------------------
* **Deterministic and free** — pure set/rank arithmetic over chunk IDs. No LLM
  touches this stage.
* No third-party scorer (no RAGAS). This file *is* the metric implementation.
* Chunk IDs must match current chunking; a rebuild with different chunk sizes
  invalidates the golden labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean


@dataclass
class RetrievalScores:
    """Aggregated stage-1 results for one retriever configuration."""

    hit_rate: float
    recall_at_k: float
    mrr: float
    context_precision: float
    n_questions: int = 0
    by_question_type: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float]:
        """Flat mapping for the results table and the JSON results file."""
        return {
            "hit_rate": self.hit_rate,
            "recall_at_k": self.recall_at_k,
            "mrr": self.mrr,
            "context_precision": self.context_precision,
        }


def _dedupe(ids: list[str]) -> list[str]:
    """Preserve rank order while dropping repeats.

    A chunk can surface from several retrieval arms; counting it twice would
    inflate precision and distort ranks.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for chunk_id in ids:
        if chunk_id and chunk_id not in seen:
            seen.add(chunk_id)
            ordered.append(chunk_id)
    return ordered


def hit_rate(retrieved_ids: list[str], gold_ids: list[str]) -> float:
    """1.0 if any gold chunk appears in ``retrieved_ids``, else 0.0."""
    if not gold_ids:
        return 0.0
    return 1.0 if set(retrieved_ids) & set(gold_ids) else 0.0


def recall_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    """Fraction of gold chunks present in the top-k retrieved IDs."""
    if not gold_ids:
        return 0.0
    top_k = set(_dedupe(retrieved_ids)[:k])
    return len(top_k & set(gold_ids)) / len(set(gold_ids))


def reciprocal_rank(retrieved_ids: list[str], gold_ids: list[str]) -> float:
    """``1 / rank`` of the first gold chunk (1-indexed); 0.0 if absent."""
    gold = set(gold_ids)
    for rank, chunk_id in enumerate(_dedupe(retrieved_ids), start=1):
        if chunk_id in gold:
            return 1.0 / rank
    return 0.0


def context_precision(retrieved_ids: list[str], gold_ids: list[str]) -> float:
    """Fraction of retrieved chunks that are gold — measures noise in context."""
    retrieved = _dedupe(retrieved_ids)
    if not retrieved:
        return 0.0
    return sum(1 for chunk_id in retrieved if chunk_id in set(gold_ids)) / len(retrieved)


def score_question(retrieved_ids: list[str], gold_ids: list[str], k: int) -> dict[str, float]:
    """Compute all four metrics for a single golden question, at a fixed cutoff.

    **Every metric is measured on the same top-k slice.** Configurations return
    different numbers of documents — plain dense returns 10, hybrid ~19, and
    re-ranking trims to 5 — so scoring the full returned list makes the
    comparison meaningless: re-ranking would appear to *lose* hit rate purely
    for returning fewer documents, and to gain precision for the same reason.
    Truncating everything to k is what makes the accuracy table an
    apples-to-apples comparison of ranking quality.
    """
    top_k = _dedupe(retrieved_ids)[:k]
    return {
        "hit_rate": hit_rate(top_k, gold_ids),
        "recall_at_k": recall_at_k(top_k, gold_ids, k),
        "mrr": reciprocal_rank(top_k, gold_ids),
        "context_precision": context_precision(top_k, gold_ids),
    }


def aggregate(
    per_question: list[dict[str, float]],
    question_types: list[str] | None = None,
) -> RetrievalScores:
    """Mean each metric across the golden set.

    A plain mean, deliberately — the numbers stay explainable to a judge.
    """
    if not per_question:
        return RetrievalScores(0.0, 0.0, 0.0, 0.0, 0)

    scores = RetrievalScores(
        hit_rate=fmean(item["hit_rate"] for item in per_question),
        recall_at_k=fmean(item["recall_at_k"] for item in per_question),
        mrr=fmean(item["mrr"] for item in per_question),
        context_precision=fmean(item["context_precision"] for item in per_question),
        n_questions=len(per_question),
    )

    if question_types:
        buckets: dict[str, list[float]] = {}
        for qtype, item in zip(question_types, per_question):
            buckets.setdefault(qtype or "unspecified", []).append(item["hit_rate"])
        scores.by_question_type = {name: fmean(values) for name, values in buckets.items()}

    return scores
