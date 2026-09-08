"""Eval harness entrypoint — runs both stages and prints the accuracy table.

Responsibility
--------------
Produce the "accuracy increased" evidence the buildathon is judged on
(requirements.md FR-21..FR-25). For each configuration, run stage 1
(deterministic retrieval metrics) and stage 2 (Sarvam-105B judge), then print
the stacked comparison:

    Configuration                    Hit Rate   MRR    Faithfulness
    baseline (dense top-k)             ...      ...        ...
    + hybrid (dense ∪ BM25)            ...      ...        ...
    + rerank (cross-encoder)           ...      ...        ...
    + query expansion (MQ / HyDE)      ...      ...        ...

Hard rules
----------
* **Text-only** — no STT/TTS/translate anywhere in this loop (FR-25): golden
  questions are already English, and voice adds cost without signal.
* Configurations differ **only** by ``RetrieverConfig`` flags, so a delta is
  attributable to exactly one technique.
* Results are written to ``eval/results/`` and pasted into the README table.

Usage
-----
    python eval/run_eval.py                       # both stages, all configs
    python eval/run_eval.py --stage 1             # retrieval only (free)
    python eval/run_eval.py --config baseline,hybrid
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Run as a script (`python eval/run_eval.py`) as well as an import.
# Paths are anchored to the project root, not the working directory, so the
# harness works no matter where it is launched from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval import judge as judge_module  # noqa: E402
from eval import metrics  # noqa: E402
from rag.retrievers import RetrieverConfig, build_retriever  # noqa: E402

logger = logging.getLogger(__name__)


GOLDEN_SET_PATH = str(PROJECT_ROOT / "eval" / "golden_set.jsonl")
DOCUMENTS_JSONL = str(PROJECT_ROOT / "data" / "processed" / "documents.jsonl")
RESULTS_DIR = str(PROJECT_ROOT / "eval" / "results")

REQUIRED_FIELDS = ("question", "expected_answer", "relevant_chunk_ids")

# All configurations are scored at this cutoff, regardless of how many documents
# each returns. Re-ranking trims to 5 while dense returns 10 and hybrid ~19;
# without a common cutoff the table compares result-set size, not ranking
# quality. 5 is what the farmer's answer is actually generated from.
EVAL_TOP_K = 5

# The four rows of the README accuracy table. Each name maps to a
# ``RetrieverConfig``; flags are added one layer at a time so every delta is
# attributable to exactly one technique.
CONFIGURATIONS: dict[str, RetrieverConfig] = {
    "baseline": RetrieverConfig(),
    "hybrid": RetrieverConfig(hybrid=True),
    "hybrid+rerank": RetrieverConfig(hybrid=True, rerank=True),
    "hybrid+rerank+expansion": RetrieverConfig(hybrid=True, rerank=True, multi_query=True),
}

CONFIG_NAMES: tuple[str, ...] = tuple(CONFIGURATIONS)


def load_golden_set(path: str | Path = GOLDEN_SET_PATH) -> list[dict[str, Any]]:
    """Read the golden set: question, expected_answer, relevant_chunk_ids, …

    Warns when a labelled ``chunk_id`` is missing from ``documents.jsonl``,
    which is the signature of chunking drift after a rebuild.
    """
    src = Path(path)
    if not src.exists() or not src.stat().st_size:
        raise FileNotFoundError(
            f"{src} is empty or missing. Add labelled questions "
            "(question, expected_answer, relevant_chunk_ids, question_type)."
        )

    golden: list[dict[str, Any]] = []
    with src.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            missing = [f for f in REQUIRED_FIELDS if not item.get(f)]
            if missing:
                raise ValueError(f"{src}:{line_no} missing required field(s): {missing}")
            golden.append(item)

    _warn_on_chunk_drift(golden)
    logger.info("Loaded %s golden question(s)", len(golden))
    return golden


def _warn_on_chunk_drift(golden: list[dict[str, Any]]) -> None:
    """Warn if golden chunk IDs no longer exist in the current corpus."""
    corpus = Path(DOCUMENTS_JSONL)
    if not corpus.exists():
        return

    known: set[str] = set()
    with corpus.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                known.add(json.loads(line)["metadata"]["chunk_id"])

    missing = {
        chunk_id
        for item in golden
        for chunk_id in item["relevant_chunk_ids"]
        if chunk_id not in known
    }
    if missing:
        logger.warning(
            "%s golden chunk_id(s) are not in the current index (e.g. %s). "
            "Re-label the golden set or rebuild with the original chunking.",
            len(missing),
            sorted(missing)[:3],
        )


def configurations(names: list[str] | None = None) -> dict[str, RetrieverConfig]:
    """Build the named ``RetrieverConfig`` objects for the accuracy table."""
    if not names:
        return dict(CONFIGURATIONS)

    unknown = [name for name in names if name not in CONFIGURATIONS]
    if unknown:
        raise ValueError(f"Unknown config(s) {unknown}. Choose from {list(CONFIGURATIONS)}")
    return {name: CONFIGURATIONS[name] for name in names}


def run_stage1(golden: list[dict[str, Any]], config: RetrieverConfig) -> metrics.RetrievalScores:
    """Stage 1 — retrieve per question, score chunk IDs with ``metrics.py``.

    Retrieval only, no generation: deterministic and free.
    """
    retriever = build_retriever(config)
    per_question: list[dict[str, float]] = []
    question_types: list[str] = []

    for item in golden:
        try:
            docs = retriever.invoke(item["question"])
        except Exception as exc:  # noqa: BLE001 - score 0, keep the run alive
            logger.warning("Retrieval failed for %r: %s", item["question"][:60], exc)
            docs = []
        retrieved_ids = [doc.metadata.get("chunk_id", "") for doc in docs]
        per_question.append(
            metrics.score_question(retrieved_ids, item["relevant_chunk_ids"], EVAL_TOP_K)
        )
        question_types.append(item.get("question_type", "unspecified"))

    return metrics.aggregate(per_question, question_types)


def run_stage2(golden: list[dict[str, Any]], config: RetrieverConfig) -> dict[str, float]:
    """Stage 2 — generate answers, score faithfulness + correctness.

    Text-only: the chain is invoked without a ``session_id``, so no memory, no
    STT/TTS, no translation of the (already English) golden questions.
    """
    from rag.chain import build_chain, format_context

    chain = build_chain(config)
    verdicts: list[judge_module.JudgeVerdict] = []

    for item in golden:
        # One bad question must not void an entire paid judging run — score it
        # zero, log it, and carry on.
        try:
            payload = chain.invoke({"query": item["question"], "language": "en"})
            verdicts.append(
                judge_module.judge(
                    answer=payload.get("answer", ""),
                    context=format_context(payload.get("docs") or []),
                    expected_answer=item["expected_answer"],
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Generation/judging failed for %r: %s", item["question"][:60], exc)
            verdicts.append(
                judge_module.JudgeVerdict(0.0, 0.0, reasoning=f"run failed: {exc}")
            )

    return judge_module.aggregate_verdicts(verdicts)


def print_table(results: dict[str, dict[str, float]]) -> None:
    """Print the baseline → +technique comparison as a markdown table.

    Formatted to paste straight into the README accuracy table, with the delta
    versus baseline shown per row.
    """
    if not results:
        print("No results.")
        return

    columns = ["hit_rate", "mrr", "context_precision", "faithfulness", "correctness"]
    headers = ["Hit Rate", "MRR", "Ctx Prec", "Faithful", "Correct"]
    present = [(col, head) for col, head in zip(columns, headers) if any(col in r for r in results.values())]

    baseline = results.get("baseline", {})
    # The header itself is the longest label in most runs — include it, or the
    # separator row comes out shorter than the header.
    width = max(len("Configuration"), *(len(name) for name in results)) + 2

    print()
    print("| " + "Configuration".ljust(width) + " | " + " | ".join(h.ljust(16) for _, h in present) + " |")
    print("|" + "-" * (width + 2) + "|" + "|".join("-" * 18 for _ in present) + "|")

    for name, scores in results.items():
        cells = []
        for column, _ in present:
            if column not in scores:
                cells.append("—".ljust(16))
                continue
            value = scores[column]
            delta = value - baseline.get(column, value)
            suffix = f" ({delta:+.3f})" if name != "baseline" and baseline.get(column) is not None else ""
            cells.append(f"{value:.3f}{suffix}".ljust(16))
        print("| " + name.ljust(width) + " | " + " | ".join(cells) + " |")
    print()


def save_results(
    results: dict[str, dict[str, float]],
    configs: dict[str, RetrieverConfig],
    n_questions: int,
    results_dir: str | Path = RESULTS_DIR,
) -> Path:
    """Write a timestamped JSON results file to ``eval/results/``.

    Records the config flags, golden-set size, and judge prompt version so any
    number can be traced back to the run that produced it.
    """
    from dataclasses import asdict

    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"eval-{stamp}.json"

    path.write_text(
        json.dumps(
            {
                "timestamp": stamp,
                "n_questions": n_questions,
                "judge_prompt_version": judge_module.JUDGE_PROMPT_VERSION,
                "configurations": {name: asdict(cfg) for name, cfg in configs.items()},
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def main() -> None:
    """CLI entrypoint: load golden set → run each configuration → print + save."""
    parser = argparse.ArgumentParser(description="Two-stage accuracy harness (text-only).")
    parser.add_argument("--stage", choices=["1", "2", "both"], default="both",
                        help="1 = retrieval metrics (free), 2 = LLM judge, both = default")
    parser.add_argument("--config", help=f"Comma-separated subset of {list(CONFIGURATIONS)}")
    parser.add_argument("--golden", default=GOLDEN_SET_PATH, help="Path to golden_set.jsonl")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    golden = load_golden_set(args.golden)
    configs = configurations(args.config.split(",") if args.config else None)
    results: dict[str, dict[str, float]] = {}

    for name, config in configs.items():
        logger.info("Evaluating %s…", name)
        scores: dict[str, float] = {}
        try:
            if args.stage in ("1", "both"):
                scores.update(run_stage1(golden, config).as_dict())
            if args.stage in ("2", "both"):
                scores.update(run_stage2(golden, config))
        except Exception as exc:  # noqa: BLE001 - keep the other rows of the table
            logger.error("Configuration %r failed: %s", name, exc)
        results[name] = scores

    print_table(results)
    print(f"Results written to {save_results(results, configs, len(golden))}")


if __name__ == "__main__":
    main()
