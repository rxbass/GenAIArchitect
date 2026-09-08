"""Stage 2 — home-grown LLM-as-judge on Sarvam-105B. No RAGAS, no DeepEval.

Responsibility
--------------
Score generated answers on two axes (requirements.md FR-23):

    faithfulness   is every claim supported by the retrieved context?
    correctness    does the answer match the golden ``expected_answer``?

Hard rules (CLAUDE.md §When Editing the Eval)
---------------------------------------------
* The judge is **our prompt on Sarvam-105B** — kept in-repo and versioned
  (``JUDGE_PROMPT_VERSION``), never an imported scorer.
* Judge at ``temperature=0`` and parse a strict JSON verdict so reruns are as
  stable as an LLM allows.
* **Text-only** — never call STT/TTS/translate from the eval loop.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from statistics import fmean
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

JUDGE_PROMPT_VERSION = "v1"
JUDGE_TEMPERATURE = 0.0

# Versioned in-repo so a score is always reproducible against a known prompt.
FAITHFULNESS_PROMPT = """\
You are a strict grader checking whether an answer is grounded in its source \
documents. You are NOT judging whether the answer is helpful or well-written.

Given the CONTEXT (extracts from official Indian government scheme documents) \
and the ANSWER, score from 0.0 to 1.0 how fully every claim in the ANSWER is \
supported by the CONTEXT.

Scoring guide:
- 1.0  every claim is directly supported by the CONTEXT
- 0.5  the main claim is supported but some details are not stated in CONTEXT
- 0.0  the ANSWER asserts eligibility rules, amounts, deadlines, or required \
documents that do not appear in the CONTEXT

An answer that correctly says it does not know scores 1.0.

Reply with ONLY a JSON object, no other text:
{{"score": <float>, "unsupported_claims": [<string>], "reasoning": "<one sentence>"}}

CONTEXT:
{context}

ANSWER:
{answer}"""

CORRECTNESS_PROMPT = """\
You are grading a farmer-assistant answer against a reference answer. Judge \
factual agreement, NOT wording, length, or language — the answer may be written \
in an Indian language while the reference is in English; translate mentally and \
compare the facts.

Score from 0.0 to 1.0:
- 1.0  all key facts of the EXPECTED_ANSWER are present and none contradict it
- 0.5  partially correct — some key facts missing, nothing contradictory
- 0.0  contradicts the EXPECTED_ANSWER, or misses all of its key facts

Reply with ONLY a JSON object, no other text:
{{"score": <float>, "missing": [<string>], "contradictions": [<string>], "reasoning": "<one sentence>"}}

EXPECTED_ANSWER:
{expected_answer}

ANSWER:
{answer}"""

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)

_judge_llm: BaseChatModel | None = None


@dataclass
class JudgeVerdict:
    """One judged answer: both scores plus the reasoning that produced them."""

    faithfulness: float
    correctness: float
    unsupported_claims: list[str] = field(default_factory=list)
    reasoning: str = ""
    prompt_version: str = JUDGE_PROMPT_VERSION

    def as_dict(self) -> dict[str, Any]:
        """Serializable form for ``eval/results/``."""
        return asdict(self)


def get_judge_llm() -> BaseChatModel:
    """Sarvam-105B at ``temperature=0`` for grading.

    Reuses ``services.sarvam.get_chat_model`` — same vendor, same wrapper.
    """
    global _judge_llm
    if _judge_llm is None:
        from services.sarvam import get_chat_model

        _judge_llm = get_chat_model(temperature=JUDGE_TEMPERATURE)
    return _judge_llm


def parse_verdict(raw: str) -> dict[str, Any]:
    """Parse the judge's JSON reply defensively.

    Strips code fences and tolerates trailing prose. A reply that cannot be
    parsed is reported with the raw text attached rather than silently scored 0
    — a parse failure is a harness bug, not a model failure.
    """
    text = str(raw).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()

    start = text.find("{")
    if start == -1:
        raise ValueError(f"Judge reply contained no JSON object: {text[:200]!r}")

    # `raw_decode` reads the FIRST complete object and ignores whatever follows.
    # A greedy `\{.*\}` match instead ran to the last brace in the reply, so a
    # perfectly good verdict with one stray trailing brace —
    #   {"score": 1.0, ...}}
    # — raised "Extra data" and was recorded as 0.0. That silently deflated
    # every faithfulness number in the accuracy table.
    try:
        verdict, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as exc:
        # The model sometimes emits genuinely invalid JSON — most often an
        # unescaped quote inside its own reasoning string. The score is the only
        # field the harness actually needs, so salvage it rather than throw away
        # a real judgement and record a 0.0 that never happened.
        salvaged = re.search(r'"score"\s*:\s*([0-9]*\.?[0-9]+)', text)
        if not salvaged:
            raise ValueError(
                f"Judge reply was not valid JSON ({exc}): {text[:200]!r}"
            ) from exc
        logger.warning("Malformed judge JSON (%s) — salvaged score only", exc)
        verdict = {
            "score": float(salvaged.group(1)),
            "reasoning": "score salvaged from malformed JSON",
            "partial_parse": True,
        }

    if not isinstance(verdict, dict):
        raise ValueError(f"Judge reply was not a JSON object: {text[:200]!r}")

    verdict["score"] = max(0.0, min(1.0, float(verdict.get("score", 0.0))))
    return verdict


def _judge_once(prompt: str, llm: BaseChatModel | None, **values: str) -> dict[str, Any]:
    """Render one judge prompt, invoke, and parse — with a visible failure mode."""
    llm = llm or get_judge_llm()
    try:
        reply = llm.invoke(prompt.format(**values)).content
    except Exception as exc:  # noqa: BLE001 - one API hiccup must not void a paid run
        logger.warning("Judge call failed: %s", exc)
        return {"score": 0.0, "reasoning": f"judge call failed: {exc}", "judge_error": True}
    try:
        return parse_verdict(reply)
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning("Unparseable judge verdict (%s): %s", exc, str(reply)[:200])
        return {"score": 0.0, "reasoning": f"unparseable verdict: {exc}", "parse_error": True}


def judge_faithfulness(answer: str, context: str, llm: BaseChatModel | None = None) -> dict[str, Any]:
    """Score whether the answer is grounded in the retrieved context."""
    return _judge_once(FAITHFULNESS_PROMPT, llm, context=context, answer=answer)


def judge_correctness(answer: str, expected_answer: str, llm: BaseChatModel | None = None) -> dict[str, Any]:
    """Score the answer against the golden expected answer."""
    return _judge_once(CORRECTNESS_PROMPT, llm, expected_answer=expected_answer, answer=answer)


def judge(answer: str, context: str, expected_answer: str) -> JudgeVerdict:
    """Run both judgements for one golden question and merge them.

    The two scorers are independent, so they are fanned out with
    ``RunnableParallel`` — one round trip of latency instead of two.
    """
    from langchain_core.runnables import RunnableLambda, RunnableParallel

    llm = get_judge_llm()
    results = RunnableParallel(
        faithfulness=RunnableLambda(lambda _: judge_faithfulness(answer, context, llm)),
        correctness=RunnableLambda(lambda _: judge_correctness(answer, expected_answer, llm)),
    ).invoke({})

    faithfulness, correctness = results["faithfulness"], results["correctness"]
    reasoning = "; ".join(
        part for part in (faithfulness.get("reasoning"), correctness.get("reasoning")) if part
    )
    return JudgeVerdict(
        faithfulness=float(faithfulness.get("score", 0.0)),
        correctness=float(correctness.get("score", 0.0)),
        unsupported_claims=list(faithfulness.get("unsupported_claims") or []),
        reasoning=reasoning,
    )


def aggregate_verdicts(verdicts: list[JudgeVerdict]) -> dict[str, float]:
    """Mean faithfulness and correctness across the golden set."""
    if not verdicts:
        return {"faithfulness": 0.0, "correctness": 0.0}
    return {
        "faithfulness": fmean(verdict.faithfulness for verdict in verdicts),
        "correctness": fmean(verdict.correctness for verdict in verdicts),
    }
