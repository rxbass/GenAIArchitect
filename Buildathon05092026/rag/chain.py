"""The LCEL pipeline — the whole request lifecycle as one Runnable.

Responsibility
--------------
Compose stages 2–11 of ``README.md`` §Request Lifecycle into a single LangChain
Runnable that ``app.py`` (UI) and ``eval/run_eval.py`` (harness) both invoke.

    input guard ─▶ query→EN ─▶ expand ─▶ retrieve ─▶ rerank ─▶ [gate]
        ├─ weak  ─▶ no-confident-answer fallback
        └─ ok    ─▶ generate (in farmer's language, grounded)
                  ─▶ groundedness gate ─▶ topical rail ─▶ memory + transcript

Constraints (CLAUDE.md Golden Rules)
------------------------------------
* **LCEL only** — ``RunnableSequence`` / ``RunnableBranch`` / ``RunnableLambda``
  / ``RunnableParallel`` / ``RunnablePassthrough`` /
  ``RunnableWithMessageHistory``. **No LangGraph.**
* **Answer in the farmer's language directly** (path A) — Sarvam-105B generates
  in-language from English context; there is no answer-then-translate round trip.
* **Grounded or nothing** — the weak-retrieval branch returns the fallback.
* Voice (Saaras/Bulbul) lives at the edges in ``app.py``; the chain itself is
  text-only, which is what makes the eval harness able to reuse it.

Payload contract
----------------
A plain dict flows through every stage, accumulating keys via
``RunnablePassthrough.assign``::

    {
      "query":            str,   # raw farmer input (cleaned by the input guard)
      "language":         str,   # ta | hi | kn | te | ml | en
      "translated_query": str,   # English, what retrieval actually saw
      "docs":             list[Document],
      "answer":           str,
      "citations":        list[dict],
      "retrieval":        {"n_docs", "top_score", "fallback"},
      "guards":           {"input", "injection", "retrieval", "grounded", "on_topic"},
      "blocked":          bool,
      "session_id":       str,
    }
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
)

from rag import validators
from rag.retrievers import RetrieverConfig, build_retriever

if TYPE_CHECKING:  # pragma: no cover - typing only
    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_core.documents import Document
    from langchain_core.language_models import BaseChatModel
    from langchain_core.runnables import Runnable

logger = logging.getLogger(__name__)

LANGUAGE_NAMES: dict[str, str] = {
    "ta": "Tamil",
    "hi": "Hindi",
    "kn": "Kannada",
    "te": "Telugu",
    "ml": "Malayalam",
    "en": "English",
}

# Grounded-answer system prompt. Kept in-repo and versioned; it is the contract
# that keeps Sarvam-105B inside the retrieved context and in-language.
ANSWER_SYSTEM_PROMPT = """\
You are Kisan Sahayak, an assistant that helps Indian farmers understand \
government agricultural schemes.

Rules you must follow:
1. Answer ONLY using the CONTEXT below. It is extracted from official scheme \
documents.
2. If the CONTEXT does not contain the answer, say you do not know and suggest \
the farmer ask about a specific scheme. Never guess an eligibility rule, \
amount, deadline, or document list.
3. Write your entire answer in {language_name}. The CONTEXT is in English; \
translate the meaning naturally rather than word-for-word. Keep official scheme \
names (e.g. PM-KISAN) in their original form.
4. Name the **scheme** itself when you can (e.g. PM-KISAN, Kisan Credit Card). \
Never mention documents, files, sources, page titles, or "the context" — the app \
shows the farmer exactly which sources were used, so saying it in the answer only \
wastes their time. Do not begin with phrases like "Based on the context" or \
"This information comes from...". Start with the answer.
5. Be brief and concrete: 2-5 sentences, plain words a farmer can act on. Use a \
short list when describing amounts, eligibility conditions, or required documents.

CONTEXT:
{context}"""

# A chat model can return EMPTY content with finish_reason="length" when it
# spends its whole output budget reasoning (see services/sarvam.py — this is why
# the default model is the non-reasoning `sarvam-105b-conversations`). The retry
# below stays as a safety net for that and for transient truncation; prompt size
# is NOT the driver, so the shrink is a cheap extra roll of the dice, not a fix.
# Context sizes tried in order across generation attempts.
CONTEXT_DOCS_BY_ATTEMPT = (6, 3, 2)
MAX_CONTEXT_DOCS = CONTEXT_DOCS_BY_ATTEMPT[0]

# Sent to the model on a retry: a shorter answer fits the remaining budget.
BREVITY_INSTRUCTION = (
    " IMPORTANT: Reply with at most 2 short sentences. Do not explain your "
    "reasoning. Give only the answer."
)

# A follow-up like "Are there Union Government schemes?" carries its meaning in
# the previous turn, not in its own words. Retrieval sees only the query string,
# so without this the search is done on five ambiguous words and finds nothing.
CONDENSE_PROMPT = """Rewrite the farmer's latest question as a standalone search query about Indian
government agricultural schemes, using the conversation for context.

Rules:
- Keep it under 20 words and in English.
- Resolve references ("it", "that scheme", "there") into explicit names.
- Expand synonyms to the corpus's official wording, e.g.
  "Union Government" -> "Central Government", "central" -> "centrally sponsored".
- **Scope replaces, it does not accumulate.** If the latest question moves to a
  different scope than the previous turn (a different state, or from a state to
  the Central Government), keep ONLY the new scope. Do not carry the old one
  over: "Are there Union Government schemes?" after a Tamil Nadu question
  becomes "Central Government schemes for farmers", NOT "Central Government
  schemes for Tamil Nadu farmers".
- Only borrow from the conversation what is needed to make the question
  self-contained. When in doubt, stay close to the farmer's own words.
- If the question already stands alone, return it unchanged.
- Return ONLY the rewritten query, nothing else.

Conversation so far:
{history}

Latest question: {question}

Standalone search query:"""

# How many times to generate-and-verify before returning the no-answer
# fallback. The gate itself is unchanged — this only decides how many times a
# rejected answer is regenerated. Broad questions ("tell me about sugarcane
# schemes") invite the model to summarise past its sources and get correctly
# rejected; a third attempt, which is strict, usually lands inside the context.
GROUNDING_ATTEMPTS = 3

# ...but bounded by wall clock, not just attempt count. Each attempt is a
# generate + judge round trip, and Sarvam calls run 3-13s, so three attempts can
# reach 30s+ of a farmer watching a spinner. Once the budget is spent the answer
# falls back rather than retrying again: a fast honest "I could not find this"
# beats a slow one. Measured before this cap: 31s / 7 sequential calls.
GENERATION_TIME_BUDGET = 20.0

NEWLINE = chr(10)

_llm_cache: dict[float, BaseChatModel] = {}
_session_histories: dict[str, BaseChatMessageHistory] = {}


def get_llm(temperature: float = 0.2) -> BaseChatModel:
    """Sarvam-105B through the OpenAI-compatible endpoint.

    Cached per temperature — the chain builds several references to the same
    model and there is no reason to construct a client each time.
    """
    if temperature not in _llm_cache:
        from services.sarvam import get_chat_model

        _llm_cache[temperature] = get_chat_model(temperature=temperature)
    return _llm_cache[temperature]


def translate_query_step() -> Runnable:
    """Stage 3 — farmer's language → English for retrieval.

    English passes through untouched. Voice input arrives already translated by
    Saaras ``translate`` mode, so a pre-set ``translated_query`` is respected.
    """

    def translate(payload: dict[str, Any]) -> str:
        if payload.get("translated_query"):
            return payload["translated_query"]

        query, language = payload["query"], payload.get("language", "en")
        if language == "en" or payload.get("blocked"):
            return query

        from services import sarvam

        try:
            return sarvam.translate_text(query, source=language, target="en")
        except Exception as exc:  # noqa: BLE001 - degrade to the raw query
            logger.warning("Query translation failed (%s) — retrieving with raw text", exc)
            return query

    return RunnableLambda(translate).with_config(run_name="translate_query")


def condense_step(llm: BaseChatModel) -> Runnable:
    """Rewrite a follow-up into a standalone retrieval query, using history.

    Memory previously fed only the *generation* prompt, so a follow-up was
    retrieved on its own bare words and returned nothing. This closes that gap
    while staying in LCEL. It costs one extra LLM call, and only when there is
    history to resolve — a first question skips it entirely.
    """
    condense = (
        ChatPromptTemplate.from_template(CONDENSE_PROMPT) | llm | StrOutputParser()
    )

    def run(payload: dict[str, Any]) -> str:
        query = payload.get("translated_query") or payload["query"]
        history = payload.get("history") or []
        if payload.get("blocked") or not history:
            return query

        # Only the last few turns matter for resolving a reference.
        transcript_lines = []
        for message in history[-4:]:
            role = "Farmer" if message.type == "human" else "Assistant"
            transcript_lines.append(f"{role}: {str(message.content)[:200]}")

        try:
            rewritten = condense.invoke(
                {"history": NEWLINE.join(transcript_lines), "question": query}
            ).strip()
        except Exception as exc:  # noqa: BLE001 - degrade to the raw query
            logger.warning("Condense failed (%s) — retrieving with the raw query", exc)
            return query

        # A model that rambles instead of rewriting must not poison retrieval.
        if not rewritten or len(rewritten) > 300:
            return query
        if rewritten != query:
            logger.info("Condensed follow-up: %r -> %r", query, rewritten)
        return rewritten

    return RunnableLambda(run).with_config(run_name="condense_question")


def query_expansion_step(llm: BaseChatModel, config: RetrieverConfig) -> Runnable:
    """Stage 4 — multi-query / HyDE variants.

    Expansion is owned by the retriever itself (``MultiQueryRetriever``,
    RAG-fusion, HyDE all wrap the base retriever inside
    ``retrievers.build_retriever``), so this step only records which expansion
    is active — keeping the toggle visible in the payload and the transcript.
    """
    techniques = [
        name for name in ("multi_query", "rag_fusion", "hyde") if getattr(config, name)
    ]
    return RunnableLambda(lambda _payload: techniques).with_config(run_name="query_expansion")


def retrieval_step(config: RetrieverConfig, llm: BaseChatModel) -> Runnable:
    """Stages 5–6 — hybrid retrieve then cross-encoder rerank.

    The retriever is built **once** per chain, not per query.
    """
    retriever = build_retriever(config, llm=llm)

    def retrieve(payload: dict[str, Any]) -> list[Document]:
        if payload.get("blocked"):
            return []
        query = payload.get("translated_query") or payload["query"]
        try:
            return list(retriever.invoke(query))
        except Exception as exc:  # noqa: BLE001
            # One safety net covers every technique: if an LLM-backed retriever
            # (multi-query, HyDE, RAG-fusion, self-query) or the re-ranker fails,
            # answer from plain dense search rather than giving up on the turn.
            logger.warning("Retriever '%s' failed (%s) — retrying with dense only", config.label, exc)
            try:
                from rag.retrievers import build_dense_retriever, load_vectorstore

                return list(build_dense_retriever(load_vectorstore(), config.top_k).invoke(query))
            except Exception as inner:  # noqa: BLE001 - index itself is unusable
                logger.exception("Dense fallback also failed: %s", inner)
                return []

    return RunnableLambda(retrieve).with_config(run_name="retrieve")


def format_context(docs: list[Document], limit: int | None = None) -> str:
    """Render retrieved chunks into the grounded-answer context block.

    Each chunk carries its scheme, source and chunk ID so the model can cite it
    and ``attach_citations`` can line up with what the model actually saw.
    """
    if not docs:
        return "(no documents retrieved)"

    blocks: list[str] = []
    for index, doc in enumerate(docs[: limit or MAX_CONTEXT_DOCS], start=1):
        # Deliberately NO file path, URL, page number or chunk id here. The model
        # used to echo them back at the farmer ("This information comes from
        # tnau-paddy-schemes") — an internal slug that means nothing to them.
        # Citations are attached programmatically from metadata in
        # `validators.attach_citations`, so the model never needs to cite, and
        # anything it cannot see it cannot repeat.
        scheme = doc.metadata.get("scheme_name")
        header = f"[{index}]" + (f" scheme: {scheme}" if scheme else "")
        blocks.append(f"{header}\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(blocks)


def generation_step(llm: BaseChatModel) -> Runnable:
    """Stage 8 — Sarvam-105B answers **in the farmer's language**, grounded.

    Retries once if the model returns empty content, which happens when its
    internal reasoning consumes the whole output budget (see MAX_CONTEXT_DOCS).
    Raises after the last attempt so ``build_answer_branch`` produces the
    grounded fallback — an empty answer is never shown to a farmer.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ANSWER_SYSTEM_PROMPT),
            MessagesPlaceholder("history", optional=True),
            ("human", "{query}"),
        ]
    )
    generate = prompt | llm | StrOutputParser()

    def run(payload: dict[str, Any]) -> str:
        docs = payload.get("docs") or []
        language_name = LANGUAGE_NAMES.get(payload.get("language", "en"), "English")
        strict = payload.get("strict", False)

        for attempt, doc_limit in enumerate(CONTEXT_DOCS_BY_ATTEMPT, start=1):
            # The language directive is repeated HERE, in the human turn, so it
            # lands *after* the conversation history. In the system prompt alone
            # it loses to recent context: after two English exchanges the model
            # answered a Tamil question in English (measured 1 in 2 runs), while
            # the same question with no history answered in Tamil every time.
            # Whatever the model reads last wins.
            query = f"{payload['query']}\n\n(Answer in {language_name}.)"
            if strict:
                query += (
                    " IMPORTANT: State ONLY facts that appear verbatim in the "
                    "CONTEXT above. Do not add scheme details from memory. If the "
                    "CONTEXT does not answer the question, say so plainly."
                )
            if attempt > 1:
                # Shrink the prompt and ask for less prose, so the answer fits
                # in whatever output budget the model's reasoning leaves behind.
                query = query + BREVITY_INSTRUCTION
            answer = (
                generate.invoke(
                    {
                        "query": query,
                        "context": format_context(docs, limit=doc_limit),
                        "language_name": language_name,
                        "history": payload.get("history", []),
                    }
                )
                or ""
            ).strip()
            if answer:
                if attempt > 1:
                    logger.info("Generation succeeded on attempt %s (%s docs)", attempt, doc_limit)
                return answer
            logger.warning(
                "Empty generation (attempt %s/%s, %s docs, %s) — the model spent its "
                "output budget on reasoning; retrying with a smaller prompt",
                attempt,
                len(CONTEXT_DOCS_BY_ATTEMPT),
                doc_limit,
                language_name,
            )
        raise RuntimeError("Model returned an empty answer after retries")

    return RunnableLambda(run).with_config(run_name="generate")


def fallback_step() -> Runnable:
    """The no-confident-answer branch.

    Returns ``validators.NO_ANSWER_FALLBACK`` in the farmer's language, with no
    citations. Translating a fixed string is one cheap Mayura call; the *answer*
    itself is never generated by the LLM here.
    """

    def fallback(payload: dict[str, Any]) -> dict[str, Any]:
        # A guard that already short-circuited has set its own message.
        if payload.get("blocked"):
            # A guard already chose the message; localize_step translates it.
            return {**payload, "citations": []}
        return {
            **payload,
            "answer": validators.NO_ANSWER_FALLBACK,
            "canned": True,
            "citations": [],
        }

    return RunnableLambda(fallback).with_config(run_name="fallback")


def localize_step() -> Runnable:
    """Translate canned guard/fallback messages into the farmer's language.

    Runs last, so it catches every short-circuit path — weak retrieval, an
    ungrounded answer, an off-topic refusal, a blocked injection. Generated
    answers are already in-language and are left untouched.
    """

    def localize(payload: dict[str, Any]) -> dict[str, Any]:
        language = payload.get("language", "en")
        if not payload.get("canned") or language == "en":
            return payload
        return {**payload, "answer": _localize(payload.get("answer", ""), language)}

    return RunnableLambda(localize).with_config(run_name="localize")


def _localize(text: str, language: str) -> str:
    """Translate a canned English message into the farmer's language."""
    if language == "en":
        return text
    from services import sarvam

    try:
        return sarvam.translate_text(text, source="en", target=language)
    except Exception as exc:  # noqa: BLE001 - showing English beats showing nothing
        logger.warning("Could not localize fallback message: %s", exc)
        return text


def build_answer_branch(llm: BaseChatModel) -> Runnable:
    """``RunnableBranch``: weak retrieval → fallback, else generate and verify.

    This branch is what enforces Golden Rule 6 — no un-grounded answers.

    Generation is **verified here, with one retry**. The groundedness gate is a
    per-turn LLM judgement and it is not perfectly stable: on a question whose
    retrieval scored 0.997, the model would occasionally write a claim the
    context does not support, the gate would (correctly) reject it, and the
    farmer got "I could not find this" even though the right documents were
    right there. Retrying costs one call and usually produces a supported
    answer; only a second rejection falls back. The guarantee is unchanged —
    nothing ungrounded is ever shown — but a recoverable miss no longer looks
    like a missing document.
    """
    generation = generation_step(llm)

    def generate(payload: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        for attempt in range(1, GROUNDING_ATTEMPTS + 1):
            if attempt > 1 and time.monotonic() - started > GENERATION_TIME_BUDGET:
                logger.info(
                    "Grounding budget spent (%.1fs) after %s attempt(s) — falling back",
                    time.monotonic() - started,
                    attempt - 1,
                )
                break
            try:
                answer = generation.invoke(
                    # On a retry, tell the model to hug the context harder.
                    {**payload, "strict": attempt > 1}
                )
            except Exception as exc:  # noqa: BLE001 - LLM outage must not raise at the farmer
                logger.exception("Generation failed: %s", exc)
                return {
                    **payload,
                    "answer": validators.NO_ANSWER_FALLBACK,
                    "canned": True,
                    "generation_failed": True,
                    "retrieval": {**payload.get("retrieval", {}), "fallback": True},
                }

            candidate = {**payload, "answer": answer}
            verified = validators.validate_groundedness(candidate, llm)
            if not verified.get("blocked"):
                return verified

            logger.info(
                "Answer failed the groundedness gate (attempt %s/%s)%s",
                attempt,
                GROUNDING_ATTEMPTS,
                " — regenerating" if attempt < GROUNDING_ATTEMPTS else " — falling back",
            )

        return {
            **payload,
            "answer": validators.NO_ANSWER_FALLBACK,
            "canned": True,
            "blocked": True,
            "citations": [],
            "guards": {**payload.get("guards", {}), "grounded": False},
        }

    return RunnableBranch(
        (validators.is_retrieval_weak, fallback_step()),
        RunnableLambda(generate).with_config(run_name="generate_and_verify"),
    ).with_config(run_name="answer_branch")


def build_chain(config: RetrieverConfig | None = None) -> Runnable:
    """Assemble the full text pipeline (stages 2–10), without memory.

    This is the entrypoint ``eval/run_eval.py`` uses — text-only, no STT/TTS in
    the loop. State accumulates through ``RunnablePassthrough.assign``.
    """
    config = config or RetrieverConfig()
    llm = get_llm()

    return (
        validators.input_guard_chain(llm)
        | RunnablePassthrough.assign(translated_query=translate_query_step())
        | RunnablePassthrough.assign(translated_query=condense_step(llm))
        | RunnablePassthrough.assign(expansion=query_expansion_step(llm, config))
        | RunnablePassthrough.assign(docs=retrieval_step(config, llm))
        | validators.as_runnable(validators.validate_retrieval, "validate_retrieval")
        | build_answer_branch(llm)
        | validators.output_guard_chain(llm)
        | localize_step()
    ).with_config(run_name=f"kisan_sahayak[{config.label}]")


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Per-session ``InMemoryChatMessageHistory`` for the memory wrapper.

    In-process only; the durable record is the JSONL transcript in
    ``services/transcript.py``, not this.
    """
    from langchain_core.chat_history import InMemoryChatMessageHistory

    if session_id not in _session_histories:
        _session_histories[session_id] = InMemoryChatMessageHistory()
    return _session_histories[session_id]


def build_conversational_chain(config: RetrieverConfig | None = None) -> Runnable:
    """Stage 11 — wrap :func:`build_chain` in session memory.

    ``RunnableWithMessageHistory`` keyed by ``session_id`` (techstack.md §9).
    No LangGraph checkpointer.
    """
    from langchain_core.runnables.history import RunnableWithMessageHistory

    return RunnableWithMessageHistory(
        build_chain(config),
        get_session_history,
        input_messages_key="query",
        history_messages_key="history",
        output_messages_key="answer",
    )


def answer(
    query: str,
    *,
    language: str | None = None,
    session_id: str | None = None,
    translated_query: str | None = None,
    config: RetrieverConfig | None = None,
) -> dict[str, Any]:
    """Convenience wrapper: one farmer turn in, one answer payload out.

    Used by ``app.py``. ``translated_query`` lets voice input skip Mayura, since
    Saaras already returned English.
    """
    payload: dict[str, Any] = {"query": query, "session_id": session_id or "default"}
    if language:
        payload["language"] = language
    if translated_query:
        payload["translated_query"] = translated_query

    if session_id:
        chain = build_conversational_chain(config)
        return chain.invoke(payload, config={"configurable": {"session_id": session_id}})
    return build_chain(config).invoke(payload)
