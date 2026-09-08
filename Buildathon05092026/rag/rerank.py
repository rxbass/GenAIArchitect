"""Re-ranker — ``bge-reranker-v2-m3`` cross-encoder, local on GPU.

Responsibility
--------------
Reorder the hybrid retriever's candidates by true query-document relevance and
trim to the top-n handed to Sarvam-105B. Exposed as a LangChain
``BaseDocumentCompressor`` so it slots into ``ContextualCompressionRetriever``.

Design notes (techstack.md §5; NFR-3)
-------------------------------------
* Runs **locally** on the RTX 3050 6GB in FP16, CPU fallback. The LLM and voice
  are API-side, so the GPU is otherwise idle.
* Robust on imperfectly-translated queries — which matters, since farmer queries
  reach retrieval after Mayura/Saaras translation.
* Model is loaded **once** and cached; never per query.
* The rerank score also feeds the retrieval-relevance gate in
  ``rag/validators.py`` — a weak top score triggers the no-answer fallback.
"""

from __future__ import annotations

import logging
import math
import os
from typing import TYPE_CHECKING, Any, Sequence

from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor

if TYPE_CHECKING:  # pragma: no cover - typing only
    from langchain_core.callbacks import Callbacks

logger = logging.getLogger(__name__)

# Two rerankers, chosen by what the machine can actually run.
#
# bge-reranker-v2-m3 (568M params) is the quality choice and what techstack.md
# specifies — but it is a GPU model. Measured on this CPU-only box: ~1,500 ms
# PER DOCUMENT, i.e. ~29s to rerank 19 candidates. Unusable interactively.
#
# ms-marco-MiniLM-L6-v2 (22M params) scores the same 19 candidates in 0.56s
# (30 ms/doc, ~50x faster) and picked the identical top document in testing.
# It is English-only, which is fine here: the farmer's query is translated to
# English before retrieval and the corpus is English by design.
GPU_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
CPU_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
RERANKER_MODEL = GPU_RERANKER_MODEL  # kept for callers that reference it

DEFAULT_TOP_N = 5
BATCH_SIZE = 32

# Rerank is O(candidates). Multi-query can hand us 45+ documents; scoring all of
# them to return 5 is wasted work, and the tail is rarely relevant anyway.
RERANK_CANDIDATE_LIMIT = 20

# Below this rerank score, the best hit is considered off-topic; the chain
# returns the no-confident-answer fallback (requirements.md FR-16).
#
# Calibrated against measured separation, not intuition:
#   off-topic queries ("who won the cricket world cup")  -> 0.0000
#   short relevant queries                               -> 0.85 - 0.99
#   VERBOSE relevant queries                             -> 0.25 - 0.50
# A conversational question ("Sugarcane, what are the government schemes and
# subsidies needed for sugarcane farmers? Please tell me a little about them")
# scored 0.246 and was refused at the old 0.3 threshold, even though the corpus
# answers it. Verbosity dilutes a cross-encoder score; it does not mean the
# corpus is silent. 0.10 still leaves a wide margin over the 0.0000 floor, and
# the groundedness gate remains the real backstop against a bad answer.
RELEVANCE_THRESHOLD = 0.10

_RERANKER: Any = None
_RERANKER_NAME: str | None = None


def _sigmoid(value: float) -> float:
    """Squash a raw cross-encoder logit into 0..1 so the threshold is meaningful."""
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _select_model_and_device() -> tuple[str, str, bool]:
    """Pick the reranker that suits the available hardware.

    Returns ``(model_name, device, use_fp16)``. An explicit ``RERANKER_MODEL``
    env var always wins, so a GPU box can be forced onto either model.
    """
    device, use_fp16 = "cpu", False
    try:
        import torch

        if torch.cuda.is_available():
            device, use_fp16 = "cuda", True
    except ImportError:  # pragma: no cover - torch ships with the reranker
        pass

    override = os.environ.get("RERANKER_MODEL", "").strip()
    default = GPU_RERANKER_MODEL if device == "cuda" else CPU_RERANKER_MODEL
    return override or default, device, use_fp16


def load_reranker(model_name: str | None = None):
    """Load and cache the cross-encoder for this machine.

    Memoized at module level so Streamlit reruns and eval loops reuse one
    instance rather than reloading the weights per query.
    """
    global _RERANKER, _RERANKER_NAME

    selected, device, use_fp16 = _select_model_and_device()
    model_name = model_name or selected
    if _RERANKER is not None and _RERANKER_NAME == model_name:
        return _RERANKER

    from sentence_transformers import CrossEncoder

    logger.info("Loading reranker %s on %s (fp16=%s)", model_name, device, use_fp16)
    if device == "cpu" and model_name == GPU_RERANKER_MODEL:
        logger.warning(
            "%s is a GPU model running on CPU — expect ~1.5s per document. "
            "Install a CUDA build of torch, or unset RERANKER_MODEL to use %s.",
            model_name,
            CPU_RERANKER_MODEL,
        )

    _RERANKER = CrossEncoder(model_name, device=device, max_length=512)
    if use_fp16:
        try:
            _RERANKER.model.half()
        except Exception as exc:  # noqa: BLE001 - fp32 on GPU is still fast
            logger.warning("Could not enable fp16 (%s) — continuing in fp32", exc)
    _RERANKER_NAME = model_name
    return _RERANKER


def score_pairs(query: str, docs: Sequence[Document]) -> list[float]:
    """Return one relevance score per (query, document) pair, normalized to 0..1.

    Raw cross-encoder logits are squashed through a sigmoid so
    ``RELEVANCE_THRESHOLD`` means the same thing whichever model is loaded.
    Measured separation on this corpus: relevant queries score 0.85-0.99,
    off-topic queries score 0.0000.
    """
    if not docs:
        return []

    pairs = [[query, doc.page_content] for doc in docs]
    raw = load_reranker().predict(pairs, batch_size=BATCH_SIZE, show_progress_bar=False)
    return [_sigmoid(float(score)) for score in raw]


def rerank(query: str, docs: Sequence[Document], top_n: int = DEFAULT_TOP_N) -> list[Document]:
    """Score, sort descending, and return the top-n documents.

    Each score is stashed in ``doc.metadata["rerank_score"]`` so the validators
    and the UI citation panel can read it without re-scoring.
    """
    if not docs:
        return []

    # Cap the candidate set before scoring — this is the dominant cost.
    docs = list(docs)[:RERANK_CANDIDATE_LIMIT]

    try:
        scores = score_pairs(query, docs)
    except (NameError, AttributeError, TypeError):
        # A bug in this module, not an environment failure. Never silently
        # degrade on these — a broken reranker that "works" is worse than a
        # crash, because it looks fine while doing nothing.
        raise
    except Exception as exc:  # noqa: BLE001 - model OOM/download failure
        # Degrade to the retriever's own ordering rather than to no answer. The
        # documents are still relevant; only the reordering is lost.
        logger.warning("Re-ranking failed (%s) — keeping retrieval order", exc)
        return list(docs)[:top_n]

    scored = [
        (score, Document(page_content=doc.page_content, metadata={**doc.metadata, "rerank_score": score}))
        for score, doc in zip(scores, docs)
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [doc for _, doc in scored[:top_n]]


class BgeRerankerCompressor(BaseDocumentCompressor):
    """``BaseDocumentCompressor`` wrapping :func:`rerank`.

    This is what ``ContextualCompressionRetriever`` consumes in
    ``rag.retrievers.build_compression_retriever``.
    """

    top_n: int = DEFAULT_TOP_N

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        """Reorder and trim the retrieved candidates for one query."""
        return rerank(query, documents, top_n=self.top_n)


def build_reranker_compressor(top_n: int = DEFAULT_TOP_N) -> BgeRerankerCompressor:
    """Return the compressor used by ``ContextualCompressionRetriever``."""
    return BgeRerankerCompressor(top_n=top_n)


def top_score(docs: Sequence[Document]) -> float:
    """Read the best ``rerank_score`` off an already-reranked list.

    Returns 0.0 for an empty list. Documents that never passed through the
    reranker have no score; they are treated as 0.0 rather than assumed good.
    """
    if not docs:
        return 0.0
    return max(float(doc.metadata.get("rerank_score", 0.0)) for doc in docs)
