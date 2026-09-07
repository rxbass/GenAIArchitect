"""Retrievers — hybrid, multi-query, RAG-fusion, HyDE, self-query, parent-doc.

Responsibility
--------------
Every retrieval technique in ``requirements.md`` §6, each built from a
**LangChain-native** component (techstack.md §4) and each **toggleable**, so
``eval/run_eval.py`` can measure baseline → +hybrid → +rerank → +query-expansion
deltas (CLAUDE.md §When Adding a RAG Technique).

:func:`build_retriever` is the **single factory** both ``rag/chain.py`` and the
eval harness call. Configurations differ only by ``RetrieverConfig`` flags, so
each accuracy delta is attributable to exactly one technique.

Technique → component map
-------------------------
    dense              FAISS ``IndexFlatIP`` ``.as_retriever()``
    lexical            ``BM25Retriever`` (rank-bm25)
    hybrid             ``EnsembleRetriever([dense, bm25], weights=[0.6, 0.4])``
    multi-query        ``MultiQueryRetriever`` (Sarvam-105B generates variants)
    RAG-fusion         custom LCEL step — reciprocal rank fusion over variants
    HyDE               LCEL chain — hypothetical answer → embed → dense search
    self-query         ``SelfQueryRetriever`` over the common metadata schema
    parent-document    ``ParentDocumentRetriever`` — embed child, return section
    compression+rerank ``ContextualCompressionRetriever`` + cross-encoder

No LangGraph. Everything composes into ``rag/chain.py`` via LCEL.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda

# LangChain 1.x moved the classic retrievers into `langchain_classic`; 0.x had
# them under `langchain.retrievers`. This project runs on 1.x, so that path is
# tried FIRST and the 0.x path is the fallback — the imports a reader sees first
# are then the ones that actually execute here.
#
# The 0.x branch is unresolvable on a 1.x install, which is correct but makes
# type checkers report a missing import for a line that never runs. The
# suppressions below silence that; they do not hide a real error. Verify with:
#   python -c "import rag.retrievers as R; print(R.EnsembleRetriever.__module__)"
try:  # pragma: no cover - import shim
    from langchain_classic.retrievers import (
        ContextualCompressionRetriever,
        EnsembleRetriever,
        ParentDocumentRetriever,
    )
    from langchain_classic.retrievers.multi_query import MultiQueryRetriever
except ImportError:  # pragma: no cover - LangChain 0.x
    from langchain.retrievers import (  # type: ignore[import-not-found]  # noqa: I001
        ContextualCompressionRetriever,
        EnsembleRetriever,
        ParentDocumentRetriever,
    )
    from langchain.retrievers.multi_query import (  # type: ignore[import-not-found]
        MultiQueryRetriever,
    )

if TYPE_CHECKING:  # pragma: no cover - typing only
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)
# Paths are anchored to the project root, not the working directory, so the
# app and the CLIs work no matter where they are launched from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


DEFAULT_TOP_K = 10
RERANK_TOP_N = 5
HYBRID_WEIGHTS = (0.6, 0.4)  # dense, lexical — faiss-vectorstore.md §6
RRF_K = 60  # reciprocal-rank-fusion smoothing constant
N_QUERY_VARIANTS = 3

FAISS_INDEX_DIR = str(PROJECT_ROOT / "data" / "processed" / "faiss_index")
DOCUMENTS_JSONL = str(PROJECT_ROOT / "data" / "processed" / "documents.jsonl")
PARENTS_JSONL = str(PROJECT_ROOT / "data" / "processed" / "parents.jsonl")

# Metadata fields the self-query retriever is allowed to filter on. These are
# exactly the schema fields designed for it (faiss-vectorstore.md §1, §7).
SELF_QUERY_FIELDS = (
    ("scheme_name", "string", "Name of the scheme, e.g. 'PM-KISAN' or 'Kisan Credit Card'"),
    ("scheme_category", "string", "One of: income-support, credit, insurance, subsidy"),
    ("state", "string", "State the scheme applies to, or 'all-india'"),
    ("source_type", "string", "One of: pdf, html, web"),
)

_VECTORSTORE: Any = None
_CHUNKS: list[Document] | None = None
_PARENTS: list[Document] | None = None


@dataclass
class RetrieverConfig:
    """Toggle set describing one retrieval configuration.

    Each flag maps to one row of the README accuracy table, so the eval harness
    can construct: baseline (dense only) → +hybrid → +rerank → +query-expansion.
    """

    top_k: int = DEFAULT_TOP_K
    rerank_top_n: int = RERANK_TOP_N
    hybrid: bool = False
    multi_query: bool = False
    rag_fusion: bool = False
    hyde: bool = False
    self_query: bool = False
    parent_document: bool = False
    rerank: bool = False

    def __post_init__(self) -> None:
        if self.rag_fusion and self.multi_query:
            raise ValueError(
                "rag_fusion and multi_query both expand the query — enable one, so the "
                "eval can attribute the delta to a single technique."
            )
        if self.self_query and self.parent_document:
            raise ValueError("self_query and parent_document both replace the base retriever")
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")

    @property
    def label(self) -> str:
        """Short human name for results tables and logs."""
        parts = [name for name, on in asdict(self).items() if on is True]
        return "+".join(parts) if parts else "baseline"


def load_vectorstore(embeddings: Embeddings | None = None):
    """Load the persisted FAISS store from ``data/processed/faiss_index``.

    Cached at module level — app startup and the eval loop must not re-read the
    index per query. ``allow_dangerous_deserialization`` is safe here: the index
    is local and built by us (faiss-vectorstore.md §5).
    """
    global _VECTORSTORE
    if _VECTORSTORE is not None:
        return _VECTORSTORE

    from langchain_community.vectorstores import FAISS

    from ingestion.build_index import get_embeddings

    if not Path(FAISS_INDEX_DIR).exists():
        raise FileNotFoundError(
            f"{FAISS_INDEX_DIR} not found. Build the index first: "
            "python ingestion/build_index.py"
        )

    _VECTORSTORE = FAISS.load_local(
        FAISS_INDEX_DIR,
        embeddings or get_embeddings(),
        allow_dangerous_deserialization=True,
    )
    logger.info("Loaded FAISS index (%s vectors)", _VECTORSTORE.index.ntotal)
    return _VECTORSTORE


def load_chunks() -> list[Document]:
    """Read ``data/processed/documents.jsonl`` — the BM25 / parent-doc corpus.

    Delegates to ``ingestion.normalize`` so BM25 and FAISS always see the same
    chunk IDs.
    """
    global _CHUNKS
    if _CHUNKS is None:
        from ingestion.normalize import read_documents_jsonl

        _CHUNKS = read_documents_jsonl(DOCUMENTS_JSONL)
        logger.info("Loaded %s chunk(s) for BM25", len(_CHUNKS))
    return _CHUNKS


def load_parents() -> list[Document]:
    """Read the parent sections written by ``build_index``."""
    global _PARENTS
    if _PARENTS is None:
        from ingestion.normalize import read_documents_jsonl

        path = PARENTS_JSONL if Path(PARENTS_JSONL).exists() else DOCUMENTS_JSONL
        _PARENTS = read_documents_jsonl(path)
    return _PARENTS


def build_dense_retriever(vectorstore, top_k: int = DEFAULT_TOP_K) -> BaseRetriever:
    """Baseline: naive dense top-k over FAISS."""
    return vectorstore.as_retriever(search_kwargs={"k": top_k})


def build_bm25_retriever(chunks: list[Document], top_k: int = DEFAULT_TOP_K) -> BaseRetriever:
    """Lexical arm — exact-match recall that dense embeddings miss."""
    from langchain_community.retrievers import BM25Retriever

    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = top_k
    return retriever


def build_hybrid_retriever(
    dense: BaseRetriever,
    bm25: BaseRetriever,
    weights: tuple[float, float] = HYBRID_WEIGHTS,
) -> BaseRetriever:
    """Fuse dense ∪ lexical with ``EnsembleRetriever``."""
    return EnsembleRetriever(retrievers=[dense, bm25], weights=list(weights))


def build_multi_query_retriever(base: BaseRetriever, llm: BaseChatModel) -> BaseRetriever:
    """Paraphrase coverage — ``MultiQueryRetriever`` driven by Sarvam-105B.

    The prompt deliberately does **not** hand the model a list of scheme jargon.
    An earlier version told it to "use official scheme terminology (eligibility,
    beneficiary, subsidy, installment, land holding)" and it dutifully stuffed
    all five words into every variant — "What are the eligibility criteria for a
    farmer beneficiary to avail the Kisan Credit Card subsidy?" — which dragged
    retrieval off target and cost 16 points of hit rate in the eval. Variants
    must stay short and preserve the farmer's own specifics.
    """
    from langchain_core.prompts import PromptTemplate

    prompt = PromptTemplate(
        input_variables=["question"],
        template=(
            "You are helping search a database of Indian government agricultural "
            "scheme documents. Write {n} alternative phrasings of the farmer's "
            "question.\n\n"
            "Rules:\n"
            "- Keep each variant SHORT and about the same single fact.\n"
            "- Preserve every specific term the farmer used: scheme name, crop, "
            "state, machine, amount.\n"
            "- Do NOT add words the question did not imply. Padding a variant "
            "with scheme jargon makes the search worse, not better.\n"
            "- One per line, no numbering.\n\nQuestion: {{question}}"
        ).format(n=N_QUERY_VARIANTS),
    )
    return MultiQueryRetriever.from_llm(retriever=base, llm=llm, prompt=prompt)


def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]], k: int = RRF_K
) -> list[Document]:
    """Fuse several ranked result lists into one via RRF (``1 / (k + rank)``).

    Scored by ``chunk_id`` so the same chunk surfacing in several arms collapses
    into one entry with an accumulated score.
    """
    scores: dict[str, float] = {}
    documents: dict[str, Document] = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            key = doc.metadata.get("chunk_id") or doc.page_content[:120]
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            documents.setdefault(key, doc)

    ordered = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [documents[key] for key in ordered]


def build_rag_fusion_retriever(base: BaseRetriever, llm: BaseChatModel) -> BaseRetriever:
    """RAG-fusion — generate query variants, retrieve each, fuse by RRF.

    Built as a ``RunnableLambda`` so it drops into the LCEL chain unchanged, and
    fans the variants out with ``.batch`` for concurrency.
    """
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    variant_chain = (
        ChatPromptTemplate.from_template(
            "Generate {n} alternative search queries for the following farmer "
            "question about Indian government agricultural schemes. Use official "
            "scheme terminology. One query per line, no numbering.\n\n"
            "Question: {{question}}".format(n=N_QUERY_VARIANTS)
        )
        | llm
        | StrOutputParser()
    )

    def fuse(query: str) -> list[Document]:
        raw = variant_chain.invoke({"question": query})
        variants = [line.strip() for line in raw.splitlines() if line.strip()]
        # Always retrieve for the original query too, so fusion can only add.
        queries = [query, *variants[:N_QUERY_VARIANTS]]
        return reciprocal_rank_fusion(base.batch(queries))

    return RunnableLambda(fuse).with_config(run_name="rag_fusion")


def build_hyde_retriever(
    base: BaseRetriever, llm: BaseChatModel, embeddings: Embeddings
) -> BaseRetriever:
    """HyDE — draft a hypothetical scheme answer, embed *that*, then search.

    Helps on sparse or awkwardly-phrased farmer queries, where the question
    shares little vocabulary with the source document.
    """
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    hypothesis_chain = (
        ChatPromptTemplate.from_template(
            "Write a short factual paragraph, in the style of an official Indian "
            "government scheme document, that would answer this farmer's question. "
            "Invent plausible specifics if needed — this text is used only to "
            "search for real documents, never shown to anyone.\n\n"
            "Question: {question}"
        )
        | llm
        | StrOutputParser()
    )

    def retrieve(query: str) -> list[Document]:
        hypothetical = hypothesis_chain.invoke({"question": query})
        # Search with the hypothetical document, not the raw question.
        return base.invoke(hypothetical)

    return RunnableLambda(retrieve).with_config(run_name="hyde")


FILTER_PROMPT = """\
Extract metadata filters from a farmer's question about Indian agricultural
schemes. Reply with ONLY a JSON object, no other text.

Allowed keys (omit a key entirely when the question does not constrain it):
  "scheme_name"     one of: {schemes}
  "scheme_category" one of: {categories}
  "state"           one of: {states}

Rules:
- Use a key ONLY when the question clearly names that scheme, category or state.
- A general question returns {{}}.
- Never invent a value outside the lists above.

Question: {question}

JSON:"""


def _filter_vocabulary(chunks: list[Document]) -> dict[str, list[str]]:
    """The metadata values that actually exist in the corpus.

    Offering the model only real values is what keeps the filter from silently
    matching nothing — a filter for a scheme that was never ingested returns
    zero documents and looks exactly like "we have no answer".
    """
    vocabulary: dict[str, set[str]] = {"scheme_name": set(), "scheme_category": set(), "state": set()}
    for chunk in chunks:
        for field in vocabulary:
            value = chunk.metadata.get(field)
            if value:
                vocabulary[field].add(str(value))
    return {field: sorted(values) for field, values in vocabulary.items()}


def build_self_query_retriever(vectorstore, llm: BaseChatModel, top_k: int = DEFAULT_TOP_K):
    """Metadata filtering — "only PM-KISAN", "only Odisha irrigation schemes".

    Implemented directly against FAISS's ``filter`` rather than LangChain's
    ``SelfQueryRetriever``: that class is unusable in this environment, because
    ``langchain_classic`` imports ``DatabricksVectorSearch`` from
    ``langchain_community``, which no longer exports it. This keeps the
    technique — an LLM parses the question into a metadata filter — with one
    fewer broken dependency, and it only ever filters on values the corpus
    actually contains.

    Falls back to unfiltered dense search whenever the filter cannot be parsed
    or would return nothing.
    """
    import json

    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    vocabulary = _filter_vocabulary(load_chunks())
    extract = ChatPromptTemplate.from_template(FILTER_PROMPT) | llm | StrOutputParser()

    def retrieve(query: str) -> list[Document]:
        filters: dict[str, str] = {}
        try:
            raw = extract.invoke(
                {
                    "question": query,
                    "schemes": ", ".join(vocabulary["scheme_name"]) or "(none)",
                    "categories": ", ".join(vocabulary["scheme_category"]) or "(none)",
                    "states": ", ".join(vocabulary["state"]) or "(none)",
                }
            )
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
                filters = {
                    field: value
                    for field, value in parsed.items()
                    if field in vocabulary and value in vocabulary[field]
                }
        except Exception as exc:  # noqa: BLE001 - filtering is an optimisation
            logger.warning("Self-query filter extraction failed (%s) — unfiltered", exc)

        if filters:
            logger.info("Self-query filter: %s", filters)
            docs = vectorstore.similarity_search(query, k=top_k, filter=filters)
            if docs:
                return docs
            logger.info("Filter %s matched nothing — retrying unfiltered", filters)

        return vectorstore.similarity_search(query, k=top_k)

    return RunnableLambda(retrieve).with_config(run_name="self_query")


def build_parent_document_retriever(vectorstore, chunks: list[Document]) -> BaseRetriever:
    """Embed small child chunks, return the full parent section to the LLM.

    Implemented directly against the ``parent_id`` metadata rather than through
    ``ParentDocumentRetriever``'s own splitter, because ``build_index`` already
    wrote the child/parent split and the chunk IDs must stay stable.
    """
    parents_by_id = {
        parent.metadata["parent_id"]: parent
        for parent in load_parents()
        if parent.metadata.get("parent_id")
    }
    dense = build_dense_retriever(vectorstore, DEFAULT_TOP_K)

    def retrieve(query: str) -> list[Document]:
        seen: set[str] = set()
        expanded: list[Document] = []
        for child in dense.invoke(query):
            parent_id = child.metadata.get("parent_id")
            parent = parents_by_id.get(parent_id) if parent_id else None
            key = parent_id or child.metadata.get("chunk_id", "")
            if key in seen:
                continue
            seen.add(key)
            # Keep the child's chunk_id on the returned parent so citations and
            # the eval's chunk-ID comparison still line up.
            if parent is not None:
                expanded.append(
                    Document(
                        page_content=parent.page_content,
                        metadata={**parent.metadata, "chunk_id": child.metadata.get("chunk_id")},
                    )
                )
            else:
                expanded.append(child)
        return expanded

    return RunnableLambda(retrieve).with_config(run_name="parent_document")


def build_compression_retriever(base: BaseRetriever, top_n: int = RERANK_TOP_N) -> BaseRetriever:
    """Wrap a retriever in ``ContextualCompressionRetriever`` + cross-encoder."""
    from rag.rerank import build_reranker_compressor

    if not isinstance(base, BaseRetriever):
        # HyDE / RAG-fusion / parent-doc are Runnables, not BaseRetrievers.
        # Rerank them inline instead of through the compression wrapper.
        from rag.rerank import rerank

        def rerank_runnable(query: str) -> list[Document]:
            return rerank(query, base.invoke(query), top_n=top_n)

        return RunnableLambda(rerank_runnable).with_config(run_name="rerank")

    return ContextualCompressionRetriever(
        base_retriever=base,
        base_compressor=build_reranker_compressor(top_n=top_n),
    )


def build_retriever(
    config: RetrieverConfig | None = None,
    llm: BaseChatModel | None = None,
    embeddings: Embeddings | None = None,
):
    """Assemble the retriever stack described by ``config``.

    The single factory the chain and the eval harness both call — the accuracy
    table is produced by varying ``config`` alone. Returns something invokable
    with a query string that yields ``list[Document]`` (a ``BaseRetriever`` or
    an equivalent ``Runnable``).

    Layering order: dense → (hybrid) → (self-query | parent-doc | expansion) →
    (rerank).
    """
    config = config or RetrieverConfig()
    vectorstore = load_vectorstore(embeddings)

    if llm is None and (config.multi_query or config.rag_fusion or config.hyde or config.self_query):
        from rag.chain import get_llm

        llm = get_llm()

    # ── base arm ──
    if config.self_query:
        retriever = build_self_query_retriever(vectorstore, llm, config.top_k)
    elif config.parent_document:
        retriever = build_parent_document_retriever(vectorstore, load_chunks())
    else:
        retriever = build_dense_retriever(vectorstore, config.top_k)
        if config.hybrid:
            retriever = build_hybrid_retriever(
                retriever, build_bm25_retriever(load_chunks(), config.top_k)
            )

    # ── query expansion ──
    if config.multi_query:
        retriever = build_multi_query_retriever(retriever, llm)
    elif config.rag_fusion:
        retriever = build_rag_fusion_retriever(retriever, llm)

    if config.hyde:
        retriever = build_hyde_retriever(retriever, llm, embeddings or vectorstore.embeddings)

    # ── rerank ──
    if config.rerank:
        retriever = build_compression_retriever(retriever, config.rerank_top_n)

    logger.debug("Built retriever: %s", config.label)
    return retriever


def reset_caches() -> None:
    """Drop the cached index/corpus — used by the eval harness between builds."""
    global _VECTORSTORE, _CHUNKS, _PARENTS
    _VECTORSTORE = _CHUNKS = _PARENTS = None
