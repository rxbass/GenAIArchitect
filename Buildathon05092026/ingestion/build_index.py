"""Index builder — chunk → embed → FAISS. Run **once, offline**, before launch.

Responsibility
--------------
The one-shot ingestion entrypoint. Loads every source in ``data/raw/``,
normalizes it, applies child/parent chunking, embeds child chunks with OpenAI
``text-embedding-3-small`` (L2-normalized), builds an ``IndexFlatIP`` FAISS
store, and persists everything to ``data/processed/``.

Artifacts written (faiss-vectorstore.md §5)
-------------------------------------------
    data/processed/documents.jsonl      normalized child chunks (BM25 + golden set)
    data/processed/parents.jsonl        parent sections (parent-doc retrieval)
    data/processed/faiss_index/
        index.faiss                     the vectors
        index.pkl                       docstore + id map

Hard rules
----------
* Offline only — Playwright output is read from ``data/raw/``, never scraped
  here (CLAUDE.md Golden Rule 4).
* Corpus is English-only; embeddings are OpenAI ``text-embedding-3-small``.
* Chunk IDs must be stable — changing chunking invalidates
  ``eval/golden_set.jsonl``.

Usage
-----
    python ingestion/build_index.py [--raw-dir data/raw] [--from-jsonl]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

# Run as a script (`python ingestion/build_index.py`) as well as an import.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402
from langchain_core.documents import Document  # noqa: E402

from ingestion import load_html, load_pdf, normalize, scrape_playwright  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover - typing only
    from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = str(PROJECT_ROOT / "data" / "raw")
PROCESSED_DIR = str(PROJECT_ROOT / "data" / "processed")
DOCUMENTS_JSONL = str(PROJECT_ROOT / "data" / "processed" / "documents.jsonl")
PARENTS_JSONL = str(PROJECT_ROOT / "data" / "processed" / "parents.jsonl")
FAISS_INDEX_DIR = str(PROJECT_ROOT / "data" / "processed" / "faiss_index")

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

# faiss-vectorstore.md §2 — child chunks are embedded, parent sections are returned.
CHILD_CHUNK_SIZE = 500
CHILD_CHUNK_OVERLAP = 80
CHILD_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# A "section" is a heading-ish line: short, title-cased or numbered, no full stop.
PARENT_MAX_CHARS = 4000
_HEADING_MAX_WORDS = 12

# Government scheme PDFs are table-heavy, and table cells extract as very short
# lines. Without a floor, every cell would look like a heading and the corpus
# would shred into 20-character fragments that cannot ground an answer.
MIN_PARENT_CHARS = 500

# A chunk shorter than this cannot support a grounded answer — it is a page
# number, a stray table cell, or a wrapped word. Dropped before embedding.
MIN_CHUNK_CHARS = 120


def load_all_sources(raw_dir: str | Path = RAW_DIR) -> list[Document]:
    """Load every raw source (PDF batch, HTML file, saved web scrape).

    Each source is normalized to the common schema before being returned, so
    downstream stages never branch on ``source_type``.
    """
    raw_path = Path(raw_dir)
    docs: list[Document] = []

    pdf_docs = load_pdf.load_pdf_dir(raw_path)
    if pdf_docs:
        docs.extend(normalize.normalize_pdf_docs(pdf_docs))
    logger.info("PDF   : %s page(s)", len(pdf_docs))

    html_docs = load_html.load_html_dir(raw_path)
    if html_docs:
        docs.extend(normalize.normalize_html_docs(html_docs))
    logger.info("HTML  : %s doc(s)", len(html_docs))

    web_count = 0
    for json_path in sorted(raw_path.glob("*.json")):
        if json_path.name == Path(load_pdf.SCHEME_MAP_PATH).name:
            continue  # metadata sidecar, not a scraped document
        record = scrape_playwright.load_scraped(json_path)
        if not record.get("text"):
            logger.warning("Skipping %s — no text field", json_path.name)
            continue
        docs.extend(normalize.normalize_web_record(record))
        web_count += 1
    logger.info("Web   : %s scraped doc(s)", web_count)

    if not docs:
        raise RuntimeError(
            f"No source documents found in {raw_path}/. Add PDFs/HTML, or run "
            "python ingestion/scrape_playwright.py <url> first."
        )

    _assert_unique_doc_ids(docs)
    return docs


def _assert_unique_doc_ids(docs: list[Document]) -> None:
    """Two different source files must never share a ``doc_id``.

    ``SCHEMES.pdf`` and ``Schemes.htm`` both slugify to ``schemes``. Left
    unchecked they merge into one identity: citations become ambiguous, and
    self-query filtering cannot tell the documents apart. Fail loudly at build
    time — this is a corpus-naming mistake, not something to paper over.
    """
    owners: dict[str, set[str]] = {}
    for doc in docs:
        source = doc.metadata.get("source_path") or doc.metadata.get("source_url") or "?"
        owners.setdefault(doc.metadata["doc_id"], set()).add(str(source))

    clashes = {doc_id: sources for doc_id, sources in owners.items() if len(sources) > 1}
    if clashes:
        lines = [
            f"  doc_id {doc_id!r} claimed by: {', '.join(sorted(sources))}"
            for doc_id, sources in sorted(clashes.items())
        ]
        raise RuntimeError(
            "Two source files resolve to the same doc_id, which would merge them "
            "into one document and make citations ambiguous:"
            + chr(10)
            + chr(10).join(lines)
            + chr(10)
            + "Rename one of the files so their stems differ."
        )


def _looks_like_heading(line: str) -> bool:
    """Heuristic section boundary: a short, non-sentence, multi-word line.

    Deliberately conservative. A false positive costs more than a false
    negative: splitting mid-table produces fragments, whereas missing a heading
    just yields a slightly larger section, which ``PARENT_MAX_CHARS`` caps.
    """
    stripped = line.strip()
    if not stripped or len(stripped) < 8:
        return False

    words = stripped.split()
    if len(words) < 2 or len(words) > _HEADING_MAX_WORDS:
        return False
    # Pure numbers / page furniture are never headings.
    if all(not ch.isalpha() for ch in stripped):
        return False
    if stripped.endswith((".", ",", ";")):
        return False
    if stripped.endswith(":"):
        return True
    return stripped.isupper() or stripped.istitle() or bool(
        stripped[0].isdigit() and "." in stripped[:4]
    )


def split_parents(docs: list[Document]) -> list[Document]:
    """Split normalized docs into parent (section-level) units.

    Semantic boundaries first — headings and scheme sub-sections — with each
    parent capped at ``PARENT_MAX_CHARS`` so a heading-less document still
    yields usable sections. Each parent gets a ``parent_id``.
    """
    parents: list[Document] = []

    for doc in docs:
        doc_id = doc.metadata["doc_id"]
        section_index = _next_section_index(parents, doc_id)
        buffer: list[str] = []

        def flush(index: int, *, _doc: Document = doc, _doc_id: str = doc_id) -> int:
            text = "\n".join(buffer).strip()
            if not text:
                return index
            metadata = dict(_doc.metadata)
            metadata["parent_id"] = normalize.make_parent_id(_doc_id, index)
            # section_title is a transient loader field, not part of the schema.
            metadata.pop("section_title", None)
            parents.append(Document(page_content=text, metadata=metadata))
            buffer.clear()
            return index + 1

        for line in doc.page_content.splitlines():
            buffered = sum(len(item) for item in buffer)
            # Only honour a heading once the current section has real content,
            # otherwise a run of short table lines becomes a run of fragments.
            starting_new = buffered >= MIN_PARENT_CHARS and _looks_like_heading(line)
            too_long = buffered > PARENT_MAX_CHARS
            if starting_new or too_long:
                section_index = flush(section_index)
            buffer.append(line)

        section_index = flush(section_index)

    tagged = normalize.enrich_parents(parents)
    logger.info(
        "Parents: %s section(s); scheme_name resolved on %s (%s%%)",
        len(parents), tagged, 100 * tagged // max(len(parents), 1),
    )
    return parents


def _next_section_index(parents: list[Document], doc_id: str) -> int:
    """Continue section numbering across pages of the same document."""
    return sum(1 for parent in parents if parent.metadata["doc_id"] == doc_id)


def split_children(parents: list[Document]) -> list[Document]:
    """Split parents into embedded child chunks (~500 chars / 80 overlap).

    Metadata (including ``scheme_name`` and ``parent_id``) is carried into every
    child so self-query filtering survives chunking. Chunk IDs are assigned
    per ``doc_id`` in document order, which is what makes them stable.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
        separators=CHILD_SEPARATORS,
    )

    children: list[Document] = []
    counters: dict[str, int] = {}
    dropped = 0
    tagged = 0

    for parent in parents:
        doc_id = parent.metadata["doc_id"]
        for piece in splitter.split_text(parent.page_content):
            text = piece.strip()
            # Fragments below MIN_CHUNK_CHARS cannot ground an answer and only
            # add noise to both dense and BM25 retrieval.
            if len(text) < MIN_CHUNK_CHARS:
                dropped += 1
                continue
            index = counters.get(doc_id, 0)
            counters[doc_id] = index + 1

            metadata = dict(parent.metadata)
            metadata["chunk_id"] = normalize.make_chunk_id(doc_id, index)
            normalize.validate_metadata(metadata)
            child = Document(page_content=text, metadata=metadata)
            # Scheme metadata is resolved per chunk, inheriting from the section
            # it came from when the chunk itself does not name the scheme.
            tagged += normalize.enrich_scheme_metadata([child], inherit_from=parent)
            children.append(child)

    logger.info(
        "Children: %s chunk(s) (dropped %s under %s chars); scheme_name set on %s (%s%%)",
        len(children), dropped, MIN_CHUNK_CHARS, tagged,
        100 * tagged // max(len(children), 1),
    )
    return children


def get_embeddings() -> Embeddings:
    """Build the OpenAI ``text-embedding-3-small`` embedder (1536-dim).

    The *only* embedder in the project. Keys come from ``.env`` — never
    hardcoded (CLAUDE.md §Conventions).
    """
    from langchain_openai import OpenAIEmbeddings

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key "
            "(embeddings only — the LLM is Sarvam)."
        )
    return OpenAIEmbeddings(
        model=os.environ.get("EMBEDDING_MODEL", EMBEDDING_MODEL),
        api_key=api_key,
    )


def build_faiss(child_docs: list[Document], embeddings: Embeddings):
    """Embed child chunks and build the ``IndexFlatIP`` FAISS store.

    ``MAX_INNER_PRODUCT`` selects ``IndexFlatIP``; ``normalize_L2=True`` makes
    inner product equal cosine similarity (faiss-vectorstore.md §3–4).
    """
    from langchain_community.vectorstores import FAISS
    from langchain_community.vectorstores.utils import DistanceStrategy

    logger.info("Embedding %s chunk(s) with %s…", len(child_docs), EMBEDDING_MODEL)
    return FAISS.from_documents(
        child_docs,
        embeddings,
        normalize_L2=True,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )


def persist(vectorstore, child_docs: list[Document], parent_docs: list[Document]) -> None:
    """Save the FAISS index, the child ``documents.jsonl``, and the parent sections."""
    Path(PROCESSED_DIR).mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(FAISS_INDEX_DIR)
    normalize.write_documents_jsonl(child_docs, DOCUMENTS_JSONL)
    normalize.write_documents_jsonl(parent_docs, PARENTS_JSONL)
    logger.info("Saved index → %s", FAISS_INDEX_DIR)


def verify_build(child_docs: list[Document], vectorstore) -> None:
    """Run the build checklist from faiss-vectorstore.md §8."""
    chunk_ids = [doc.metadata["chunk_id"] for doc in child_docs]
    duplicates = len(chunk_ids) - len(set(chunk_ids))
    if duplicates:
        raise RuntimeError(f"{duplicates} duplicate chunk_id(s) — IDs must be unique and stable")

    if vectorstore.index.ntotal != len(child_docs):
        raise RuntimeError(
            f"Index holds {vectorstore.index.ntotal} vectors but there are "
            f"{len(child_docs)} chunks"
        )
    if vectorstore.index.d != EMBEDDING_DIMENSIONS:
        raise RuntimeError(f"Expected {EMBEDDING_DIMENSIONS}-dim vectors, got {vectorstore.index.d}")

    for doc in child_docs:
        normalize.validate_metadata(doc.metadata)

    _warn_on_golden_drift(set(chunk_ids))
    logger.info("Verified: %s unique chunks, %s vectors", len(chunk_ids), vectorstore.index.ntotal)


def _warn_on_golden_drift(chunk_ids: set[str]) -> None:
    """Warn when the golden set references chunk IDs this build no longer produces."""
    golden = PROJECT_ROOT / "eval" / "golden_set.jsonl"
    if not golden.exists() or not golden.stat().st_size:
        return

    missing: set[str] = set()
    with golden.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            missing |= {cid for cid in item.get("relevant_chunk_ids", []) if cid not in chunk_ids}

    if missing:
        logger.warning(
            "%s golden chunk_id(s) no longer exist after this build (e.g. %s). "
            "Re-label eval/golden_set.jsonl.",
            len(missing),
            sorted(missing)[:3],
        )


def main() -> None:
    """CLI entrypoint: load → normalize → chunk → embed → FAISS → persist."""
    parser = argparse.ArgumentParser(description="Build the FAISS index (offline, run once).")
    parser.add_argument("--raw-dir", default=RAW_DIR, help=f"Source directory (default: {RAW_DIR})")
    parser.add_argument(
        "--from-jsonl",
        action="store_true",
        help="Re-embed from data/processed/documents.jsonl instead of re-reading data/raw/",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    if args.from_jsonl:
        children = normalize.read_documents_jsonl(DOCUMENTS_JSONL)
        parents = (
            normalize.read_documents_jsonl(PARENTS_JSONL)
            if Path(PARENTS_JSONL).exists()
            else children
        )
        logger.info("Rebuilding from %s (%s chunks)", DOCUMENTS_JSONL, len(children))
    else:
        docs = load_all_sources(args.raw_dir)
        parents = split_parents(docs)
        children = split_children(parents)

    vectorstore = build_faiss(children, get_embeddings())
    verify_build(children, vectorstore)
    persist(vectorstore, children, parents)

    print(
        f"\nBuilt {vectorstore.index.ntotal} vectors from {len(children)} chunks "
        f"across {len({d.metadata['doc_id'] for d in children})} document(s)."
        f"\nNext: streamlit run app.py"
    )


if __name__ == "__main__":
    main()
