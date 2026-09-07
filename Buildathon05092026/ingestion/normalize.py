"""Normalizer — every source becomes the one common ``Document`` schema.

Responsibility
--------------
This is the **single place** where the metadata contract lives. PDF, HTML, and
Playwright records all land here and leave as ``Document`` objects carrying the
exact fields specified in ``faiss-vectorstore.md`` §1:

    doc_id, chunk_id, parent_id, source_type, source_path, source_url,
    scheme_name, scheme_category, state, department, language,
    page_number, ingested_at

Conventions (CLAUDE.md §Conventions)
------------------------------------
* Do **not** add source-specific metadata fields in the loaders — extend the
  schema here, in one place, so the rest of the pipeline stays source-agnostic.
* ``chunk_id`` is ``"{doc_id}::{NNNN}"`` and must be **stable**:
  ``eval/golden_set.jsonl`` references these IDs, so re-chunking invalidates the
  golden set.
* ``parent_id`` is ``"{doc_id}::sec-{n}"`` — the section a child chunk belongs
  to, used by the parent-document retriever.
* ``language`` is always ``"en"``; the corpus is curated English-only.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from langchain_core.documents import Document

SourceType = Literal["pdf", "html", "web"]

# The authoritative metadata field list. Every emitted Document carries all of
# these keys (value may be None) so downstream filtering never KeyErrors.
SCHEMA_FIELDS: tuple[str, ...] = (
    "doc_id",
    "chunk_id",
    "parent_id",
    "source_type",
    "source_path",
    "source_url",
    "scheme_name",
    "scheme_category",
    "state",
    "department",
    "language",
    "page_number",
    "ingested_at",
)

CORPUS_LANGUAGE = "en"
VALID_SOURCE_TYPES: frozenset[str] = frozenset(("pdf", "html", "web"))

# ── Scheme detection ─────────────────────────────────────────────────────────
#
# `scheme_name` / `scheme_category` / `state` are what the self-query retriever
# filters on (faiss-vectorstore.md §7), and what stops the model reaching for a
# filename when it wants to name a scheme. A single source document covers many
# schemes, so a per-file mapping is not enough — these are detected per section.
#
# Deterministic and ₹0 on purpose: an LLM pass over 1,400 chunks would cost real
# money and be unreproducible, and these names are unambiguous enough to match.
# Ordered most-specific first; the first match wins.
SCHEME_PATTERNS: tuple[tuple[str, str, str], ...] = (
    ("PM-KISAN", "income-support", r"PM[\s\-]?KISAN|Kisan Samman Nidhi"),
    ("Pradhan Mantri Fasal Bima Yojana", "insurance", r"PMFBY|Fasal Bima"),
    ("Pradhan Mantri Krishi Sinchayee Yojana", "irrigation", r"PMKSY|Krishi Sinchayee"),
    ("Rashtriya Krishi Vikas Yojana", "development", r"RKVY|Rashtriya Krishi Vikas"),
    ("National Food Security Mission", "food-security", r"NFSM|National Food Security Mission"),
    ("National Agricultural Insurance Scheme", "insurance", r"NAIS|National Agricultural Insurance"),
    ("Kisan Credit Card", "credit", r"Kisan Credit Card|\bKCCS?\b"),
    ("Soil Health Card", "soil-health", r"Soil Health Card"),
    ("Odisha Millets Mission", "food-security", r"Odisha Millets Mission|Millets Mission"),
    ("National Horticulture Mission", "horticulture", r"National Horticulture Mission|\bNHM\b"),
    ("Krishonnati Yojana", "development", r"Krishonnati Yojana"),
    ("Minikit Programme for Rice", "seeds", r"Minikit"),
    ("Seed Village Programme", "seeds", r"Seed Village"),
    ("Minimum Support Price", "marketing", r"Minimum Support Price|\bMSP\b"),
    ("Gramin Bhandaran Yojana", "marketing", r"Gramin Bhandaran"),
    ("Pledge Loan Scheme", "credit", r"Pledge Loan"),
    ("Agricultural Mechanisation Programme", "mechanisation", r"Mechani[sz]ation Programme"),
    ("Integrated Pest Management", "crop-protection", r"Integrated Pest Management|\bIPM\b"),
    ("Land Development Scheme", "land-development", r"Land Development Scheme"),
    ("Crop Insurance Scheme", "insurance", r"Crop Insurance Scheme"),
)

_COMPILED_SCHEMES = tuple(
    (name, category, re.compile(pattern, re.IGNORECASE))
    for name, category, pattern in SCHEME_PATTERNS
)

# States the corpus actually covers, as slugs (faiss-vectorstore.md §1).
STATE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("odisha", r"Odisha|Orissa"),
    ("tamil-nadu", r"Tamil\s?Nadu"),
    ("kerala", r"Kerala"),
    ("karnataka", r"Karnataka"),
)

_COMPILED_STATES = tuple(
    (slug, re.compile(pattern, re.IGNORECASE)) for slug, pattern in STATE_PATTERNS
)

DEFAULT_STATE = "all-india"


def detect_scheme(text: str) -> tuple[str | None, str | None]:
    """Return ``(scheme_name, scheme_category)`` for a passage, or ``(None, None)``."""
    for name, category, pattern in _COMPILED_SCHEMES:
        if pattern.search(text):
            return name, category
    return None, None


def detect_state(text: str) -> str | None:
    """Return the state slug a passage is about, or ``None`` if it does not say."""
    for slug, pattern in _COMPILED_STATES:
        if pattern.search(text):
            return slug
    return None


# How many consecutive scheme-less sections may inherit the previous section's
# scheme. Scheme documents run "heading, then pages of detail", so a short
# carry-forward captures the detail pages. It is deliberately short: carrying
# indefinitely would bleed one scheme's name across an entire document, and a
# WRONG label is worse than none — self-query filters on it and citations show it.
SCHEME_CARRY_FORWARD = 2


def enrich_parents(parents: list[Document]) -> int:
    """Resolve scheme metadata for each section, with a short carry-forward.

    Returns how many sections ended up with a scheme name.
    """
    tagged = 0
    last: dict[str, tuple[str | None, str | None, str | None, int]] = {}

    for parent in parents:
        doc_id = parent.metadata["doc_id"]
        scheme, category = detect_scheme(parent.page_content)
        state = detect_state(parent.page_content)

        prev_scheme, prev_category, prev_state, remaining = last.get(
            doc_id, (None, None, None, 0)
        )
        if scheme:
            remaining = SCHEME_CARRY_FORWARD
        elif remaining > 0:
            scheme, category, remaining = prev_scheme, prev_category, remaining - 1

        # State carries forward under the SAME budget as the scheme. Unbounded,
        # a single "Odisha" mention on page 3 labelled the entire 296-page
        # document — including its central schemes — as an Odisha scheme.
        # Document-wide truths belong in data/raw/scheme_map.json, not in a
        # heuristic that never resets.
        if not state and remaining > 0:
            state = prev_state
        parent.metadata["scheme_name"] = scheme
        parent.metadata["scheme_category"] = category
        if parent.metadata.get("state") in (None, DEFAULT_STATE):
            parent.metadata["state"] = state or DEFAULT_STATE

        last[doc_id] = (scheme, category, state, remaining)
        tagged += bool(scheme)
    return tagged


def enrich_scheme_metadata(docs: list[Document], inherit_from: Document | None = None) -> int:
    """Fill ``scheme_name`` / ``scheme_category`` / ``state`` in place.

    A chunk that continues a section often does not repeat the scheme name, so
    ``inherit_from`` (its parent section) supplies it. An explicit mention in the
    chunk itself always wins over the inherited value.

    Returns how many documents ended up with a scheme name.
    """
    tagged = 0
    parent_scheme = parent_category = parent_state = None
    if inherit_from is not None:
        # Use the parent's resolved metadata (enrich_parents already applied
        # carry-forward) rather than re-detecting from its text.
        parent_scheme = inherit_from.metadata.get("scheme_name")
        parent_category = inherit_from.metadata.get("scheme_category")
        parent_state = inherit_from.metadata.get("state")

    for doc in docs:
        scheme, category = detect_scheme(doc.page_content)
        state = detect_state(doc.page_content)

        doc.metadata["scheme_name"] = scheme or parent_scheme or doc.metadata.get("scheme_name")
        doc.metadata["scheme_category"] = (
            category or parent_category or doc.metadata.get("scheme_category")
        )
        # Never overwrite a state the loader set deliberately with a guess.
        if doc.metadata.get("state") in (None, DEFAULT_STATE):
            doc.metadata["state"] = state or parent_state or DEFAULT_STATE

        if doc.metadata.get("scheme_name"):
            tagged += 1
    return tagged

# Fields the loaders may set that are not part of the schema but are useful
# while chunking. They are dropped before the Document is emitted.
_TRANSIENT_FIELDS: frozenset[str] = frozenset(("section_title", "total_pages"))


def _utc_now() -> str:
    """ISO-8601 UTC timestamp for ``ingested_at``."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def slugify(value: str) -> str:
    """Lowercase, hyphenated, filesystem- and ID-safe slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
    return re.sub(r"-{2,}", "-", slug) or "untitled"


def build_metadata(
    *,
    doc_id: str,
    source_type: SourceType,
    source_path: str | None = None,
    source_url: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Construct one fully-populated metadata dict covering ``SCHEMA_FIELDS``.

    Unknown keys are rejected so the schema stays closed — a typo'd field name
    is a build-time error, not a silently unfilterable document.
    """
    if source_type not in VALID_SOURCE_TYPES:
        raise ValueError(f"source_type must be one of {sorted(VALID_SOURCE_TYPES)}, got {source_type!r}")

    unknown = set(extra) - set(SCHEMA_FIELDS) - _TRANSIENT_FIELDS
    if unknown:
        raise ValueError(
            f"Unknown metadata field(s) {sorted(unknown)}. Extend SCHEMA_FIELDS in "
            "ingestion/normalize.py rather than adding source-specific fields."
        )

    metadata: dict[str, Any] = dict.fromkeys(SCHEMA_FIELDS)
    metadata.update(
        {
            "doc_id": doc_id,
            "source_type": source_type,
            "source_path": str(source_path) if source_path else None,
            "source_url": source_url,
            "language": CORPUS_LANGUAGE,
            "ingested_at": _utc_now(),
            "state": "all-india",
        }
    )
    metadata.update({k: v for k, v in extra.items() if v is not None})
    return metadata


def make_chunk_id(doc_id: str, index: int) -> str:
    """Return the stable chunk ID ``"{doc_id}::{index:04d}"``.

    This format is referenced by ``eval/golden_set.jsonl`` — do not change it
    without re-labelling the golden set.
    """
    return f"{doc_id}::{index:04d}"


def make_parent_id(doc_id: str, section_index: int) -> str:
    """Return the section-level parent ID ``"{doc_id}::sec-{section_index}"``."""
    return f"{doc_id}::sec-{section_index}"


def validate_metadata(metadata: dict[str, Any]) -> None:
    """Assert a metadata dict satisfies the common schema."""
    missing = [f for f in SCHEMA_FIELDS if f not in metadata]
    if missing:
        raise ValueError(f"Metadata missing required field(s): {missing}")

    if metadata["language"] != CORPUS_LANGUAGE:
        raise ValueError(
            f"Corpus is English-only; got language={metadata['language']!r} "
            f"for {metadata.get('doc_id')}"
        )
    if metadata["source_type"] not in VALID_SOURCE_TYPES:
        raise ValueError(f"Invalid source_type: {metadata['source_type']!r}")
    if not metadata["doc_id"]:
        raise ValueError("doc_id is required")

    chunk_id = metadata.get("chunk_id")
    if chunk_id and not str(chunk_id).startswith(f"{metadata['doc_id']}::"):
        raise ValueError(f"chunk_id {chunk_id!r} does not belong to doc_id {metadata['doc_id']!r}")


def _normalize_batch(
    raw_docs: list[Document],
    source_type: SourceType,
    **overrides: Any,
) -> list[Document]:
    """Shared lift: raw loader output → schema-complete Documents.

    Chunk IDs are *not* assigned here — ``build_index`` assigns them after
    chunking, since one raw page becomes several child chunks.
    """
    normalized: list[Document] = []
    for raw in raw_docs:
        raw_meta = dict(raw.metadata)
        doc_id = overrides.get("doc_id") or raw_meta.get("doc_id")
        if not doc_id:
            raise ValueError("doc_id must be supplied by the loader or as an override")

        # Carry every schema field the loader already filled in — notably the
        # scheme_name/category/state/department hints that power self-query.
        # chunk_id is assigned after chunking, so it is never carried here.
        carried = {
            key: raw_meta[key]
            for key in (*SCHEMA_FIELDS, *_TRANSIENT_FIELDS)
            if key not in ("doc_id", "chunk_id") and raw_meta.get(key) is not None
        }
        carried.update({k: v for k, v in overrides.items() if k != "doc_id"})

        metadata = build_metadata(
            doc_id=doc_id,
            source_type=source_type,
            source_path=carried.pop("source_path", None) or raw_meta.get("source"),
            source_url=carried.pop("source_url", None),
            **carried,
        )
        validate_metadata(metadata)
        normalized.append(Document(page_content=raw.page_content, metadata=metadata))
    return normalized


def normalize_pdf_docs(raw_docs: list[Document], **overrides: Any) -> list[Document]:
    """Lift PyMuPDF page records into the common schema (``source_type="pdf"``)."""
    return _normalize_batch(raw_docs, "pdf", **overrides)


def normalize_html_docs(raw_docs: list[Document], **overrides: Any) -> list[Document]:
    """Lift local-HTML records into the common schema (``source_type="html"``)."""
    return _normalize_batch(raw_docs, "html", **overrides)


def normalize_web_record(record: dict[str, Any], **overrides: Any) -> list[Document]:
    """Lift a Playwright scrape record into the common schema (``source_type="web"``).

    ``source_url`` comes from the scrape; ``source_path`` points at the saved
    ``data/raw/<slug>.json`` so the corpus stays reproducible offline.
    """
    doc_id = overrides.pop("doc_id", None) or record.get("doc_id") or slugify(record.get("slug", "web-doc"))
    # A page <title> is not a scheme name. "Agriculture Schemes in India:
    # Features, Benefits & UPSC Notes" was ending up in `scheme_name` and then
    # in citations. Accept the scraped title only if it names a known scheme.
    title = record.get("scheme_name")
    recognised, _ = detect_scheme(title) if title else (None, None)
    metadata_overrides = {
        "scheme_name": recognised,
        "department": record.get("department"),
        **{k: v for k, v in overrides.items() if v is not None},
    }
    raw = Document(
        page_content=record.get("text", ""),
        metadata={
            "doc_id": doc_id,
            "source_url": record.get("source_url") or record.get("url"),
            "source_path": record.get("saved_path"),
            **{k: v for k, v in metadata_overrides.items() if v is not None},
        },
    )
    return _normalize_batch([raw], "web", doc_id=doc_id)


def write_documents_jsonl(docs: list[Document], path: str | Path) -> None:
    """Write normalized chunks to ``data/processed/documents.jsonl``.

    One JSON object per line ``{page_content, metadata}``. BM25 and the
    golden-set labelling both read this file, so it must stay in lockstep with
    the FAISS index (faiss-vectorstore.md §5).
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for doc in docs:
            record = {"page_content": doc.page_content, "metadata": doc.metadata}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_documents_jsonl(path: str | Path) -> list[Document]:
    """Read ``documents.jsonl`` back into ``Document`` objects.

    Used by ``build_index.py`` (rebuild) and by BM25 at app startup.
    """
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(
            f"{src} not found. Build the corpus first: python ingestion/build_index.py"
        )
    docs: list[Document] = []
    with src.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            docs.append(
                Document(page_content=record["page_content"], metadata=record["metadata"])
            )
    return docs
