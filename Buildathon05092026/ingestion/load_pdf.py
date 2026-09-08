"""PDF loader — PyMuPDF text extraction with a Sarvam Vision OCR fallback.

Responsibility
--------------
Turn a scheme PDF in ``data/raw/`` into raw per-page records that
``normalize.py`` can lift into the common ``Document`` schema
(``faiss-vectorstore.md`` §1).

Design notes (see requirements.md FR-10, FR-13; techstack.md §7)
---------------------------------------------------------------
* ``PyMuPDFLoader`` is the primary path — fast, keeps ``page_number``.
* A page that yields little or no text is treated as **scanned** and routed
  through Sarvam Vision (Document AI) OCR via ``services.sarvam``. OCR is a
  one-time, build-time cost — it is never invoked at query time.
* Corpus is **English-only** by design. A non-English PDF is flagged and
  skipped, not embedded (CLAUDE.md Golden Rule 3).
* This module emits *raw* page records only; it does not chunk and does not
  touch the FAISS index.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from ingestion.normalize import slugify

logger = logging.getLogger(__name__)
# Paths are anchored to the project root, not the working directory, so the
# app and the CLIs work no matter where they are launched from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Below this many characters on a page, we assume the page is scanned image-only
# and fall back to OCR.
OCR_TEXT_THRESHOLD_CHARS = 50

# Hand-maintained scheme metadata, keyed by doc_id. Keeping it beside the corpus
# (rather than in code) lets the corpus grow without edits here.
SCHEME_MAP_PATH = str(PROJECT_ROOT / "data" / "raw" / "scheme_map.json")

# Cheap English detection: if a page is mostly non-Latin, it isn't our corpus.
_LATIN_RATIO_THRESHOLD = 0.5


def pdf_doc_id(path: str | Path) -> str:
    """Derive a stable ``doc_id`` slug from the PDF filename.

    Stability matters — ``eval/golden_set.jsonl`` references chunk IDs built
    from this.
    """
    return slugify(Path(path).stem)


def is_scanned_page(page: Document) -> bool:
    """Return ``True`` when a page carries too little text to be digital-native."""
    return len(page.page_content.strip()) < OCR_TEXT_THRESHOLD_CHARS


def flatten_ocr_markup(text: str) -> str:
    """Turn Document AI's HTML tables into plain, embeddable text.

    Sarvam returns tables as HTML even in markdown mode. Left as-is, **59% of an
    OCR'd chunk is `<tr>`/`<td>` markup** — embedding budget spent on syntax, so
    a page titled "Model Estimate for cultivation of 1 Ha. of Potato" lost to a
    page about subsidies for the query "cost of cultivating one hectare of
    potato". Rows become "cell | cell | cell" lines, which read naturally and
    embed on their content.
    """
    if "<" not in text:
        return text

    from bs4 import BeautifulSoup

    soup = BeautifulSoup(text, "lxml")

    for table in soup.find_all("table"):
        lines = []
        for row in table.find_all("tr"):
            cells = [
                " ".join(cell.get_text(separator=" ").split())
                for cell in row.find_all(["th", "td"])
            ]
            cells = [c for c in cells if c]
            if cells:
                lines.append(" | ".join(cells))
        table.replace_with("\n".join(lines) + "\n" if lines else "")

    flattened = soup.get_text(separator="\n")
    # Collapse the blank lines the replacement leaves behind.
    flattened = re.sub(r"[ \t]+", " ", flattened)
    return re.sub(r"\n{3,}", "\n\n", flattened).strip()


def ocr_page(path: str | Path, page_number: int) -> str:
    """OCR one PDF page through Sarvam Vision (Document AI). Build-time only.

    Extracts the single page into an in-memory one-page PDF first — uploading the
    whole source document once per scanned page would mean hundreds of megabytes
    on a large corpus.
    """
    from services import sarvam

    try:
        import pymupdf

        source = pymupdf.open(path)
        try:
            single = pymupdf.open()
            single.insert_pdf(source, from_page=page_number - 1, to_page=page_number - 1)
            payload = single.tobytes()
            single.close()
        finally:
            source.close()
    except Exception as exc:  # noqa: BLE001 - fall back to sending the whole file
        logger.warning("Could not slice page %s of %s (%s)", page_number, Path(path).name, exc)
        payload = None

    try:
        if payload is not None:
            raw = sarvam.ocr_document(
                payload, page_number=1, filename=f"{Path(path).stem}-p{page_number}.pdf"
            )
        else:
            raw = sarvam.ocr_document(path, page_number=page_number)
        return flatten_ocr_markup(raw)
    except Exception as exc:  # noqa: BLE001 - a failed OCR must not kill the build
        logger.warning("OCR failed for %s page %s: %s", Path(path).name, page_number, exc)
        return ""


def assert_english(doc: Document) -> None:
    """Guard the English-only corpus rule.

    Raises ``ValueError`` for content that is predominantly non-Latin, so Indic
    text never reaches the OpenAI embedder.
    """
    text = doc.page_content.strip()
    if len(text) < OCR_TEXT_THRESHOLD_CHARS:
        return  # too short to judge; the caller drops empty pages anyway

    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return
    latin = sum(1 for ch in letters if ch.isascii())
    if latin / len(letters) < _LATIN_RATIO_THRESHOLD:
        raise ValueError(
            f"{doc.metadata.get('doc_id')} page {doc.metadata.get('page_number')} "
            "appears non-English. The corpus is curated English-only — skipping."
        )


def _scheme_hints(path: str | Path) -> dict[str, Any]:
    """Best-effort scheme metadata (name/category/state/department) for a file.

    Read from ``data/raw/scheme_map.json`` so ``normalize`` can enrich the
    metadata that powers self-query filtering. Missing entries are fine — the
    fields simply stay ``None``.
    """
    doc_id = pdf_doc_id(path)
    map_path = Path(SCHEME_MAP_PATH)
    if not map_path.exists():
        return {}
    try:
        mapping = json.loads(map_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning("Could not parse %s: %s", map_path, exc)
        return {}

    entry = mapping.get(doc_id, {})
    allowed = ("scheme_name", "scheme_category", "state", "department")
    return {k: entry[k] for k in allowed if entry.get(k)}


def load_pdf(path: str | Path) -> list[Document]:
    """Load a single PDF into one ``Document`` per page.

    Pages below ``OCR_TEXT_THRESHOLD_CHARS`` are re-read through Sarvam Vision.
    Empty and non-English pages are dropped with a warning.
    """
    from langchain_community.document_loaders import PyMuPDFLoader

    path = Path(path)
    doc_id = pdf_doc_id(path)
    hints = _scheme_hints(path)

    pages = PyMuPDFLoader(str(path)).load()
    loaded: list[Document] = []

    for index, page in enumerate(pages):
        # PyMuPDF pages are 0-indexed; our citations are 1-indexed.
        page_number = int(page.metadata.get("page", index)) + 1
        text = page.page_content.strip()

        if len(text) < OCR_TEXT_THRESHOLD_CHARS:
            logger.info("Page %s of %s looks scanned — running OCR", page_number, path.name)
            text = ocr_page(path, page_number).strip() or text

        if not text:
            logger.warning("Dropping empty page %s of %s", page_number, path.name)
            continue

        doc = Document(
            page_content=text,
            metadata={
                "doc_id": doc_id,
                "source_path": str(path),
                "page_number": page_number,
                **hints,
            },
        )
        try:
            assert_english(doc)
        except ValueError as exc:
            logger.warning("%s", exc)
            continue
        loaded.append(doc)

    logger.info("Loaded %s page(s) from %s", len(loaded), path.name)
    return loaded


def load_pdf_dir(dir_path: str | Path) -> list[Document]:
    """Batch-load every ``*.pdf`` under ``dir_path`` (default ``data/raw/``)."""
    docs: list[Document] = []
    for pdf in sorted(Path(dir_path).glob("*.pdf")):
        docs.extend(load_pdf(pdf))
    return docs
