"""HTML loader — local scheme HTML files via ``BSHTMLLoader``.

Responsibility
--------------
Read a locally saved scheme page from ``data/raw/*.html``, strip chrome
(nav/footer/scripts), and emit raw records for ``normalize.py``.

Design notes (requirements.md FR-10; techstack.md §7)
-----------------------------------------------------
* One representative HTML example is enough for the demo.
* This is the **local-file** path. Live pages are captured separately and
  offline by ``scrape_playwright.py`` — nothing here touches the network.
* Corpus is English-only; non-English HTML is flagged, not embedded.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from langchain_core.documents import Document

from ingestion.load_pdf import assert_english
from ingestion.normalize import slugify

logger = logging.getLogger(__name__)

# Page furniture that carries no scheme information.
_BOILERPLATE_TAGS = ("script", "style", "nav", "footer", "header", "noscript", "form")

_BOILERPLATE_LINES = re.compile(
    r"^\s*(skip to (main )?content|cookie|accept all|privacy policy|"
    r"last updated|print this page|share this page|screen reader access|"
    r"about us|contact us|home|top|back|next|previous|copyright.*|"
    r"all rights reserved|site ?map|feedback|disclaimer|terms of use|help)\s*$",
    re.IGNORECASE,
)

# A block whose text is mostly hyperlink is a menu, not content. Menu text is
# poison in a RAG corpus: "ABOUT US CONTACT US State Level Schemes Tamil Nadu"
# scores ~0.99 against "Tamil Nadu schemes" while carrying no answer at all.
_LINK_DENSITY_LIMIT = 0.5
_LINK_BLOCK_MIN_CHARS = 40


def _strip_link_heavy_blocks(soup) -> int:
    """Remove nav/menu blocks, detected by hyperlink density. Returns the count.

    Walks innermost-first so a menu table is removed without taking the page
    body with it.
    """
    removed = 0
    for block in reversed(soup.find_all(["table", "ul", "ol", "div"])):
        if block.decomposed:
            continue
        text = " ".join(block.get_text(separator=" ").split())
        if len(text) < _LINK_BLOCK_MIN_CHARS:
            continue
        link_text = " ".join(
            " ".join(a.get_text(separator=" ").split()) for a in block.find_all("a")
        )
        if len(link_text) / max(len(text), 1) > _LINK_DENSITY_LIMIT:
            block.decompose()
            removed += 1
    return removed


def html_doc_id(path: str | Path) -> str:
    """Derive a stable ``doc_id`` slug from the HTML filename."""
    return slugify(Path(path).stem)


def clean_html_text(raw_text: str) -> str:
    """Collapse whitespace and drop boilerplate from extracted HTML text.

    Removes nav/cookie-banner residue and normalizes whitespace so chunk
    boundaries land on real content rather than on runs of blank lines.
    """
    # Source HTML is full of runs of spaces from table markup ("Replacement  of
    # old Pumpsets"). Collapse them so chunks tokenize the way a reader expects.
    lines = [" ".join(line.split()) for line in raw_text.splitlines()]
    kept = [
        line
        for line in lines
        # Single-word leftovers are almost always menu items.
        if line and not _BOILERPLATE_LINES.match(line) and len(line.split()) > 1
    ]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def extract_headings(path: str | Path) -> list[str]:
    """Pull the heading outline (h1..h3) from the page, in document order.

    ``build_index`` uses these to place section boundaries for ``parent_id``
    (faiss-vectorstore.md §2).
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(Path(path).read_text(encoding="utf-8", errors="ignore"), "lxml")
    return [
        heading.get_text(strip=True)
        for heading in soup.find_all(["h1", "h2", "h3"])
        if heading.get_text(strip=True)
    ]


def load_html(path: str | Path) -> list[Document]:
    """Load one local HTML file into ``Document`` form."""
    from bs4 import BeautifulSoup

    path = Path(path)
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "lxml")
    for tag in soup(list(_BOILERPLATE_TAGS)):
        tag.decompose()

    removed = _strip_link_heavy_blocks(soup)
    if removed:
        logger.info("Dropped %s link-heavy (menu) block(s) from %s", removed, path.name)

    text = clean_html_text(soup.get_text(separator="\n"))
    if not text:
        logger.warning("No usable text in %s — skipping", path.name)
        return []

    title = soup.title.get_text(strip=True) if soup.title else path.stem
    doc = Document(
        page_content=text,
        metadata={
            "doc_id": html_doc_id(path),
            "source_path": str(path),
            "section_title": title,
        },
    )
    try:
        assert_english(doc)
    except ValueError as exc:
        logger.warning("%s", exc)
        return []

    logger.info("Loaded %s (%s chars) from %s", doc.metadata["doc_id"], len(text), path.name)
    return [doc]


# Browsers save pages as .htm as often as .html — accept both, or a saved page
# is silently skipped at build time with no error.
HTML_SUFFIXES = ("*.html", "*.htm")


def load_html_dir(dir_path: str | Path) -> list[Document]:
    """Batch-load every ``*.html`` / ``*.htm`` file directly under ``dir_path``.

    Not recursive: a browser "save page" drops a sibling ``<name>_files/``
    directory of CSS and images that must not be ingested as documents.
    """
    root = Path(dir_path)
    paths = sorted({path for pattern in HTML_SUFFIXES for path in root.glob(pattern)})
    docs: list[Document] = []
    for html in paths:
        docs.extend(load_html(html))
    return docs
