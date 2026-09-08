"""Playwright web scraper — scheme page → file. **Build-time only.**

Responsibility
--------------
Fetch one scheme web page with headless Chromium, extract its main content,
and write it to ``data/raw/<slug>.json``. A later ingestion pass reads that
file; the live URL is never fetched again.

Hard rule (CLAUDE.md Golden Rule 4 / requirements.md FR-11)
-----------------------------------------------------------
**No realtime scraping.** This module is executed manually, once, before the
index is built. Nothing in ``app.py`` or ``rag/`` may import it at query time.

Usage
-----
    python ingestion/scrape_playwright.py <url> [--slug my-scheme]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# Run as a script (`python ingestion/scrape_playwright.py`) as well as an import.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.load_html import clean_html_text  # noqa: E402
from ingestion.normalize import slugify  # noqa: E402

logger = logging.getLogger(__name__)
# Paths are anchored to the project root, not the working directory, so the
# app and the CLIs work no matter where they are launched from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


RAW_DIR = str(PROJECT_ROOT / "data" / "raw")
NAV_TIMEOUT_MS = 30_000

# Tried in order — the first selector that yields real text wins.
_CONTENT_SELECTORS = ("main", "article", "[role=main]", "#content", ".content", "body")
MIN_CONTENT_CHARS = 150

# A scraped page must carry real content. Anything shorter than this is a block
# page, a redirect stub, or a JS shell — never a scheme document.
MIN_USABLE_CHARS = 400

# Signatures of a refusal or error page. Saving one would poison the corpus:
# a farmer could be shown "Access Denied" as retrieved scheme context.
_BLOCK_SIGNATURES = (
    "access denied",
    "you don't have permission to access",
    "403 forbidden",
    "404 not found",
    "page not found",
    "request unsuccessful",
    "are you a robot",
    "captcha",
    "enable javascript to continue",
    "your request has been blocked",
)
_STRIP_TAGS = ("script", "style", "nav", "footer", "header", "noscript", "form", "aside")


def slugify_url(url: str) -> str:
    """Derive a stable filename slug (and ``doc_id``) from a URL."""
    parsed = urlparse(url)
    return slugify(f"{parsed.netloc}{parsed.path}".rstrip("/"))


def extract_main_content(html: str) -> str:
    """Reduce a full page to its article/main body text."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    for tag in soup(list(_STRIP_TAGS)):
        tag.decompose()

    # Take the first selector that yields substantial text, else the longest
    # candidate. Never fall back to the whole soup — that pulls in <head>/<title>.
    best = ""
    for selector in _CONTENT_SELECTORS:
        node = soup.select_one(selector)
        if not node:
            continue
        text = clean_html_text(node.get_text(separator="\n"))
        if len(text) >= MIN_CONTENT_CHARS:
            return text
        if len(text) > len(best):
            best = text
    return best


def extract_metadata(url: str, html: str) -> dict[str, Any]:
    """Capture page metadata for the common schema."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    title = soup.title.get_text(strip=True) if soup.title else ""
    h1 = soup.find("h1")

    description = ""
    meta = soup.find("meta", attrs={"name": "description"})
    if meta and meta.get("content"):
        description = meta["content"].strip()

    return {
        "title": title,
        # The h1 is a better scheme name than the title, which often carries
        # portal branding ("… | National Portal of India").
        "scheme_name": (h1.get_text(strip=True) if h1 else title.split("|")[0].strip()) or None,
        "description": description,
        "source_url": url,
    }


def scrape(url: str, slug: str | None = None) -> dict[str, Any]:
    """Scrape one page and return the raw record (title, text, url, timestamp)."""
    from playwright.sync_api import sync_playwright

    logger.info("Scraping %s", url)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.goto(url, wait_until="networkidle", timeout=NAV_TIMEOUT_MS)
            html = page.content()
        finally:
            browser.close()

    record = extract_metadata(url, html)
    record.update(
        {
            "slug": slug or slugify_url(url),
            "doc_id": slug or slugify_url(url),
            "text": extract_main_content(html),
            "scraped_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    )
    verify_scraped(record, url)
    logger.info("Extracted %s chars from %s", len(record["text"]), url)
    return record


def verify_scraped(record: dict[str, Any], url: str) -> None:
    """Refuse to accept a block page, error page, or empty shell as content.

    The corpus is what the farmer's answers are grounded in — an "Access Denied"
    body embedded as a scheme document is worse than having no document at all.
    """
    text = record.get("text") or ""
    haystack = f"{record.get('title') or ''} {text[:1000]}".lower()

    hit = next((sig for sig in _BLOCK_SIGNATURES if sig in haystack), None)
    if hit:
        raise RuntimeError(
            f"{url} returned a block/error page (matched {hit!r}). "
            "The site refused the request. Save the page from your browser into "
            "data/raw/<name>.html and ingest it as a local HTML source instead."
        )
    if len(text) < MIN_USABLE_CHARS:
        raise RuntimeError(
            f"{url} yielded only {len(text)} chars — too little to be a scheme "
            f"document (need >= {MIN_USABLE_CHARS}). The page is probably "
            "JavaScript-rendered or behind a block."
        )


def save_raw(record: dict[str, Any], slug: str, raw_dir: str | Path = RAW_DIR) -> Path:
    """Persist the scrape to ``data/raw/<slug>.json`` and return the path.

    Written as pretty JSON so the captured corpus stays reviewable in the repo.
    """
    out_dir = Path(raw_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{slug}.json"

    record = {**record, "saved_path": str(path)}
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_scraped(path: str | Path) -> dict[str, Any]:
    """Read back a previously scraped record from disk (the query-time-safe path)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    """CLI entrypoint: scrape a URL and write it to ``data/raw/``."""
    parser = argparse.ArgumentParser(description="Scrape one scheme page to file (offline, one-time).")
    parser.add_argument("url", help="Scheme page URL to capture")
    parser.add_argument("--slug", help="Filename/doc_id slug (default: derived from the URL)")
    parser.add_argument("--raw-dir", default=RAW_DIR, help=f"Output directory (default: {RAW_DIR})")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    slug = args.slug or slugify_url(args.url)
    path = save_raw(scrape(args.url, slug), slug, args.raw_dir)
    print(f"Saved {path}")
    print("Next: python ingestion/build_index.py")


if __name__ == "__main__":
    main()
