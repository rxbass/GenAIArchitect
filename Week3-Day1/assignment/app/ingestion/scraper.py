import json
import time
import requests
import os

from bs4 import BeautifulSoup
from urllib.parse import urlparse


ALLOWED_DOMAIN = "tn.gov.in"

HEADERS = {
    "User-Agent": "TN-Government-Scheme-RAG/1.0"
}


def is_allowed_url(url: str) -> bool:
    parsed = urlparse(url)

    domain = parsed.netloc.lower()

    return (
        parsed.scheme in {"http", "https"}
        and (
            domain == ALLOWED_DOMAIN
            or domain == f"www.{ALLOWED_DOMAIN}"
        )
    )

def fetch_page(url: str) -> str:

    if not is_allowed_url(url):
        raise ValueError(
            f"Blocked URL: {url}"
        )

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=30,
        allow_redirects=False
    )

    if response.status_code in {301, 302, 303, 307, 308}:

        location = response.headers.get("Location")

        raise RuntimeError(
            f"Source redirected: {url} -> {location}"
        )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    for element in soup(
        ["script", "style", "noscript"]
    ):
        element.decompose()

    return soup.get_text(
        separator="\n",
        strip=True
    )


def load_sources(path: str = "data/sources.json"):

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ingest_sources():

    sources = load_sources()

    documents = []

    for source in sources:

        print(
            f"Fetching {source['id']}/"
            f"{len(sources)}: "
            f"{source['name']}"
        )

        try:

            text = fetch_page(source["url"])

            documents.append({
                "id": source["id"],
                "name": source["name"],
                "url": source["url"],
                "text": text,
                "status": "success"
            })

        except Exception as e:

            print(
                f"ERROR: {source['url']} -> {e}"
            )

            documents.append({
                "id": source["id"],
                "name": source["name"],
                "url": source["url"],
                "text": "",
                "status": "failed",
                "error": str(e)
            })

        # Be polite to the government server
        time.sleep(1)

    return documents

def save_documents(documents, path="data/raw/schemes.json"):

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            documents,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"\nSaved corpus to: {path}")

if __name__ == "__main__":

    documents = ingest_sources()

    successful = sum(
        1
        for d in documents
        if d["status"] == "success"
    )

    failed = len(documents) - successful

    print()
    print("========== INGESTION SUMMARY ==========")
    print(f"Total   : {len(documents)}")
    print(f"Success : {successful}")
    print(f"Failed  : {failed}")

    save_documents(documents)