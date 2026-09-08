"""Sarvam AI wrappers — LLM, STT, TTS, translation, language ID, OCR.

Responsibility
--------------
The **single** place that talks to Sarvam. One vendor covers the brain and the
voice (techstack.md §6):

    Sarvam-105B   chat / generation   OpenAI-compatible ``/v1/chat/completions``
    Saaras v3     speech-to-text      ``/speech-to-text``  (transcribe|translate)
    Bulbul v3     text-to-speech      ``/text-to-speech``  (base64 audio out)
    Mayura        translation         ``/translate``
    —             language ID         ``/text-lid``
    Sarvam Vision OCR fallback        Document AI (build-time only)

Auth quirk (CLAUDE.md §Conventions)
-----------------------------------
Chat uses ``Authorization: Bearer <key>``; the voice/translate endpoints use the
``api-subscription-key`` header. **Both live here** — do not scatter header
logic across the codebase.

Keys load from ``.env`` via ``python-dotenv``. Never hardcode secrets.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import requests
from dotenv import load_dotenv

if TYPE_CHECKING:  # pragma: no cover - typing only
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SARVAM_BASE_URL = "https://api.sarvam.ai/v1"

# Sarvam exposes two chat models. `sarvam-105b` is a REASONING model: it spends
# its entire ~2,048-token output budget thinking and then returns EMPTY content
# with finish_reason="length". Measured 15/15 empty on a table-heavy scheme
# question; no parameter fixes it (max_tokens, max_completion_tokens and
# reasoning_effort are all ignored or capped server-side).
# `sarvam-105b-conversations` answers directly in 65-171 tokens with the same
# quality, so it is the default for this app. Override with LLM_MODEL in .env.
LLM_MODEL = "sarvam-105b-conversations"
REASONING_MODEL = "sarvam-105b"
STT_MODEL = "saaras:v3"
TTS_MODEL = "bulbul:v3"
TRANSLATE_MODEL = "mayura:v1"

# Must be a speaker the configured TTS model supports — bulbul:v3 voices are
# aditya, ritu, ashutosh, priya, neha, rahul, pooja, rohan, simran, kavya, amit,
# dev, ishita, shreya, ratan, varun, manan, sumit, roopa, kabir, aayan.
DEFAULT_SPEAKER = "ritu"
REQUEST_TIMEOUT = 60

# Document AI is an async job API: submit, poll, download. A single page
# digitises in seconds, but the ceiling stops a stuck job stalling the build.
OCR_JOB_TIMEOUT = 180
OCR_POLL_SECONDS = 2
MAX_RETRIES = 3
RETRY_BACKOFF = 1.5

# Sarvam language codes for the six supported languages (requirements.md §5).
LANGUAGE_CODES: dict[str, str] = {
    "ta": "ta-IN",
    "hi": "hi-IN",
    "kn": "kn-IN",
    "te": "te-IN",
    "ml": "ml-IN",
    "en": "en-IN",
}

# Reverse map, for turning Sarvam's replies back into our short codes.
_SHORT_CODES: dict[str, str] = {v: k for k, v in LANGUAGE_CODES.items()}

SttMode = Literal["transcribe", "translate"]

_ENV_LOADED = False


def _load_env() -> dict[str, str]:
    """Read Sarvam config from ``.env``.

    Loads once per process, then serves from ``os.environ``. Raises a clear
    error naming the missing variable rather than failing deep inside a request.
    """
    global _ENV_LOADED
    if not _ENV_LOADED:
        load_dotenv(PROJECT_ROOT / ".env")
        _ENV_LOADED = True

    api_key = os.environ.get("SARVAM_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "SARVAM_API_KEY is not set. Copy .env.example to .env and add your key "
            "(https://dashboard.sarvam.ai)."
        )
    return {
        "api_key": api_key,
        "base_url": os.environ.get("SARVAM_BASE_URL", SARVAM_BASE_URL).rstrip("/"),
    }


def _voice_base_url() -> str:
    """Root for the non-chat endpoints.

    ``SARVAM_BASE_URL`` points at the OpenAI-compatible ``/v1`` path used by
    chat. The voice/translate endpoints hang off the bare host, so strip it.
    """
    base = _load_env()["base_url"]
    return base[: -len("/v1")] if base.endswith("/v1") else base


def _chat_headers() -> dict[str, str]:
    """Headers for the OpenAI-compatible chat endpoint."""
    return {"Authorization": f"Bearer {_load_env()['api_key']}"}


def _voice_headers() -> dict[str, str]:
    """Headers for the voice/translate endpoints."""
    return {"api-subscription-key": _load_env()["api_key"]}


def normalize_language(code: str | None) -> str:
    """Coerce any language spelling into one of our six short codes.

    Accepts ``"hi"``, ``"hi-IN"``, or ``"Hindi"``; defaults to ``"en"``.
    """
    if not code:
        return "en"
    raw = code.strip()
    if raw in LANGUAGE_CODES:
        return raw
    if raw in _SHORT_CODES:
        return _SHORT_CODES[raw]
    head = raw.replace("_", "-").split("-")[0].lower()
    return head if head in LANGUAGE_CODES else "en"


def to_sarvam_code(code: str | None) -> str:
    """Map a short code to the ``xx-IN`` form the Sarvam API expects."""
    return LANGUAGE_CODES[normalize_language(code)]


# ── LLM ───────────────────────────────────────────────────────────────────────

def get_chat_model(
    temperature: float = 0.2, model: str | None = None, **kwargs: Any
) -> BaseChatModel:
    """Sarvam-105B as a LangChain ``ChatOpenAI`` pointed at ``base_url``.

    The only chat-model constructor in the project — ``rag.chain.get_llm`` and
    ``eval.judge`` both route through here. ``model`` overrides the configured
    default, so a caller can pick a different Sarvam model without touching env.
    """
    from langchain_openai import ChatOpenAI

    env = _load_env()
    return ChatOpenAI(
        model=model or os.environ.get("LLM_MODEL", LLM_MODEL),
        base_url=env["base_url"],
        api_key=env["api_key"],
        temperature=temperature,
        timeout=REQUEST_TIMEOUT,
        max_retries=MAX_RETRIES,
        **kwargs,
    )


# ── Speech ────────────────────────────────────────────────────────────────────

def speech_to_text(
    audio: bytes | str | Path,
    *,
    mode: SttMode = "translate",
    language: str | None = None,
) -> dict[str, Any]:
    """Saaras v3 STT. ``mode="translate"`` does STT **and** EN-translate in one call.

    Returns ``{"text": str, "language": str}``. The one-call translate mode is
    why voice input skips the Mayura step in the chain.
    """
    endpoint = "speech-to-text-translate" if mode == "translate" else "speech-to-text"
    url = f"{_voice_base_url()}/{endpoint}"

    if isinstance(audio, (str, Path)):
        payload = Path(audio).read_bytes()
        filename = Path(audio).name
    else:
        payload = audio
        filename = "input.wav"

    data: dict[str, str] = {"model": os.environ.get("STT_MODEL", STT_MODEL)}
    if mode == "transcribe" and language:
        data["language_code"] = to_sarvam_code(language)

    response = _post(
        url,
        headers=_voice_headers(),
        files={"file": (filename, payload, "audio/wav")},
        data=data,
    )

    detected = response.get("language_code") or response.get("source_language_code")
    return {
        "text": (response.get("transcript") or response.get("text") or "").strip(),
        "language": normalize_language(detected) if detected else normalize_language(language),
    }


def text_to_speech(text: str, language: str, *, speaker: str | None = None) -> bytes:
    """Bulbul v3 TTS — returns decoded audio bytes for the speak-aloud control.

    Sarvam returns base64, so decode before returning. Called on demand from the
    UI only, never inside the eval loop (FR-25).
    """
    response = _post(
        f"{_voice_base_url()}/text-to-speech",
        headers={**_voice_headers(), "Content-Type": "application/json"},
        json={
            "text": text,
            "target_language_code": to_sarvam_code(language),
            "speaker": speaker or os.environ.get("TTS_SPEAKER", DEFAULT_SPEAKER),
            "model": os.environ.get("TTS_MODEL", TTS_MODEL),
        },
    )

    audios = response.get("audios") or []
    if not audios:
        raise RuntimeError(f"Bulbul returned no audio (keys: {sorted(response)})")
    # Long inputs come back as several chunks; concatenating the decoded WAV
    # payloads is good enough for playback in Streamlit.
    return b"".join(base64.b64decode(chunk) for chunk in audios)


# ── Text services ─────────────────────────────────────────────────────────────

def translate_text(text: str, *, source: str, target: str = "en") -> str:
    """Mayura translation — used query-side (farmer's language → English).

    Not used on the answer: generation is in-language by design (path A).
    """
    source_code, target_code = normalize_language(source), normalize_language(target)
    if source_code == target_code or not text.strip():
        return text

    response = _post(
        f"{_voice_base_url()}/translate",
        headers={**_voice_headers(), "Content-Type": "application/json"},
        json={
            "input": text,
            "source_language_code": to_sarvam_code(source_code),
            "target_language_code": to_sarvam_code(target_code),
            "model": os.environ.get("TRANSLATE_MODEL", TRANSLATE_MODEL),
        },
    )
    return (response.get("translated_text") or text).strip()


def detect_language(text: str) -> str:
    """Language ID via ``/text-lid``, constrained to ``LANGUAGE_CODES``.

    Returns ``""`` on failure so the caller (``rag.validators.detect_language``)
    can fall back to ``langdetect`` rather than silently guessing English.
    """
    try:
        response = _post(
            f"{_voice_base_url()}/text-lid",
            headers={**_voice_headers(), "Content-Type": "application/json"},
            json={"input": text},
        )
    except Exception as exc:  # noqa: BLE001 - caller has a local fallback
        logger.warning("Sarvam language detection failed: %s", exc)
        return ""

    code = response.get("language_code") or response.get("lang_code")
    return normalize_language(code) if code else ""


# ── OCR (build-time only) ─────────────────────────────────────────────────────

def ocr_document(
    file_path: str | Path | bytes,
    *,
    page_number: int | None = None,
    filename: str = "page.pdf",
    language: str = "en",
) -> str:
    """Sarvam Document AI OCR for scanned PDF pages.

    The Document AI API is **asynchronous**, not a single call:

        1. POST /doc-ai/v1/job/digitise   -> {"job_id", "status"}
        2. GET  /doc-ai/v1/job/{id}/status    poll to a terminal state
        3. GET  /doc-ai/v1/job/{id}/download-url -> {"url": ...}
        4. GET  that URL -> a ZIP holding the digitised text + per-page metadata

    Uploads are capped at **10 pages**; callers pass a single-page PDF anyway
    (see ``ingestion.load_pdf.ocr_page``), so a 300-page source is never
    re-uploaded per scanned page.

    Accepts a path or raw PDF bytes. Build-time only — never called at query
    time (CLAUDE.md Golden Rule 4).
    """
    import io
    import zipfile

    if isinstance(file_path, bytes):
        payload, name = file_path, filename
    else:
        path = Path(file_path)
        payload, name = path.read_bytes(), path.name

    base = _voice_base_url()

    submitted = _post(
        f"{base}/doc-ai/v1/job/digitise",
        headers=_voice_headers(),
        files={"file": (name, payload, "application/pdf")},
        data={"language": to_sarvam_code(language), "output_format": "md"},
    )
    job_id = submitted.get("job_id")
    if not job_id:
        logger.warning("Document AI returned no job_id for %s (keys: %s)", name, sorted(submitted))
        return ""

    status = _await_ocr_job(base, job_id, name)
    if not status:
        return ""

    location = _get(f"{base}/doc-ai/v1/job/{job_id}/download-url", headers=_voice_headers())
    download_url = location.get("url")
    if not download_url:
        logger.warning("Document AI gave no download URL for job %s", job_id)
        return ""

    archive = requests.get(download_url, timeout=REQUEST_TIMEOUT)
    archive.raise_for_status()

    # The ZIP holds the digitised document plus metadata/manifest files. Take the
    # text output and ignore the bookkeeping.
    text_parts: list[str] = []
    with zipfile.ZipFile(io.BytesIO(archive.content)) as bundle:
        for entry in sorted(bundle.namelist()):
            if entry.endswith(("/", "manifest.json")) or "/metadata/" in entry:
                continue
            if entry.lower().endswith((".md", ".txt", ".html")):
                text_parts.append(bundle.read(entry).decode("utf-8", errors="ignore"))

    text = "\n\n".join(part.strip() for part in text_parts if part.strip())
    if not text:
        logger.warning("Document AI produced no text for %s page %s", name, page_number)
    return text.strip()


def _await_ocr_job(base: str, job_id: str, name: str) -> bool:
    """Poll a Document AI job to a terminal state. ``True`` when it succeeded."""
    import time as _time

    deadline = _time.monotonic() + OCR_JOB_TIMEOUT
    while _time.monotonic() < deadline:
        state = _get(f"{base}/doc-ai/v1/job/{job_id}/status", headers=_voice_headers())
        status = str(state.get("status", "")).lower()
        if status in ("completed", "succeeded", "success", "done"):
            return True
        if status in ("failed", "error", "cancelled"):
            logger.warning("Document AI job %s for %s ended as %r", job_id, name, status)
            return False
        _time.sleep(OCR_POLL_SECONDS)

    logger.warning("Document AI job %s for %s timed out after %ss", job_id, name, OCR_JOB_TIMEOUT)
    return False


def _get(url: str, *, headers: dict[str, str]) -> dict[str, Any]:
    """GET returning JSON, with the same error surfacing as :func:`_post`."""
    response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    if response.status_code >= 400:
        raise RuntimeError(
            f"Sarvam GET {url.rsplit('/', 1)[-1]} [{response.status_code}]: {response.text[:200]}"
        )
    return response.json()


def _post(url: str, *, headers: dict[str, str], **kwargs: Any) -> dict[str, Any]:
    """Shared HTTP POST with retry, timeout, and readable error surfacing.

    One request helper so rate-limit/timeout handling is uniform. The API key is
    never logged — only the endpoint and status.
    """
    kwargs.setdefault("timeout", REQUEST_TIMEOUT)
    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(url, headers=headers, **kwargs)
        except requests.RequestException as exc:
            last_error = exc
        else:
            if response.status_code < 400:
                return response.json()
            # 429/5xx are worth another try; 4xx client errors are not.
            if response.status_code not in (429, 500, 502, 503, 504):
                raise RuntimeError(
                    f"Sarvam {url.rsplit('/', 1)[-1]} failed "
                    f"[{response.status_code}]: {response.text[:300]}"
                )
            last_error = RuntimeError(
                f"Sarvam {url.rsplit('/', 1)[-1]} [{response.status_code}]: "
                f"{response.text[:200]}"
            )

        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_BACKOFF ** attempt)

    raise RuntimeError(f"Sarvam request to {url} failed after {MAX_RETRIES} attempts") from last_error
