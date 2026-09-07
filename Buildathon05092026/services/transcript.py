"""Transcript store — per-turn JSONL log plus an SMTP email on session close.

Responsibility
--------------
The audit trail (requirements.md FR-20, README §Validation & Safety). Every turn
is appended to a JSONL file as it happens; when the session closes, the whole
conversation is formatted and emailed to ``EMAIL_TO``.

Turn record
-----------
    {
      "session_id":   str,
      "timestamp":    iso8601,
      "language":     "ta" | "hi" | "kn" | "te" | "ml" | "en",
      "input_mode":   "text" | "voice",
      "query":        str,          # as the farmer asked it
      "translated_query": str,      # what retrieval actually saw
      "answer":       str,
      "citations":    [{"chunk_id", "doc_id", "scheme_name", "page_number", "source_url"}],
      "retrieval":    {"top_score": float, "n_docs": int, "fallback": bool},
      "guards":       {"injection": bool, "grounded": bool, "on_topic": bool}
    }

Notes
-----
* Append-only and flushed per turn — a crash must not lose the transcript.
* SMTP credentials come from ``.env`` (``SMTP_HOST``/``SMTP_PORT``/``SMTP_USER``/
  ``SMTP_PASS``/``EMAIL_TO``). Never hardcoded, never logged.
* Storing the guard outcomes is deliberate: the transcript is the evidence that
  the safety layers actually fired.
"""

from __future__ import annotations

import json
import logging
import os
import re
import smtplib
import uuid
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPT_DIR = str(PROJECT_ROOT / "data" / "processed" / "transcripts")

_SAFE_ID = re.compile(r"[^A-Za-z0-9._-]")


def new_session_id() -> str:
    """Generate a fresh session identifier: sortable and filesystem-safe."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:6]}"


def transcript_path(session_id: str, base_dir: str | Path = TRANSCRIPT_DIR) -> Path:
    """Resolve ``<base_dir>/<session_id>.jsonl``, creating the directory."""
    directory = Path(base_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{_SAFE_ID.sub('_', session_id)}.jsonl"


def log_turn(session_id: str, record: dict[str, Any]) -> None:
    """Append one turn to the session's JSONL file and flush.

    ``ensure_ascii=False`` keeps Indic text readable in the file. Logging must
    never break the conversation, so failures are warned about, not raised.
    """
    entry = {
        "session_id": session_id,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **record,
    }
    try:
        with transcript_path(session_id).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
            handle.flush()
    except OSError as exc:
        logger.warning("Could not write transcript for %s: %s", session_id, exc)


def read_transcript(session_id: str) -> list[dict[str, Any]]:
    """Read back every turn of a session in order."""
    path = transcript_path(session_id)
    if not path.exists():
        return []

    turns: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                turns.append(json.loads(line))
            except json.JSONDecodeError as exc:
                # A half-written final line (crash mid-append) must not cost us
                # the turns that were logged successfully before it.
                logger.warning("Skipping corrupt transcript line %s:%s (%s)", path.name, line_no, exc)
    return turns


def format_email_body(turns: list[dict[str, Any]]) -> str:
    """Render a session's turns into a readable email body."""
    if not turns:
        return "No turns recorded for this session."

    session_id = turns[0].get("session_id", "unknown")
    languages = sorted({turn.get("language", "en") for turn in turns})
    fallbacks = sum(1 for turn in turns if turn.get("retrieval", {}).get("fallback"))

    lines = [
        "Kisan Sahayak — session transcript",
        "=" * 40,
        f"Session   : {session_id}",
        f"Started   : {turns[0].get('timestamp', '')}",
        f"Ended     : {turns[-1].get('timestamp', '')}",
        f"Turns     : {len(turns)}",
        f"Languages : {', '.join(languages)}",
        f"No-answer fallbacks: {fallbacks}",
        "",
    ]

    for index, turn in enumerate(turns, start=1):
        lines.append(f"--- Turn {index} ({turn.get('input_mode', 'text')}, "
                     f"{turn.get('language', 'en')}) ---")
        lines.append(f"Q: {turn.get('query', '')}")
        if turn.get("translated_query") and turn["translated_query"] != turn.get("query"):
            lines.append(f"   (retrieved as: {turn['translated_query']})")
        lines.append(f"A: {turn.get('answer', '')}")

        citations = turn.get("citations") or []
        if citations:
            lines.append("Sources:")
            for citation in citations:
                where = (
                    f"p.{citation['page_number']}"
                    if citation.get("page_number")
                    else citation.get("source_url") or ""
                )
                lines.append(
                    f"  - {citation.get('scheme_name') or citation.get('doc_id')} "
                    f"[{citation.get('chunk_id')}] {where}".rstrip()
                )
        else:
            lines.append("Sources: (none — no confident answer)")

        guards = turn.get("guards") or {}
        if guards:
            lines.append("Guards: " + ", ".join(f"{k}={'ok' if v else 'BLOCKED'}"
                                                for k, v in guards.items()))
        lines.append("")

    return "\n".join(lines)


def send_transcript_email(session_id: str, *, to: str | None = None) -> bool:
    """Email the session transcript via ``smtplib`` on session close.

    Returns ``True`` on success. Failure is logged, never raised — a mail outage
    must not crash the app or lose the on-disk transcript.
    """
    load_dotenv(PROJECT_ROOT / ".env")

    host = os.environ.get("SMTP_HOST", "").strip()
    user = os.environ.get("SMTP_USER", "").strip()
    password = os.environ.get("SMTP_PASS", "").strip()
    recipient = (to or os.environ.get("EMAIL_TO", "")).strip()
    port = int(os.environ.get("SMTP_PORT", "587"))

    if not all((host, user, password, recipient)):
        logger.warning(
            "Transcript email skipped — set SMTP_HOST/SMTP_USER/SMTP_PASS/EMAIL_TO in .env"
        )
        return False

    turns = read_transcript(session_id)
    if not turns:
        logger.info("Nothing to email for session %s", session_id)
        return False

    try:
        message = EmailMessage()
        message["Subject"] = f"Kisan Sahayak transcript — {session_id} ({len(turns)} turns)"
        message["From"] = user
        message["To"] = recipient
        message.set_content(format_email_body(turns))
        message.add_attachment(
            transcript_path(session_id).read_bytes(),
            maintype="application",
            subtype="json",
            filename=f"{session_id}.jsonl",
        )
    except Exception as exc:  # noqa: BLE001 - the on-disk transcript still stands
        logger.warning("Could not assemble transcript email: %s", exc)
        return False

    try:
        with smtplib.SMTP(host, port, timeout=30) as server:
            server.starttls()
            server.login(user, password)
            server.send_message(message)
    except Exception as exc:  # noqa: BLE001 - never crash on mail failure
        logger.warning("Transcript email failed: %s", exc)
        return False

    logger.info("Transcript for %s emailed to %s", session_id, recipient)
    return True


def close_session(session_id: str) -> bool:
    """Finalize a session: flush the log, then dispatch the email.

    Called from the Streamlit "End session" control.
    """
    return send_transcript_email(session_id)
