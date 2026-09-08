"""Kisan Sahayak — Streamlit entrypoint.

Responsibility
--------------
Stages 1 and 12 of the request lifecycle: take the farmer's input (typed or
spoken, in any of six languages), hand it to the LCEL chain, and render the
grounded answer with citations and an on-demand speak-aloud control.

    one chat bar (type or speak) · language selector · spoken answer · citations

Boundaries (CLAUDE.md)
----------------------
* **No pipeline logic here.** Validation, retrieval, generation, and guards all
  live in ``rag/`` as LCEL steps. This module wires UI events to ``rag.chain``
  and nothing more.
* Voice is an *edge* concern: Saaras STT on the way in, Bulbul TTS on the way
  out, both via ``services.sarvam``. The chain itself stays text-only.
* **Answers are spoken automatically** (Bulbul TTS) — a voice-first app for
  farmers who may not read comfortably. This costs one TTS call per answer, so
  it is a sidebar toggle; turn it off and the 🔊 button still speaks on demand.
  Audio is cached per message, so replays and reruns never re-bill.
* The index is **loaded**, never built, here. Run
  ``python ingestion/build_index.py`` first.

Usage
-----
    streamlit run app.py
"""

from __future__ import annotations

import logging
import re
from typing import Any
from urllib.parse import quote

import streamlit as st

from rag.chain import build_conversational_chain
from rag.retrievers import RetrieverConfig
from services import transcript

logger = logging.getLogger(__name__)

APP_NAME = "Kisan Sahayak"
APP_TAGLINE = "Indian Farmer Schemes AI Assistant"
PAGE_TITLE = f"🌾 {APP_NAME}"
PAGE_CAPTION = "Ask about government schemes for farmers — by voice or text, in your language."

# Farmland scene painted as inline SVG rather than a downloaded photo: it ships
# with the repo, needs no network at run time (the offline rule in CLAUDE.md),
# stays crisp at any window size, and costs a few KB instead of a few hundred.
FIELD_SCENE_SVG = """
<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1600 900' preserveAspectRatio='xMidYMax slice'>
  <defs>
    <linearGradient id='sky' x1='0' y1='0' x2='0' y2='1'>
      <stop offset='0%' stop-color='#8fc6e8'/>
      <stop offset='45%' stop-color='#d9ecc9'/>
      <stop offset='100%' stop-color='#f6f1d8'/>
    </linearGradient>
    <linearGradient id='far' x1='0' y1='0' x2='0' y2='1'>
      <stop offset='0%' stop-color='#9dbf70'/><stop offset='100%' stop-color='#7ea855'/>
    </linearGradient>
    <linearGradient id='mid' x1='0' y1='0' x2='0' y2='1'>
      <stop offset='0%' stop-color='#7cb342'/><stop offset='100%' stop-color='#5c9a32'/>
    </linearGradient>
    <linearGradient id='near' x1='0' y1='0' x2='0' y2='1'>
      <stop offset='0%' stop-color='#4e8b2a'/><stop offset='100%' stop-color='#3d7222'/>
    </linearGradient>
  </defs>

  <rect width='1600' height='900' fill='url(#sky)'/>
  <circle cx='1290' cy='170' r='74' fill='#ffd97a' opacity='0.92'/>
  <circle cx='1290' cy='170' r='118' fill='#ffd97a' opacity='0.20'/>

  <path d='M0 430 Q 210 372 420 424 T 860 414 T 1320 428 T 1600 408 L1600 900 L0 900 Z' fill='url(#far)'/>
  <path d='M0 530 Q 300 470 610 522 T 1180 512 T 1600 528 L1600 900 L0 900 Z' fill='url(#mid)'/>
  <path d='M0 660 Q 400 596 820 656 T 1600 640 L1600 900 L0 900 Z' fill='url(#near)'/>

  <g stroke='#356b1d' stroke-width='2.5' opacity='0.35'>
    <path d='M-40 900 L620 668'/><path d='M180 900 L735 664'/><path d='M420 900 L855 662'/>
    <path d='M700 900 L980 664'/><path d='M980 900 L1105 668'/><path d='M1280 900 L1235 672'/>
    <path d='M1580 900 L1370 676'/>
  </g>

  <g stroke='#2f5f19' stroke-width='3' fill='none' opacity='0.55'>
    <path d='M120 900 v-96'/><path d='M120 826 q-20 -18 -34 -36'/><path d='M120 826 q20 -18 34 -36'/>
    <path d='M120 872 q-20 -18 -34 -36'/><path d='M120 872 q20 -18 34 -36'/>
    <path d='M1470 900 v-112'/><path d='M1470 818 q-22 -20 -38 -40'/><path d='M1470 818 q22 -20 38 -40'/>
    <path d='M1470 866 q-22 -20 -38 -40'/><path d='M1470 866 q22 -20 38 -40'/>
  </g>
</svg>
"""


def _field_background_uri() -> str:
    """The farmland scene as a CSS-ready ``data:`` URI (no network fetch)."""
    return "data:image/svg+xml;charset=utf-8," + quote(
        re.sub(r"\s+", " ", FIELD_SCENE_SVG).strip(), safe=""
    )

# requirements.md §5 — the six supported languages, shown in the selector.
LANGUAGES: dict[str, str] = {
    "en": "English",
    "hi": "हिन्दी (Hindi)",
    "ta": "தமிழ் (Tamil)",
    "kn": "ಕನ್ನಡ (Kannada)",
    "te": "తెలుగు (Telugu)",
    "ml": "മലയാളം (Malayalam)",
}

AUTO_DETECT = "auto"

# Voices bulbul:v3 accepts — the API rejects v2 speakers such as "anushka"
# with a 400, so this list is taken from that error message.
BULBUL_SPEAKERS = (
    "ritu", "kavya", "shreya", "priya", "neha", "pooja", "simran", "ishita",
    "roopa", "aditya", "ashutosh", "rahul", "rohan", "amit", "dev", "ratan",
    "varun", "manan", "sumit", "kabir", "aayan",
)

# The full technique stack — the same configuration the eval's best row measures.
DEFAULT_CONFIG = RetrieverConfig(hybrid=True, rerank=True)


def inject_theme() -> None:
    """Paint the farmland background and the recording-wave animation.

    Content surfaces get their own translucent panel so text stays legible over
    the scene — a farmer-facing app must not trade readability for decoration.
    """
    st.markdown(
        f"""
        <style>
          /* Light veil only where text sits (top), clearing toward the bottom so
             the fields stay visible. Content itself rides on opaque panels. */
          [data-testid="stAppViewContainer"] {{
              background-image:
                linear-gradient(180deg,
                    rgba(255,255,255,0.55) 0%,
                    rgba(255,255,255,0.35) 40%,
                    rgba(255,255,255,0.12) 100%),
                url("{_field_background_uri()}");
              background-size: cover;
              background-position: center bottom;
              background-attachment: fixed;
          }}
          /* No full-column panel: it ends in a hard edge halfway down the page.
             Legibility comes from per-element panels plus a halo on bare text. */
          [data-testid="stMainBlockContainer"] p,
          [data-testid="stMainBlockContainer"] label,
          [data-testid="stCaptionContainer"] {{
              text-shadow: 0 1px 3px rgba(255,255,255,0.95);
          }}
          [data-testid="stHeader"] {{ background: rgba(0,0,0,0); }}
          [data-testid="stSidebar"] > div:first-child {{
              background: rgba(247, 250, 240, 0.94);
              border-right: 1px solid rgba(90,140,50,0.30);
          }}
          [data-testid="stChatMessage"] {{
              background: rgba(255,255,255,0.90);
              border: 1px solid rgba(90,140,50,0.22);
              border-radius: 12px;
              padding: 0.85rem 1rem;
              margin-bottom: 0.5rem;
          }}
          [data-testid="stExpander"] {{
              background: rgba(255,255,255,0.86);
              border-radius: 10px;
          }}
          h1 {{ text-shadow: 0 1px 2px rgba(255,255,255,0.85); margin-bottom: 0.1rem; }}
          .kisan-tagline {{
              font-size: 1.12rem;
              font-weight: 600;
              color: #3d7222;
              letter-spacing: 0.01em;
              margin: -0.35rem 0 0.15rem 0;
              text-shadow: 0 1px 3px rgba(255,255,255,0.95);
          }}

          /* Streamlit overlays a "Press Enter to submit form" hint inside the
             input. In the narrow sidebar it sits on top of the typed address.
             The form has a visible Send button, so the hint is redundant. */
          [data-testid="stSidebar"] [data-testid="InputInstructions"] {{
              display: none !important;
          }}
          /* Long addresses must not spill out of the sidebar input either. */
          [data-testid="stSidebar"] input {{ text-overflow: ellipsis; }}

          /* ── The chat bar is the primary control: make it read that way ── */
          [data-testid="stBottomBlockContainer"] {{
              background: linear-gradient(180deg,
                  rgba(255,255,255,0.00) 0%, rgba(247,250,240,0.92) 45%);
              padding-bottom: 1.1rem;
          }}
          [data-testid="stChatInput"] {{
              background: #ffffff;
              border: 2px solid rgba(78,139,42,0.55);
              border-radius: 16px;
              box-shadow: 0 6px 20px rgba(45,80,25,0.18);
              min-height: 64px;
              transition: border-color .15s ease, box-shadow .15s ease;
          }}
          [data-testid="stChatInput"]:focus-within {{
              border-color: #4e8b2a;
              box-shadow: 0 8px 26px rgba(45,80,25,0.30);
          }}
          /* The textarea and its wrapper carry the theme's secondary colour,
             which shows through the white bar as a muddy green. */
          [data-testid="stChatInput"] > div,
          [data-testid="stChatInputTextArea"] {{
              background: transparent !important;
              background-color: transparent !important;
          }}
          [data-testid="stChatInputTextArea"] {{
              font-size: 1.06rem !important;
              color: #22331a !important;
              padding-top: 0.9rem !important;
              padding-bottom: 0.9rem !important;
          }}
          [data-testid="stChatInputTextArea"]::placeholder {{
              color: #5d6b52 !important; opacity: 1;
          }}
          /* Mic and send: visible affordances, not faint grey glyphs. */
          [data-testid="stChatInputMicButton"],
          [data-testid="stChatInputSubmitButton"] {{
              background: #4e8b2a !important;
              color: #ffffff !important;
              border-radius: 10px !important;
              width: 40px !important; height: 40px !important;
              margin: 0 3px;
              transition: background .15s ease, transform .1s ease;
          }}
          [data-testid="stChatInputMicButton"] svg,
          [data-testid="stChatInputSubmitButton"] svg {{
              fill: #ffffff !important; color: #ffffff !important;
              width: 20px; height: 20px;
          }}
          [data-testid="stChatInputMicButton"]:hover,
          [data-testid="stChatInputSubmitButton"]:hover {{
              background: #3d7222 !important; transform: translateY(-1px);
          }}
          [data-testid="stChatInputSubmitButton"]:disabled {{
              background: rgba(78,139,42,0.35) !important;
          }}

          /* Recording / transcribing indicator */
          .kisan-wave {{
              display: flex; align-items: flex-end; gap: 4px;
              height: 34px; padding: 6px 12px; border-radius: 10px;
              background: rgba(255,255,255,0.80);
              border: 1px solid rgba(90,140,50,0.30);
          }}
          .kisan-wave span {{
              display: block; width: 4px; border-radius: 2px;
              background: linear-gradient(180deg,#7cb342,#3d7222);
              animation: kisan-bounce 1.05s ease-in-out infinite;
          }}
          .kisan-wave.idle span {{ animation: none; height: 5px; opacity: 0.45; }}
          .kisan-wave span:nth-child(1) {{ animation-delay: -0.90s; }}
          .kisan-wave span:nth-child(2) {{ animation-delay: -0.78s; }}
          .kisan-wave span:nth-child(3) {{ animation-delay: -0.66s; }}
          .kisan-wave span:nth-child(4) {{ animation-delay: -0.54s; }}
          .kisan-wave span:nth-child(5) {{ animation-delay: -0.42s; }}
          .kisan-wave span:nth-child(6) {{ animation-delay: -0.30s; }}
          .kisan-wave span:nth-child(7) {{ animation-delay: -0.18s; }}
          .kisan-wave span:nth-child(8) {{ animation-delay: -0.06s; }}
          .kisan-wave span:nth-child(9) {{ animation-delay: -0.36s; }}
          .kisan-wave span:nth-child(10) {{ animation-delay: -0.62s; }}
          @keyframes kisan-bounce {{
              0%, 100% {{ height: 6px; }}
              50%      {{ height: 28px; }}
          }}
          .kisan-wave-label {{
              font-size: 0.86rem; color: #2f5f19; font-weight: 600;
              margin-left: 10px; align-self: center;
          }}
          .kisan-wave-row {{ display: flex; align-items: center; margin: 2px 0 6px 0; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_wave(label: str, *, active: bool = True) -> None:
    """Show the audio-wave indicator with a caption.

    ``active`` animates the bars (recording / transcribing); otherwise they sit
    flat, so the control reads as present-but-idle rather than stuck.
    """
    bars = "".join("<span></span>" for _ in range(10))
    css_class = "kisan-wave" if active else "kisan-wave idle"
    st.markdown(
        f'<div class="kisan-wave-row"><div class="{css_class}">{bars}</div>'
        f'<div class="kisan-wave-label">{label}</div></div>',
        unsafe_allow_html=True,
    )


def configure_page() -> None:
    """Set page config, theme, title, and caption."""
    st.set_page_config(
        page_title=f"{APP_NAME} — {APP_TAGLINE}", page_icon="🌾", layout="wide"
    )
    inject_theme()
    st.title(PAGE_TITLE)
    # The tagline is a separate line rather than crammed into the H1: the name
    # stays big and readable, and the full product name is still on screen.
    st.markdown(f'<div class="kisan-tagline">{APP_TAGLINE}</div>', unsafe_allow_html=True)
    st.caption(PAGE_CAPTION)


def init_state() -> None:
    """Initialize ``st.session_state``: session_id, messages, language."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = transcript.new_session_id()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_audio" not in st.session_state:
        st.session_state.last_audio = None
    if "detected_language" not in st.session_state:
        st.session_state.detected_language = None
    if "spoken" not in st.session_state:
        # message index -> synthesized audio bytes, so a rerun replays rather
        # than re-calling Bulbul.
        st.session_state.spoken = {}


@st.cache_resource(show_spinner="Loading scheme index and re-ranker…")
def load_pipeline(_config: RetrieverConfig):
    """Load the FAISS index, re-ranker, and conversational chain — cached once.

    The models are **warmed here on purpose**. Both the FAISS index and the
    cross-encoder otherwise load lazily inside the first query, which put ~10s
    of one-time model loading behind the "Searching the scheme documents…"
    spinner and made the first question look pathologically slow. Paying it up
    front, under an honest "Loading…" message, is the same total time but an
    honest one.

    ``cache_resource`` keeps both in memory across Streamlit reruns.
    """
    chain = build_conversational_chain(_config)

    from rag.retrievers import load_chunks, load_vectorstore

    load_vectorstore()
    load_chunks()
    if _config.rerank:
        from rag.rerank import load_reranker

        load_reranker()
    return chain


_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")


def render_transcript_email() -> None:
    """Ask for a recipient address, then send this session's transcript there.

    The address is typed per session rather than read from ``EMAIL_TO``, so a
    farmer (or a demo reviewer) can send the record wherever they need it.
    ``EMAIL_TO`` from ``.env`` only pre-fills the box as a convenience.
    """
    import os

    st.subheader("📧 Email transcript")

    turns = len(st.session_state.messages) // 2
    if not turns:
        st.caption("Ask a question first — there is nothing to send yet.")
        return

    with st.form("transcript_email", clear_on_submit=False):
        recipient = st.text_input(
            "Send to",
            value=st.session_state.get("email_to", os.environ.get("EMAIL_TO", "")),
            placeholder="name@example.com",
            help=f"Sends all {turns} turn(s) of this session, with the JSONL attached.",
        )
        submitted = st.form_submit_button("Send transcript", use_container_width=True)

    if not submitted:
        return

    recipient = recipient.strip()
    if not _EMAIL_RE.match(recipient):
        st.error("Enter a valid email address, e.g. name@example.com")
        return

    st.session_state.email_to = recipient
    with st.spinner(f"Sending to {recipient}…"):
        try:
            sent = transcript.send_transcript_email(
                st.session_state.session_id, to=recipient
            )
        except Exception as exc:  # noqa: BLE001 - the JSONL is already on disk
            logger.warning("Transcript email failed: %s", exc)
            sent = False

    if sent:
        st.success(f"Transcript sent to {recipient}.")
    else:
        st.warning(
            "Could not send. The transcript is safe on disk at "
            f"`{transcript.transcript_path(st.session_state.session_id)}`. "
            "Check SMTP_HOST / SMTP_USER / SMTP_PASS in .env."
        )


def reset_chat() -> None:
    """Start a fresh conversation without reloading the page.

    A new ``session_id`` means a new transcript file and a new memory bucket, so
    the next question genuinely starts clean — the previous session's JSONL stays
    on disk and remains emailable. Cached models are deliberately NOT cleared:
    reloading the index and re-ranker would cost ~10s for no benefit.
    """
    from rag.chain import _session_histories

    _session_histories.pop(st.session_state.session_id, None)

    st.session_state.messages = []
    st.session_state.spoken = {}
    st.session_state.last_audio = None
    st.session_state.detected_language = None
    st.session_state.session_id = transcript.new_session_id()


def render_sidebar() -> dict[str, Any]:
    """Language selector, retrieval settings, and the End-session button."""
    with st.sidebar:
        st.header("Settings")

        options = [AUTO_DETECT, *LANGUAGES]
        language = st.selectbox(
            "Language",
            options,
            format_func=lambda code: "Auto-detect" if code == AUTO_DETECT else LANGUAGES[code],
            help="The answer comes back in this language.",
        )

        st.divider()
        st.subheader("Retrieval")
        hybrid = st.checkbox("Hybrid (dense + BM25)", value=True)
        rerank = st.checkbox("Cross-encoder re-rank", value=True)
        multi_query = st.checkbox(
            "Multi-query expansion",
            value=False,
            help="Adds an LLM call and ~2x the rerank work. Off until the eval shows it pays.",
        )

        st.divider()
        st.subheader("Voice")
        st.checkbox(
            "Speak answers automatically",
            value=True,
            key="autoplay",
            help="Reads each answer aloud with Bulbul as soon as it appears. "
                 "One TTS call per answer — switch off to speak only on demand.",
        )
        st.selectbox(
            "Speaker",
            BULBUL_SPEAKERS,
            key="speaker",
            help="Bulbul voice used for the spoken answer. If one reads your "
                 "language poorly, try another.",
        )

        st.divider()
        render_transcript_email()

        st.divider()
        if st.button(
            "🧹 New conversation",
            use_container_width=True,
            help="Clears the chat and starts a new session. The previous "
                 "transcript stays saved on disk.",
            disabled=not st.session_state.messages,
        ):
            reset_chat()
            st.rerun()

        st.caption(f"Session `{st.session_state.session_id}`")
        st.caption(f"{len(st.session_state.messages) // 2} turn(s)")

    return {
        "language": None if language == AUTO_DETECT else language,
        "config": RetrieverConfig(hybrid=hybrid, rerank=rerank, multi_query=multi_query),
    }


def render_history() -> None:
    """Replay the conversation so far as chat bubbles.

    History never auto-plays: re-reading an old answer aloud because the page
    reran would be startling, and the audio is already cached for replay.
    """
    for index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_citations(message.get("citations") or [])
                render_speech(
                    message["content"],
                    message.get("language", "en"),
                    index,
                    autoplay=False,
                )


def get_chat_input() -> tuple[str, str | None, str] | None:
    """One chat bar for both typing and speaking, ChatGPT/Claude style.

    ``st.chat_input(accept_audio=True)`` puts the mic **inside** the text box and
    draws a live waveform there while recording, so text and voice share a single
    control instead of the mic sitting in its own panel above the conversation.

    Returns ``(query, translated_query, input_mode)``:
    voice arrives already translated to English by Saaras, so
    ``translated_query`` is set and the chain skips Mayura.
    """
    value = st.chat_input(
        "Ask about a scheme — eligibility, subsidy, deadline, documents…",
        accept_audio=True,
    )
    if not value:
        return None

    # Plain string when accept_audio is unsupported; ChatInputValue otherwise.
    if isinstance(value, str):
        return (value, None, "text") if value.strip() else None

    text = (getattr(value, "text", "") or "").strip()
    clip = getattr(value, "audio", None)

    if clip is not None:
        clip_id = str(getattr(clip, "file_id", getattr(clip, "name", "")))
        if clip_id == st.session_state.last_audio:
            return None  # same clip on a rerun — don't re-transcribe or re-bill
        st.session_state.last_audio = clip_id

        spoken = transcribe(clip.getvalue())
        if spoken is None:
            return None
        english, language = spoken
        st.session_state.detected_language = language
        return english, english, "voice"

    return (text, None, "text") if text else None


def transcribe(audio_bytes: bytes) -> tuple[str, str] | None:
    """Saaras STT+translate in one call. Returns ``(english_text, language)``.

    The chat bar's own waveform stops the moment recording ends, so show the
    wave indicator for the transcription leg — otherwise the farmer stares at a
    frozen screen while Saaras works.
    """
    from services import sarvam

    placeholder = st.empty()
    with placeholder.container():
        render_wave("Listening to your question…", active=True)
    try:
        result = sarvam.speech_to_text(audio_bytes, mode="translate")
    except Exception as exc:  # noqa: BLE001 - surface, don't crash the UI
        placeholder.empty()
        st.error(f"Could not transcribe the recording: {exc}")
        return None
    placeholder.empty()

    if not result["text"]:
        st.warning("Nothing was heard in that recording. Please try again.")
        return None
    return result["text"], result["language"]


def handle_turn(
    query: str,
    language: str | None,
    config: RetrieverConfig,
    *,
    translated_query: str | None = None,
    input_mode: str = "text",
) -> dict[str, Any]:
    """Run one farmer turn through the chain and log it."""
    session_id = st.session_state.session_id
    chain = load_pipeline(config)

    payload: dict[str, Any] = {"query": query, "session_id": session_id}
    if language:
        payload["language"] = language
    if translated_query:
        payload["translated_query"] = translated_query

    result = chain.invoke(payload, config={"configurable": {"session_id": session_id}})

    transcript.log_turn(
        session_id,
        {
            "language": result.get("language", language or "en"),
            "input_mode": input_mode,
            "query": query,
            "translated_query": result.get("translated_query"),
            "answer": result.get("answer", ""),
            "citations": result.get("citations") or [],
            "retrieval": result.get("retrieval") or {},
            "guards": result.get("guards") or {},
        },
    )
    return result


def render_citations(citations: list[dict[str, Any]]) -> None:
    """List the source chunks behind an answer — every answer is traceable (FR-7)."""
    if not citations:
        return

    with st.expander(f"📄 Sources ({len(citations)})"):
        for citation in citations:
            where = (
                f"page {citation['page_number']}"
                if citation.get("page_number")
                else citation.get("source_url") or citation.get("source_path") or ""
            )
            score = citation.get("rerank_score")
            line = f"**{citation.get('scheme_name') or citation.get('doc_id')}** — {where}"
            if score is not None:
                line += f"  ·  relevance {float(score):.2f}"
            st.markdown(line)
            st.caption(f"`{citation.get('chunk_id')}`")


def render_answer(payload: dict[str, Any]) -> None:
    """Render one answer with its citations."""
    st.markdown(payload.get("answer", ""))
    render_citations(payload.get("citations") or [])

    retrieval = payload.get("retrieval") or {}
    if retrieval.get("fallback"):
        st.info("No confident answer was found in the scheme documents for this question.")


def synthesize(text: str, language: str, index: int) -> bytes | None:
    """Bulbul TTS for one message, cached by message index.

    The cache is what makes auto-play affordable: Streamlit reruns the whole
    script on every interaction, so without it each rerun would re-synthesize
    every visible answer and re-bill for audio the farmer already heard.
    """
    cached = st.session_state.spoken.get(index)
    if cached is not None:
        return cached

    from services import sarvam

    try:
        audio = sarvam.text_to_speech(
            text, language, speaker=st.session_state.get("speaker")
        )
    except Exception as exc:  # noqa: BLE001 - TTS is optional, never fatal
        logger.warning("TTS failed: %s", exc)
        return None

    st.session_state.spoken[index] = audio
    return audio


def render_speech(text: str, language: str, index: int, *, autoplay: bool) -> None:
    """Speak an answer — automatically when enabled, otherwise on demand.

    Auto-play only ever fires for a message that has not been spoken yet, so
    scrolling back through history never restarts old audio.
    """
    if not text.strip():
        return

    already_spoken = index in st.session_state.spoken

    spoken_in = LANGUAGES.get(language, language)

    if autoplay and not already_spoken:
        with st.spinner("Preparing the spoken answer…"):
            audio = synthesize(text, language, index)
        if audio:
            st.audio(audio, format="audio/wav", autoplay=True)
            # Surfaced so a wrong-language answer is obvious at a glance rather
            # than something the farmer has to diagnose by ear.
            st.caption(f"🔊 Spoken in {spoken_in}")
        else:
            st.caption("🔇 Could not generate audio — the text answer is above.")
        return

    if already_spoken:
        # Replay control, no second Bulbul call.
        st.audio(st.session_state.spoken[index], format="audio/wav")
        st.caption(f"🔊 Spoken in {spoken_in}")
        return

    if st.button("🔊 Listen", key=f"tts-{index}"):
        with st.spinner("Generating audio…"):
            audio = synthesize(text, language, index)
        if audio:
            st.audio(audio, format="audio/wav", autoplay=True)
        else:
            st.error("Could not generate audio.")


def main() -> None:
    """Compose the app: config → state → sidebar → history → input → answer."""
    configure_page()
    init_state()
    settings = render_sidebar()

    # Warm the index and re-ranker at top level, BEFORE any chat bubble exists.
    # Called lazily from handle_turn instead, its cache_resource spinner nests
    # inside the assistant bubble's own spinner and Streamlit renders it as a
    # detached box floating outside the message.
    try:
        load_pipeline(settings["config"])
    except FileNotFoundError as exc:
        st.error(
            "The scheme index has not been built yet. Run "
            "`python ingestion/build_index.py` and reload this page."
        )
        st.caption(str(exc))
        return

    render_history()

    submitted = get_chat_input()
    if submitted is None:
        return
    query, translated, mode = submitted

    # Voice carries its own detected language; typed input uses the selector
    # (or auto-detect, which the chain resolves).
    language = (
        st.session_state.detected_language if mode == "voice" else settings["language"]
    )

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching the scheme documents…"):
            try:
                payload = handle_turn(
                    query,
                    language,
                    settings["config"],
                    translated_query=translated,
                    input_mode=mode,
                )
            except FileNotFoundError as exc:
                # The one failure the farmer can actually act on.
                st.error(
                    "The scheme index has not been built yet. Run "
                    "`python ingestion/build_index.py` and reload this page."
                )
                st.caption(str(exc))
                return
            except Exception as exc:  # noqa: BLE001 - show the farmer something useful
                logger.exception("Turn failed")
                st.error(
                    "Something went wrong answering that question. Please try "
                    "again. If it keeps happening, check the terminal log."
                )
                st.caption(str(exc))
                return

        render_answer(payload)
        # This message's index once it is appended below.
        render_speech(
            payload.get("answer", ""),
            payload.get("language", "en"),
            len(st.session_state.messages),
            autoplay=st.session_state.get("autoplay", True),
        )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": payload.get("answer", ""),
            "citations": payload.get("citations") or [],
            "language": payload.get("language", "en"),
        }
    )


if __name__ == "__main__":
    main()
