# Requirements — Indian Farmer Schemes AI Assistant

This document locks the finalized functional and non-functional requirements. It is the single source of truth the other docs (`techstack.md`, `faiss-vectorstore.md`, `CLAUDE.md`, `skills.md`) build on.

---

## 1. Project Summary

A multilingual, voice-enabled AI assistant that answers Indian farmers' questions about government agricultural schemes. A farmer asks — by **voice or text**, in **their own language** — and receives a grounded, source-backed answer **in that same language**, with an on-demand **speak-aloud** option.

**Deployment context:** Runs locally. The demo is screen-recorded; the code is pushed to GitHub. Judges review from **code, pipeline, accuracy, README, techniques used, and components used** — not a live/open demo.

---

## 2. What Gets Judged (Acceptance Criteria)

| Criterion | How this project satisfies it |
|-----------|-------------------------------|
| **Implementation** | Working local app: ingestion → retrieval → generation → UI |
| **Pipeline** | End-to-end LCEL pipeline, documented in README §Request Lifecycle |
| **Accuracy increase** | Two-stage eval harness over a home-grown golden set; measured gains as techniques stack |
| **README** | Super-structured, judge-facing (see `README.md`) |
| **Techniques used** | Full RAG technique set (§6), all LangChain-native |
| **Components used** | Explicit component mapping (`techstack.md`) |

---

## 3. Functional Requirements

### 3.1 Input
- **FR-1** Accept farmer input as **text** or **voice**.
- **FR-2** Support six languages: **Tamil, Hindi, Kannada, Telugu, Malayalam, English**.
- **FR-3** Auto-detect input language.

### 3.2 Retrieval & Generation
- **FR-4** Translate the query to **English** for retrieval (corpus is English).
- **FR-5** Retrieve from a **local FAISS** index using **hybrid** retrieval (dense + lexical) plus re-ranking.
- **FR-6** Generate the answer **directly in the farmer's language**, grounded in retrieved context.
- **FR-7** Every answer must be **traceable to source chunks** (no ungrounded claims).

### 3.3 Output
- **FR-8** Render the answer in the farmer's language.
- **FR-9** **Speak the answer aloud automatically** when it appears (voice-first for farmers who may not read comfortably), with a toggle to switch to on-demand playback. Audio is cached per message so replays cost nothing.

### 3.4 Data & Ingestion
- **FR-10** Ingest from **three source types**, one representative example each:
  - Local **HTML** file — 1 example
  - **Playwright** web scrape — 1 example
  - **PDF** — batch of ~10 assumed, 1 demoed
- **FR-11** **No realtime scraping.** Scrape/parse once, store to file; use metadata at query time.
- **FR-12** Normalize all sources into **one common `Document` schema** before chunking.
- **FR-13** Provide an **OCR fallback** for scanned PDFs (Sarvam Vision / Document AI).

### 3.5 Validation & Safety
- **FR-14** **Input validation** — length/charset checks, language ID.
- **FR-15** **Prompt-injection screen.**
- **FR-16** **Retrieval validation** — relevance threshold + no-confident-answer fallback.
- **FR-17** **Response validation** — groundedness/faithfulness gate.
- **FR-18** **Guardrails** — topical rail: stay on farmer-scheme topics; refuse politely otherwise.
- **FR-19** **Memory** — retain conversational context within a session.
- **FR-20** **Transcript** — store the full chat to file (JSONL) and **email** it to a recipient the user enters in the UI (`EMAIL_TO` pre-fills the field; the address is validated before sending).

### 3.6 Evaluation
- **FR-21** Ship a **home-grown golden dataset** (no third-party eval library).
- **FR-22** **Stage 1 — retrieval metrics** (deterministic): Hit Rate / Recall@k, MRR, context precision.
- **FR-23** **Stage 2 — generation judge** (LLM-as-judge on `sarvam-105b-conversations`, temperature 0): faithfulness + correctness. **Caveat:** this is the same model that writes the answers, so faithfulness is self-graded and biased upward. An independent judge would strengthen the claim.
- **FR-24** Report accuracy as techniques stack: baseline → +hybrid → +rerank → +query expansion.
- **FR-25** Eval runs **text-only** (voice/translation not exercised per eval pass).

---

## 4. Non-Functional Requirements

- **NFR-1 Cost** — Fits within a small Sarvam credit budget (≈₹1,000 funds the full local build; embeddings run at OpenAI's low `text-embedding-3-small` rate).
- **NFR-2 Offline-capable ingestion** — Index builds without live web calls; reproducible by a reviewer with API keys.
- **NFR-3 Local compute** — Runs on a **CPU-only laptop**: Intel Core i7-1165G7 (4 cores / 8 threads), 16GB RAM, Intel Iris Xe integrated graphics — **no discrete GPU**. Everything heavy is an API call (LLM, STT, TTS, translation, embeddings); the only local model is the re-ranker, which selects a lightweight CPU cross-encoder on this hardware (`techstack.md` §5). Measured ~5s per answered question.
- **NFR-4 Reproducibility** — Deterministic retrieval metrics; pinned dependencies.
- **NFR-5 Vendor simplicity** — One vendor (Sarvam) for LLM + STT + TTS + translation; OpenAI only for embeddings.
- **NFR-6 Framework constraint** — LangChain-native (LCEL / Runnable interface). **No LangGraph.**
- **NFR-7 Graceful degradation** — no single external dependency may cost the farmer an answer. Every Sarvam endpoint, the local re-ranker, each LLM-backed retriever, and SMTP has a documented fallback (README §Reliability & Fallbacks). The one intentional exception is groundedness: when an answer cannot be grounded, the no-answer fallback is the *correct* output, not a degradation.

---

## 5. Language & Voice Matrix

| Capability | Tamil | Hindi | Kannada | Telugu | Malayalam | English |
|------------|:---:|:---:|:---:|:---:|:---:|:---:|
| Text in | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Voice in (Saaras v3) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Answer out (Sarvam-105B) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Speak-aloud, automatic (Bulbul v3) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## 6. RAG Techniques (Required Set)

Ingestion: recursive + semantic chunking, metadata enrichment.
Retrieval: hybrid (FAISS dense ∪ BM25 lexical) via `EnsembleRetriever`, multi-query, RAG-fusion, HyDE, self-query metadata filtering, parent-document retrieval, contextual compression + cross-encoder re-rank.

All implemented with LangChain-native components and LCEL. See `techstack.md` for the component-to-technique mapping.

---

## 6a. Verification Status

Recorded after implementation and testing, so the spec reflects what is actually
proven rather than what is merely written.

| Requirement | Status | Evidence |
|-------------|--------|----------|
| FR-1..FR-3 (text/voice in, 6 languages, auto-detect) | ✅ | Live `/text-lid` tagged romanized Tamil `ta`; Saaras returned Tamil + English |
| FR-4 (query → English) | ✅ | Mayura verified live; voice uses the one-call Saaras translate mode |
| FR-5 (FAISS hybrid + rerank) | ✅ | `IndexFlatIP` confirmed; Ensemble + BM25 + compression wired through one factory |
| FR-6..FR-7 (in-language, grounded, traceable) | ✅ | Live on the real corpus: **6/6 test questions answered**, grounded and cited, in English, Hindi and Tamil, all five guards green. Required switching to `sarvam-105b-conversations` — see techstack.md §2 |
| FR-8..FR-9 (render + speak-aloud) | ✅ | Bulbul returned valid WAV in English and Tamil |
| FR-10..FR-12 (3 source types, offline, common schema) | ✅ | All three types in the corpus: PDF (296 pp, 1,326 chunks), local HTML (75), Playwright web scrape (58). Answers cite all three |
| FR-13 (OCR fallback) | ✅ | Sarvam Document AI (`/doc-ai/v1/job/digitise`, async submit→poll→download). Verified on real scanned pages: p.190 → 3,747 chars, p.207 → 45,750 chars including a potato cultivation cost model. Adds **480 chunks** the corpus previously could not see |
| FR-14..FR-18 (5 validation layers) | ✅ | Each guard triggered in test: injection blocked with 0 LLM calls; ungrounded answer replaced; off-topic refused |
| FR-19 (memory) | ✅ | 4 messages retained across 2 turns |
| FR-20 (transcript + email) | ⚠️ partial | JSONL round-trip verified incl. Indic text. UI asks for a recipient and validates it (3 valid / 6 invalid cases tested); form appears only once a turn exists. **SMTP send itself not exercised** (no configured mailbox) |
| FR-21..FR-25 (eval harness) | ⚠️ partial | Metrics hand-verified, judge parsing tested, table renders; **numbers need a corpus + golden set** |

---

## 7. Out of Scope

- Live/open public demo (judged from code + recording).
- Realtime scraping or scheduled data refresh.
- Native-language (Indic) source documents — corpus is curated English-only by design.
- Third-party eval frameworks (RAGAS etc.) — replaced by the home-grown harness.
- LangGraph or any non-LangChain orchestration.
