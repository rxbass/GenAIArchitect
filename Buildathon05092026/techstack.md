# Tech Stack — Indian Farmer Schemes AI Assistant

Layer-by-layer component mapping. Every choice ties back to a requirement in `requirements.md` and the "LangChain-native, no LangGraph" constraint.

> **Version note:** the stack below was built and tested against
> **LangChain 1.3.18 / langchain-core 1.6.1 / langchain-community 0.4.2 /
> langchain-openai 1.6.0 / langchain-classic 1.0.8**, with `faiss-cpu 1.15.0`,
> `rank-bm25 0.2.2`, `FlagEmbedding 1.4.2`, `streamlit 1.62.0`, `openai 3.6.0`.
>
> **LangChain 1.x import change:** `langchain.retrievers` no longer exists. The
> classic retrievers (`EnsembleRetriever`, `ContextualCompressionRetriever`,
> `MultiQueryRetriever`, `ParentDocumentRetriever`, `SelfQueryRetriever`) now
> live in **`langchain_classic`**. `rag/retrievers.py` carries a try/except shim
> so both the documented 0.x path and the 1.x path work.

---

## 1. Orchestration

| Component | Choice | Why |
|-----------|--------|-----|
| Framework | **LangChain** (LCEL / Runnable interface) | Brief rewards LangChain-native building blocks |
| Pipeline construction | `RunnableSequence` (`\|` pipe), LCEL | Readable end-to-end chain |
| Conditional routing | `RunnableBranch` | Retrieval-failed → fallback path |
| Custom validators | `RunnableLambda` | Drop input/response guards in as steps |
| Context carry-through | `RunnablePassthrough` | Keep original query + metadata alongside |
| Concurrency | `RunnableParallel` | Fan-out multi-query / hybrid retrievers |
| Memory | `RunnableWithMessageHistory` + `InMemoryChatMessageHistory` | Session memory without a LangGraph checkpointer. **Deprecated in LangChain 1.x** (its warning suggests LangGraph); kept deliberately per the no-LangGraph constraint — verified working, 4 messages retained across 2 turns |

**Not used:** LangGraph — deliberately excluded per the brief. The validation loop is expressed entirely in LCEL.

Packages: `langchain`, `langchain-core`, `langchain-community`, `langchain-openai`.

---

## 2. LLM (Brain)

| Component | Choice | Why |
|-----------|--------|-----|
| Model | **`sarvam-105b-conversations`** (128K context) | Natively Indic across all six languages. The `-conversations` variant, not the base reasoning model — see the note below |
| Access | OpenAI-compatible endpoint | Use `ChatOpenAI` with a custom `base_url` → `https://api.sarvam.ai/v1` |
| SDK path | `langchain-openai` `ChatOpenAI` | No custom client needed |

> **Two Sarvam chat models exist, and picking the wrong one silently breaks the app.**
> `GET /v1/models` returns `sarvam-105b` and `sarvam-105b-conversations`.
>
> `sarvam-105b` is a **reasoning** model. It spends its entire output budget
> thinking and returns `finish_reason="length"` with **empty `content`** — not an
> error, just nothing. Measured **15 of 15 empty** on a table-heavy scheme
> question. The ~2,048-token output cap is server-side: `max_tokens`,
> `max_completion_tokens` and `reasoning_effort` were all tested and **none lift
> it**, nor does a "do not reason" system prompt, nor shrinking the context
> (input was only ~600-1,200 tokens in every failing case).
>
> `sarvam-105b-conversations` answers directly in **65-171 output tokens** with
> the same quality. On the identical failing cases: **0 of 9 empty**, across
> English, Hindi and Tamil. It is the default (`LLM_MODEL` in
> `services/sarvam.py`); override via `.env`.
>
> The empty-answer guard in `validators.validate_groundedness` stays regardless —
> an empty completion must never reach a farmer.

---

## 3. Embeddings

| Component | Choice | Why |
|-----------|--------|-----|
| Model | **OpenAI `text-embedding-3-small`** (1536-dim) | Corpus is curated English-only → embedder never sees Indic text; cheap, ample, keeps FAISS index light |

```python
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
```

> Rationale: `text-embedding-3-large` (3072-dim) earns its cost only on large or nuance-heavy corpora; this corpus (tens of docs) does not need it.

Package: `langchain-openai`, `openai`.

---

## 4. Vector Store & Retrieval

| Component | Choice | Why |
|-----------|--------|-----|
| Vector store | **FAISS** (`IndexFlatIP`, local, persisted) | Exact search, perfect recall at this scale; no server |
| Lexical retriever | **`BM25Retriever`** (rank-bm25) | Adds exact-match/lexical recall (dense-only embeddings can't) |
| Hybrid | **`EnsembleRetriever`** | Fuses dense + BM25 |
| Multi-query | **`MultiQueryRetriever`** | Paraphrase coverage |
| Self-query | **Custom LCEL step** (LLM → metadata filter → FAISS `filter`) | Metadata filtering (scheme/category/state). **Not** `SelfQueryRetriever`: it is unusable here because `langchain_classic` imports `DatabricksVectorSearch` from `langchain_community`, which no longer exports it. The technique is kept; the broken dependency is not |
| Parent-doc | **`ParentDocumentRetriever`** | Embed small, return full context |
| Compression + rerank | **`ContextualCompressionRetriever`** + cross-encoder | Trim noise, reorder |

Packages: `faiss-cpu`, `rank-bm25`, `langchain-community`, `langchain-classic` (LangChain 1.x home of the classic retrievers), `lark` (was needed by `SelfQueryRetriever`; the native filter step does not require it).

See `faiss-vectorstore.md` for index config, metadata schema, and persistence.

---

## 5. Re-Ranker

The reranker is the **only model that runs locally** — everything else (LLM,
STT, TTS, translation, embeddings) is an API call. It is chosen by what the
machine can actually run.

**This project's target machine has no discrete GPU** (Intel i7-1165G7 + Iris Xe
integrated graphics), so the CPU path is the real configuration, not a fallback.

| Component | Choice | Why |
|-----------|--------|-----|
| GPU present | **`bge-reranker-v2-m3`** (568M) | Strongest cross-encoder; robust on imperfectly-translated queries. FP16 on CUDA |
| CPU only | **`cross-encoder/ms-marco-MiniLM-L6-v2`** (22M) | 50x faster on CPU, same top document in testing |
| Override | `RERANKER_MODEL` in `.env` | Forces either model regardless of device |

**Measured on this machine** (8-core CPU, `torch 2.13.0+cpu`, 19 candidates):

| model | ms/doc | 19 docs |
|-------|--------|---------|
| `bge-reranker-v2-m3` | ~1,500 | **~29 s** |
| `bge-reranker-base` | 217 | 4.1 s |
| `ms-marco-MiniLM-L6-v2` | **30** | **0.56 s** |

> **No CUDA GPU is present on the target machine**, and `torch` is a CPU-only
> wheel (`2.13.0+cpu`) to match. Installing a CUDA build would change nothing
> here — Iris Xe is not a CUDA device. The GPU row above documents what the code
> does on a machine that *does* have an NVIDIA card; it is not a requirement.

MiniLM being English-only is not a limitation here: the farmer's query is
translated to English before retrieval and the corpus is English by design.

**Calibration.** Raw logits are squashed through a sigmoid so
`RELEVANCE_THRESHOLD` (0.3) means the same thing for either model. Measured
separation on this corpus: relevant queries score **0.85-0.99**, off-topic
queries score **0.0000** — a clean gap, so the gate is reliable.

`RERANK_CANDIDATE_LIMIT` (20) caps how many candidates are scored: reranking is
linear in candidates, and multi-query can hand the reranker 45+ documents to
return 5.

**End-to-end latency** (CPU, hybrid + rerank): ~5-11s for an answered question,
~1s for an off-topic one (it short-circuits before generation).

---

## 6. Speech, TTS & Translation (Sarvam)

| Capability | Endpoint | Status | Notes |
|------------|----------|--------|-------|
| LLM | `POST /v1/chat/completions` | ✅ verified | `sarvam-105b` via `ChatOpenAI(base_url=...)` |
| Speech-to-text | `POST /speech-to-text` | ✅ verified | `transcribe` mode; returns `transcript` + `language_code` |
| STT + translate | `POST /speech-to-text-translate` | ✅ verified | One call does STT **and** EN-translate — this is why voice input skips Mayura |
| Text-to-speech | `POST /text-to-speech` | ✅ verified | Returns `{"audios": [base64]}`; decoded to WAV bytes |
| Translation | `POST /translate` | ✅ verified | Mayura, query-side (farmer language → EN) |
| Language ID | `POST /text-lid` | ✅ verified | Returns `language_code`; correctly tagged romanized Tamil as `ta` |
| OCR fallback | `POST /doc-ai/v1/job/digitise` | ✅ verified | Sarvam Document AI. **Asynchronous**: submit → poll `/job/{id}/status` → `GET /job/{id}/download-url` → ZIP of digitised text. Max 10 pages per upload, so `load_pdf.ocr_page` submits one page at a time. Verified on real scanned pages |

> **Endpoint base:** chat uses the `/v1` base (`SARVAM_BASE_URL`); every other
> endpoint hangs off the bare host. `services/sarvam.py` derives one from the other.

> **Speaker names are model-specific.** `bulbul:v3` rejects the `bulbul:v2`
> voices with a 400. Valid v3 speakers: `aditya, ritu, ashutosh, priya, neha,
> rahul, pooja, rohan, simran, kavya, amit, dev, ishita, shreya, ratan, varun,
> manan, sumit, roopa, kabir, aayan`. Default is `ritu`.

> Auth quirk: Sarvam chat uses `Authorization: Bearer`; voice endpoints use the `api-subscription-key` header. Wrap both in `services/sarvam.py`.

---

## 7. Ingestion / Loaders

| Source | Loader | Package |
|--------|--------|---------|
| PDF (text) | `PyMuPDFLoader` | `pymupdf` |
| PDF (scanned) | Sarvam Vision OCR → text | Sarvam API |
| HTML (local) | `BSHTMLLoader` | `beautifulsoup4`, `lxml` |
| Web (scrape→file) | Playwright script → JSON | `playwright` |

All loaders emit the **common `Document` schema** (see `faiss-vectorstore.md` §Metadata). Scraping is offline/one-time.

---

## 8. UI

| Component | Choice | Why |
|-----------|--------|-----|
| Framework | **Streamlit** | Fast to build; mic input, language selector, speak-aloud icon |
| Voice capture | **`st.chat_input(accept_audio=True)`** | Mic sits **inside the chat box** (ChatGPT/Claude pattern) and draws a live waveform from the real signal while recording. One control for typing and speaking; returns `ChatInputValue` with `.text` and `.audio` |
| Playback | `st.audio(autoplay=True)` | Answers speak themselves; bytes cached per message so reruns never re-call Bulbul |
| Processing indicator | CSS wave animation | Covers the gap the native widget leaves: after recording stops, Saaras is still transcribing |
| Background | **Inline SVG farmland scene** | Ships in the repo as a ~3KB `data:` URI — no image files, no network fetch at run time (offline rule), crisp at any size |
| Transcript email | `st.form` + text input | Recipient is typed per session; `EMAIL_TO` only pre-fills |
| Reset | "New conversation" button | New `session_id` + cleared history/audio, without a page reload. Cached models are **not** cleared — reloading the index and re-ranker would cost ~10s for nothing |
| Speaker | Sidebar picker over the bulbul:v3 voice list | Lets a voice that reads a given language poorly be swapped without a code change |
| Model warm-up | Eager load at the **top level of `main()`** | The FAISS index and cross-encoder otherwise load lazily inside the first query — which both hid ~10s behind "Searching the scheme documents…" and nested a second spinner inside the assistant bubble, which Streamlit rendered as a detached box floating outside the message |

`.streamlit/config.toml` disables the dev file-watcher. Streamlit's watcher walks
the import graph of every loaded module, and `transformers` (pulled in by the
re-ranker) exposes lazy submodules that raise `ModuleNotFoundError: torchvision`
when touched — costing ~20s of startup and flooding the log with an unrelated
traceback.

Packages: `streamlit`; `streamlit-mic-recorder` only as a fallback.

---

## 9. Memory, Transcript & Email

| Component | Choice |
|-----------|--------|
| Session memory | `RunnableWithMessageHistory` + `ChatMessageHistory` |
| Transcript store | JSONL append per turn |
| Email dispatch | Python `smtplib` (SMTP) on session close |

---

## 10. Evaluation

| Component | Choice | Why |
|-----------|--------|-----|
| Dataset | **Home-grown golden set** (`eval/golden_set.jsonl`) | No third-party; domain-specific |
| Retrieval metrics | Deterministic Python (Hit Rate, MRR, context precision) | ₹0, reproducible |
| Generation judge | **`sarvam-105b-conversations` (temp 0) + hand-written judge prompt** | Faithfulness + correctness, no RAGAS. Currently the **same model that generates the answers** — self-judging, which inflates faithfulness. Independence would need a different judge model |

---

## 11. Full Dependency List (starting pins)

```txt
# Orchestration
langchain
langchain-core
langchain-community
langchain-openai
openai

# Retrieval / vector store
faiss-cpu
rank-bm25

# Re-ranker
FlagEmbedding          # or sentence-transformers

# Ingestion
pymupdf
beautifulsoup4
lxml
playwright

# UI
streamlit
streamlit-mic-recorder

# Utilities
python-dotenv
langdetect             # backup to Sarvam /text-lid
requests               # Sarvam voice/translate endpoints
tqdm

# Not in requirements.txt but needed for one technique
lark                   # SelfQueryRetriever query-constructor grammar
```

> `requirements.txt` in the repo is the authoritative list. `lark` is the one
> known gap; every other import above was verified to resolve in this environment.

Environment variables: `SARVAM_API_KEY`, `OPENAI_API_KEY`, `SMTP_HOST`, `SMTP_USER`, `SMTP_PASS`, `EMAIL_TO`.
