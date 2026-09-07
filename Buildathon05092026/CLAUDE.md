# CLAUDE.md

Project instructions for Claude Code working in this repository. Read this before making changes.

---

## Project

**Kisan Sahayak** — a multilingual, voice-enabled AI assistant answering Indian farmers' questions about government schemes. Farmer asks by voice/text in their language → grounded answer in that language, with speak-aloud.

This is a **buildathon** project judged on: implementation, pipeline, accuracy increase, README quality, techniques used, components used. Optimize for **clarity, defensibility, and a clean pipeline** over cleverness.

---

## Golden Rules (do not violate)

1. **LangChain-native only. No LangGraph.** The whole pipeline is LCEL (`RunnableSequence`, `RunnableBranch`, `RunnableLambda`, `RunnableParallel`, `RunnableWithMessageHistory`). The brief rewards LangChain components and native functions — do not introduce LangGraph or other orchestration frameworks.
2. **Sarvam is the brain and the voice.** LLM = **`sarvam-105b-conversations`** via OpenAI-compatible endpoint (`ChatOpenAI` + `base_url`). Use the `-conversations` variant, **not** the base `sarvam-105b`, which is a reasoning model that returns empty answers (measured 15/15 on one question). STT = Saaras v3, TTS = Bulbul v3, translate = Mayura. One vendor.
3. **Embeddings are OpenAI, corpus is English-only.** Use `text-embedding-3-small`. Do not embed Indic text — the corpus is curated English by design. Do not swap to a large/Indic embedder without updating `requirements.md` and `faiss-vectorstore.md`.
4. **No realtime scraping.** Ingestion runs once, offline, to file. Playwright is a build-time script, never called at query time.
5. **No third-party eval library.** The eval harness is home-grown: deterministic retrieval metrics + a hand-written LLM-as-judge on Sarvam. Do not add RAGAS, DeepEval, etc.
6. **Every answer must be grounded.** If retrieval is weak, return the no-confident-answer fallback. Never let the LLM answer un-grounded.
7. **Answer in the farmer's language directly** (path A) — do not build an answer-then-translate round trip as the primary path.

---

## Architecture (at a glance)

```
Streamlit UI
  → input validation (lang detect, injection screen)
  → query → English (Saaras translate / Mayura)
  → query expansion (multi-query, HyDE)
  → hybrid retrieve (FAISS dense + BM25) → rerank (bge-reranker-v2-m3)
  → retrieval validation (relevance gate / fallback)
  → generate (Sarvam-105B, in farmer's language, grounded)
  → response validation (groundedness judge)
  → guardrail (on-topic)
  → memory + transcript (JSONL + email)
  → answer (+ Bulbul TTS on demand)
```

See `README.md` for the full lifecycle table and `faiss-vectorstore.md` for retrieval internals.

---

## Repository Map

| Path | Responsibility |
|------|----------------|
| `app.py` | Streamlit entrypoint (UI + voice edges only, no pipeline logic). One `st.chat_input(accept_audio=True)` bar serves both typing and speaking |
| `ingestion/` | Loaders (pdf/html/playwright), normalize, build_index |
| `ingestion/normalize.py` | **The only place the metadata schema lives** |
| `rag/retrievers.py` | Hybrid / multi-query / RAG-fusion / HyDE / self-query / parent-doc |
| `rag/rerank.py` | bge-reranker-v2-m3 wrapper (`BaseDocumentCompressor`) |
| `rag/chain.py` | The LCEL pipeline |
| `rag/validators.py` | Input / retrieval / response guards (RunnableLambda) |
| `services/sarvam.py` | STT / TTS / translate / LLM wrappers (handles the two auth headers) |
| `services/transcript.py` | JSONL store + email |
| `eval/` | golden_set.jsonl, metrics.py, judge.py, run_eval.py |
| `data/processed/` | **Generated** — documents.jsonl, parents.jsonl, faiss_index/, transcripts/ |
| `eval/results/` | **Generated** — timestamped JSON per eval run |

---

## Conventions

- **Common Document schema** — every loader emits the exact metadata in `faiss-vectorstore.md` §1. Do not add source-specific fields elsewhere; extend the schema in one place.
- **Chunk IDs are stable** — `eval/golden_set.jsonl` references them. Changing chunking invalidates the golden set; if you must, re-label.
- **Validators are LCEL steps** — implement each guard as a `RunnableLambda` so it composes into the chain, not as ad-hoc code in `app.py`.
- **Paths are anchored to `PROJECT_ROOT`, never the working directory.** Every module resolves data paths from `Path(__file__).resolve().parent.parent`, so `streamlit run`, the ingestion CLIs and the eval harness all work regardless of where they are launched. A bare relative path like `"data/processed/faiss_index"` is a bug — it silently breaks the moment someone runs the app from the repo root.
- **Secrets via `.env`** — never hardcode keys. Vars: `SARVAM_API_KEY`, `OPENAI_API_KEY`, `SMTP_*`, `EMAIL_TO`.
- **Sarvam auth** — chat uses `Authorization: Bearer`; voice uses `api-subscription-key`. Both live behind `services/sarvam.py`; don't scatter header logic.
- **Degrade, don't crash** — every external dependency (Sarvam endpoint, reranker weights, LLM-backed retriever, SMTP) must have a fallback that costs quality, not the answer. See README §Reliability & Fallbacks for the full table; add a row there when you add a dependency. The deliberate exceptions that *should* raise: missing API keys, missing FAISS index, unknown metadata field, duplicate `chunk_id`, non-English source doc.
- **A fallback must not swallow a code bug.** `except Exception` around a degradation path will happily hide a `NameError` from a bad edit — the component then silently does nothing while every test still "passes". Re-raise `NameError`/`AttributeError`/`TypeError` in these handlers (see `rerank.rerank`) and reserve the broad catch for genuine environment failures.
- **Answers are spoken automatically**, and TTS output is **cached per message** (`st.session_state.spoken`). Streamlit reruns the whole script on every interaction, so an uncached auto-play would re-synthesize every visible answer on each rerun and re-bill for audio the farmer already heard. History never auto-plays — only a newly generated answer does.
- **Answers state the answer, not their plumbing.** The model must never name a source document, file, page title or "the context" — an internal slug like `tnau-paddy-schemes` means nothing to a farmer, and provenance already has a proper home in the Sources panel (FR-7). `format_context` therefore hands the model **only** the text and the scheme name: no paths, URLs, page numbers or chunk ids. What it cannot see, it cannot echo. Citations are attached programmatically in `validators.attach_citations`.
- **Canned guard messages must be localized.** The fallback/refusal constants in `rag/validators.py` are English; `chain.localize_step` runs last and translates them into the farmer's language. Never show — or speak — an English guard message to a non-English speaker.
- **Memory must feed retrieval, not just generation.** `chain.condense_step` rewrites a follow-up into a standalone query before retrieval. Without it, "Are there Union Government schemes?" is searched on those five words alone and finds nothing. When editing the condense prompt, keep the rule that **scope replaces rather than accumulates** — carrying "Tamil Nadu" into a question about Central schemes retrieves the wrong documents.
- **Classifiers fail open** — an injection/topic classifier outage must not block a farmer. The groundedness gate is the one that fails *closed*, because an ungrounded answer is worse than none.
- **LangChain 1.x import shim** — `langchain.retrievers` no longer exists; the classic retrievers are in `langchain_classic`. `rag/retrievers.py` tries the documented 0.x path first and falls back. Keep both paths when touching those imports.
- **`RunnableWithMessageHistory` is deprecated** and its warning recommends LangGraph. Ignore that — Golden Rule 1 wins. It works; the warning is expected.
- **Never treat an empty completion as an answer.** A model that spends its output budget reasoning returns `finish_reason="length"` with empty content and no error. The guard in `validators.validate_groundedness` turns that into the no-answer fallback — keep it, and keep the retry in `chain.generation_step`.
- **`lark` is unlisted but needed** for `SelfQueryRetriever`. Absent it, self-query degrades to dense with a warning rather than erroring.

---

## Build & Run Commands

```bash
# install
pip install -r requirements.txt
pip install lark                      # self-query retriever only
playwright install chromium

# capture the one web source (offline, one-time)
python ingestion/scrape_playwright.py <url> [--slug my-scheme]

# build the index (offline, run once, before first launch)
python ingestion/build_index.py
python ingestion/build_index.py --from-jsonl   # re-embed without re-reading data/raw/

# run the app
streamlit run app.py

# run the accuracy harness (text-only)
python eval/run_eval.py --stage 1     # deterministic retrieval metrics, no API cost
python eval/run_eval.py               # both stages
python eval/run_eval.py --config baseline,hybrid
```

---

## When Adding a RAG Technique

The required set is already enumerated in `requirements.md` §6. When implementing one:
1. Keep it LangChain-native (see `techstack.md` §4 for the component).
2. Make it toggleable so the eval harness can measure **baseline → +technique** deltas.
3. Record the accuracy delta in the README accuracy table.

---

## When Editing the Eval

- **Faithfulness and correctness must be reported together.** A no-answer fallback scores *faithful* (it invents nothing) but *incorrect* (it lacks the expected fact), so a stack that refuses more often looks better on faithfulness alone. Never quote faithfulness as a headline number by itself.
- **Score every configuration at the same cutoff.** Configs return different numbers of documents (dense 10, hybrid ~19, reranked 5); scoring full result sets compares result-set size, not ranking quality.
- Retrieval metrics (`metrics.py`) stay **deterministic and ₹0** — compare retrieved chunk IDs to golden chunk IDs.
- The judge (`judge.py`) uses **`sarvam-105b-conversations` at temperature 0 + our prompt** — keep the prompt in-repo and versioned; don't import a scorer. It borrows `services.sarvam.get_chat_model`, so it is **the same model that writes the answers**: faithfulness is self-graded. If you change the app's model, the judge changes with it — check that this is what you want.
- Eval is **text-only** — never fire STT/TTS/translate inside an eval loop (cost + irrelevance).

---

## Definition of Done (per feature)

- [ ] Implemented LangChain-native (no LangGraph)
- [ ] Grounded / validated (no un-cited answers)
- [ ] Degrades gracefully if its dependency is down, and the README fallback table has a row for it
- [ ] Works across the six languages where user-facing
- [ ] Eval delta recorded if it touches retrieval/generation
- [ ] README + relevant doc updated
