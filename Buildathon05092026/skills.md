# Skills — Claude Code Build Automations

Reusable **Claude Code Skills** for building and maintaining this project. Each skill is a self-contained `SKILL.md` (plus optional scripts) that Claude Code can invoke to run a repeatable build task consistently.

> **What these are:** developer automations for *building* Kisan Sahayak — not runtime app features. They encode the project's conventions so each task (ingesting a PDF, scraping a scheme, running the eval) is done the same way every time.

Place each under `.claude/skills/<name>/SKILL.md` in the repo.

---

## Skill Index

| Skill | Trigger | What it does |
|-------|---------|--------------|
| `ingest-pdf` | "ingest this PDF", new file in `data/raw/*.pdf` | Load → OCR-fallback → normalize → chunk → add to index |
| `scrape-scheme` | "scrape scheme from <url>" | Playwright scrape → file → normalize (offline, one-time) |
| `ingest-html` | "ingest this HTML file" | BSHTMLLoader → normalize → chunk → add to index |
| `rebuild-index` | "rebuild the FAISS index" | Full re-embed + re-index from `data/processed/documents.jsonl` |
| `add-golden` | "add golden questions" | Append labelled Q→answer→chunk_ids to the golden set |
| `run-eval` | "run the eval", "measure accuracy" | Two-stage harness → prints the accuracy table |
| `add-retriever` | "add <technique> retriever" | Scaffold a LangChain-native retriever, toggleable for eval |

---

## `ingest-pdf`

```markdown
---
name: ingest-pdf
description: Ingest a PDF scheme document into the corpus. Use when a new PDF is
  added to data/raw/ or the user asks to ingest a PDF. Handles text and scanned
  PDFs, normalizes to the common Document schema, chunks, and updates the FAISS index.
---

# Ingest PDF

## Steps
1. Load with `PyMuPDFLoader`. If a page yields little/no text, treat as scanned
   and route that page through Sarvam Vision (Document AI) OCR.
2. Normalize with ingestion.normalize.normalize_pdf_docs — it fills the whole
   common schema (faiss-vectorstore.md §1). Scheme metadata (scheme_name,
   scheme_category, state, department) comes from data/raw/scheme_map.json,
   keyed by doc_id; add an entry there rather than hardcoding.
3. Apply child/parent chunking via build_index.split_parents + split_children
   (RecursiveCharacterTextSplitter, 500/80 characters).
4. Write data/processed/documents.jsonl and parents.jsonl (stable chunk_ids).
5. Rebuild the FAISS index: python ingestion/build_index.py

## Rules
- Corpus is English-only. assert_english drops any page whose alphabetic
  characters are <50% Latin — check the log rather than assuming it ingested.
- Chunk IDs must be stable and unique (doc_id::NNNN); verify_build raises on
  duplicates.
- Pages with <50 chars of text are routed through Sarvam Vision OCR. That path
  is UNVERIFIED against a real scanned PDF — inspect the extracted text.
```

---

## `scrape-scheme`

```markdown
---
name: scrape-scheme
description: Scrape a single scheme web page with Playwright and store it to file.
  Use when the user asks to scrape a scheme URL. Offline/one-time only — never at
  query time.
---

# Scrape Scheme (Playwright → file)

## Steps
1. Run: python ingestion/scrape_playwright.py <url> [--slug my-scheme]
   It launches headless chromium, waits for networkidle, and extracts the main
   content (main / article / [role=main] / #content / .content, longest wins).
2. It writes data/raw/<slug>.json with title, scheme_name, text, source_url,
   scraped_at, saved_path.
3. Rebuild: python ingestion/build_index.py — load_all_sources picks up every
   *.json in data/raw/ and normalizes it with source_type="web".

## Rules
- No realtime scraping in the app. This runs once at build time.
- One representative web example is sufficient for the demo.
- English content only.
```

---

## `run-eval`

```markdown
---
name: run-eval
description: Run the two-stage accuracy harness over the home-grown golden set and
  print the baseline → +hybrid → +rerank → +query-expansion table. Use when the user
  asks to measure accuracy or validate retrieval/generation.
---

# Run Eval

## Steps
1. Stage 1 (deterministic, ₹0): for each golden question, run retrieval and compare
   retrieved chunk_ids to the labelled ground-truth chunk_ids. Compute Hit Rate /
   Recall@k, MRR, context precision.
2. Stage 2 (LLM-as-judge): generate answers, score faithfulness + correctness with
   Sarvam-105B using eval/judge_prompt.txt.
3. Re-run across configurations by toggling RetrieverConfig flags — one flag
   per row, so each delta is attributable to exactly one technique.
4. Print the comparison table; write results to eval/results/.

## Commands
- python eval/run_eval.py --stage 1        # deterministic, no API cost
- python eval/run_eval.py                  # both stages
- python eval/run_eval.py --config baseline,hybrid

## Rules
- Text-only. Never call STT/TTS/translate inside the eval loop.
- No third-party scorer (no RAGAS). Use our metrics.py and judge.py.
- Keep the judge prompt versioned in-repo (JUDGE_PROMPT_VERSION, written into
  every results file).
- A failing question scores 0 and the run continues — check the log for
  "judge call failed" / "run failed" before trusting a low score.
```

---

## `add-golden`

```markdown
---
name: add-golden
description: Add labelled questions to the golden evaluation set. Use when expanding
  eval coverage across schemes, question types, or languages.
---

# Add Golden Questions

## Steps
1. For each new item, write: question, expected_answer, relevant_chunk_ids, scheme,
   question_type (eligibility | subsidy_amount | deadline | documents_required),
   language.
2. Verify each relevant_chunk_id exists in data/processed/documents.jsonl.
3. Append to eval/golden_set.jsonl.

## Rules
- Balance question types and schemes; don't over-index on one scheme.
- chunk_ids must match current chunking (re-check after any rebuild-index).
```

---

## `add-retriever`

```markdown
---
name: add-retriever
description: Scaffold a new LangChain-native retriever (multi-query, self-query,
  parent-doc, HyDE, RAG-fusion) wired into the hybrid pipeline and toggleable for eval.
---

# Add Retriever

## Steps
1. Implement with a LangChain-native component (see techstack.md §4). No LangGraph.
   Import classic retrievers through the langchain_classic shim in
   rag/retrievers.py — langchain.retrievers does not exist in LangChain 1.x.
2. Add a flag to RetrieverConfig and a branch in build_retriever() — that single
   factory is the only path both rag/chain.py and the eval use.
3. Add the config to CONFIGURATIONS in eval/run_eval.py, changing exactly one
   flag from the row above it.
4. Give it a fallback: if it depends on the LLM or a model file, the chain must
   still answer when it fails. retrieval_step already retries with plain dense —
   confirm your technique degrades through it.
5. Update the README accuracy table, the fallback table, and requirements.md §6.

## Rules
- Must be LangChain-native and LCEL-composable.
- Must be measurable (toggleable) for the accuracy story.
```

---

## `rebuild-index`

```markdown
---
name: rebuild-index
description: Rebuild the FAISS index and BM25 corpus from documents.jsonl after
  chunking or corpus changes.
---

# Rebuild Index

## Steps
1. Run: python ingestion/build_index.py --from-jsonl
   This re-embeds from data/processed/documents.jsonl without re-reading
   data/raw/ — use it after an embedding change, NOT after a chunking change.
2. After a chunking change, run the full build (no --from-jsonl) so chunk_ids
   are regenerated from source.
3. Vectors are L2-normalized and the index is IndexFlatIP
   (distance_strategy=MAX_INNER_PRODUCT — required, or you silently get L2).
4. BM25 is rebuilt at app start from documents.jsonl; there is no separate step.
5. verify_build warns if eval/golden_set.jsonl references chunk_ids the new
   build no longer produces. Heed it — the accuracy numbers are void otherwise.
```

---

## Notes

- All skills honor the **Golden Rules** in `CLAUDE.md` (LangChain-native, no LangGraph; English-only corpus; offline ingestion; home-grown eval).
- Skills are build-time helpers; they do not ship as runtime app features.
