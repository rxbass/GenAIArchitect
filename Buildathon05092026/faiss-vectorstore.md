# FAISS Vector Store — Design & Configuration

Everything about how documents become a searchable local index: the common schema, chunking, embedding config, FAISS index type, metadata design, persistence, and how the store wires into hybrid retrieval.

---

## 1. Common Document Schema

Every loader (PDF, HTML, Playwright) normalizes to **one schema** before chunking. This is what makes the rest of the pipeline source-agnostic and enables metadata-filtered (self-query) retrieval.

```python
# LangChain Document
Document(
    page_content="<clean text>",
    metadata={
        "doc_id":        "pmkisan-guidelines-2024",   # stable per source doc
        "chunk_id":      "pmkisan-guidelines-2024::0007",
        "parent_id":     "pmkisan-guidelines-2024::sec-3",  # for parent-doc retrieval
        "source_type":   "pdf",           # pdf | html | web
        "source_path":   "data/raw/pmkisan.pdf",
        "source_url":    None,            # set for web-scraped docs
        "scheme_name":   "PM-KISAN",
        "scheme_category": "income-support",   # income-support | credit | insurance | subsidy | ...
        "state":         "all-india",     # or specific state
        "department":    "Ministry of Agriculture & Farmers Welfare",
        "language":      "en",            # corpus is English-only by design
        "page_number":   4,               # PDFs only
        "ingested_at":   "2026-09-05T10:00:00Z",
    },
)
```

**Why these fields:**
- `scheme_name`, `scheme_category`, `state`, `department` → power **self-query** filtering ("only PM-KISAN", "only Tamil Nadu schemes").
- `parent_id` → **parent-document retriever** (embed small chunks, return the full section).
- `source_type` / `source_path` / `source_url` → traceability; every answer cites back here.
- `page_number` → citations point to a PDF page.

---

## 2. Chunking Strategy

Two-level, so retrieval is precise but context returned to the LLM is complete.

| Level | Splitter | Size | Overlap | Purpose |
|-------|----------|------|---------|---------|
| **Child** (embedded) | `RecursiveCharacterTextSplitter` | 500 chars | 80 | Precise dense matching |
| **Parent** (returned) | Section-level split | 500-4000 chars | — | Complete context to LLM |

**Two floors keep the corpus usable** (`MIN_PARENT_CHARS=500`,
`MIN_CHUNK_CHARS=120`). Without them, a heading heuristic applied to a
table-heavy government PDF treats every table cell as a section boundary:

| | without floors | with floors |
|---|---|---|
| chunks | 4,385 | 1,459 |
| median chunk | **25 chars** | 446 chars |
| chunks under 100 chars | 2,910 | **0** |
| parents (median) | 3,736 (16 chars) | 598 (734 chars) |

The "without" column is a real measurement from the first build of
`SCHEMES.pdf` — chunks were fragments like `'FW:'`, `'150'`, `'Implementing'`,
which cannot ground an answer and pollute both dense and BM25 retrieval.

- Use **semantic boundaries first** (headings, scheme sub-sections), then recursive character splitting inside.
- Preserve `scheme_name` / section headers into child metadata so filtering survives chunking.
- `chunk_size` is measured in **characters**, not tokens (`RecursiveCharacterTextSplitter`
  is character-based). 500/80 is the working pin.
- Section detection is heuristic: a short line (<= 12 words) that is upper/title-case,
  numbered, or ends in a colon starts a new parent. A heading-less document still
  splits every `PARENT_MAX_CHARS` (4000) so one giant parent never forms.
- **Verified:** identical input yields identical `chunk_id`s across rebuilds, and
  `scheme_name`/`scheme_category`/`state`/`department` survive normalization *and*
  chunking into every child.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=80,
    separators=["\n\n", "\n", ". ", " ", ""],
)
```

---

## 3. Embedding Configuration

| Setting | Value |
|---------|-------|
| Model | OpenAI `text-embedding-3-small` |
| Dimensions | 1536 |
| Normalize | **Yes** (L2-normalize → cosine via inner product) |
| Corpus language | English only (embedder never sees Indic text) |
| Batch | Embed offline in one pass during `build_index.py` |

```python
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
```

---

## 4. FAISS Index Type

| Decision | Choice | Reason |
|----------|--------|--------|
| Index | **`IndexFlatIP`** (inner product) | Exact search, perfect recall; corpus is small (thousands of chunks), so no ANN needed |
| Metric | Inner product on **normalized** vectors = cosine similarity |
| Backend | `faiss-cpu` | No GPU needed for search at this scale; simpler than `faiss-gpu` CUDA pinning |

> **Scale-up note (not needed now):** past ~1M vectors, switch to `IndexIVFFlat` (add `nlist` clusters + `nprobe` at query time) or `IndexHNSWFlat` for ANN. Documented here only so the choice is intentional.

```python
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

vs = FAISS.from_documents(
    child_docs, embeddings,
    normalize_L2=True,
    distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,   # -> IndexFlatIP
)
```

> **`distance_strategy` is required.** Without it LangChain builds an
> `IndexFlatL2` regardless of `normalize_L2`. Only `MAX_INNER_PRODUCT` selects
> `IndexFlatIP`. Verified: `type(vs.index).__name__ == "IndexFlatIP"`.
>
> **Expect this warning, and ignore it:**
> `UserWarning: Normalizing L2 is not applicable for metric type: MAX_INNER_PRODUCT`.
> LangChain skips its own L2 normalization on an IP index — which is harmless
> here because **OpenAI already returns unit-length embeddings**. Measured on the
> built index: every vector norm is exactly `1.000000`, so inner product *is*
> cosine similarity. Do not "fix" this by switching to L2.

---

## 5. Persistence

Build once, load at app start — consistent with the "no realtime, store to file" requirement.

```python
# Build (ingestion/build_index.py)
vs.save_local("data/processed/faiss_index")

# Load (app startup)
vs = FAISS.load_local(
    "data/processed/faiss_index",
    embeddings,
    allow_dangerous_deserialization=True,   # local, trusted index
)
```

Artifacts written to `data/processed/`:
```
faiss_index/
├── index.faiss        # the vectors
└── index.pkl          # docstore + id map (page_content + metadata)
```

Two JSONL files are written alongside the index:

| File | Contents | Read by |
|------|----------|---------|
| `data/processed/documents.jsonl` | Child chunks (embedded) | BM25, golden-set labelling, `--from-jsonl` rebuild |
| `data/processed/parents.jsonl` | Parent sections | Parent-document retrieval |

Both stay in lockstep with the index — every child carries the `parent_id` of
the section it came from, so a child hit can return its full parent section.

---

## 6. Wiring Into Hybrid Retrieval

FAISS provides the **dense** arm; BM25 provides the **lexical** arm; `EnsembleRetriever` fuses them; the cross-encoder reranks.

```python
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever

dense = vs.as_retriever(search_kwargs={"k": 10})
bm25  = BM25Retriever.from_documents(child_docs); bm25.k = 10

hybrid = EnsembleRetriever(retrievers=[dense, bm25], weights=[0.6, 0.4])

# + rerank (bge-reranker-v2-m3) via ContextualCompressionRetriever
compressed = ContextualCompressionRetriever(
    base_retriever=hybrid,
    base_compressor=reranker,   # cross-encoder wrapper
)
```

Self-query and parent-document retrievers wrap the same FAISS store:
- **Self-query** — LLM parses "PM-KISAN deadline in Tamil Nadu" into a metadata filter (`scheme_name=PM-KISAN`, `state=tamil-nadu`) + semantic query.
- **Parent-document** — child chunks are indexed in FAISS; on hit, `parent_id` fetches the full section from the docstore.

---

## 7. Metadata Filter Examples (Self-Query)

`scheme_name` / `scheme_category` / `state` are resolved **per section** at
ingest by `ingestion/normalize.py`, not per file — one 296-page document covers
dozens of schemes. Detection is a deterministic pattern table (₹0, reproducible);
a section that does not name a scheme inherits from the previous one for at most
`SCHEME_CARRY_FORWARD` (2) sections, because a *wrong* label is worse than none
when it is used as a filter.

Measured coverage on the current corpus: **scheme_name on 68% of chunks**
(1,005 of 1,457), across 14 distinct schemes. Document-wide facts (the Odisha
booklet's state and department) come from `data/raw/scheme_map.json` rather than
from the heuristic.


| Farmer question | Parsed filter |
|-----------------|---------------|
| "PM-KISAN eligibility" | `scheme_name == "PM-KISAN"` |
| "crop insurance schemes" | `scheme_category == "insurance"` |
| "subsidy schemes in Karnataka" | `scheme_category == "subsidy" AND state == "karnataka"` |
| "documents needed for KCC" | `scheme_name == "Kisan Credit Card"` |

---

## 8. Build Checklist

`build_index.py` enforces most of this automatically via `verify_build()`, which
raises rather than shipping a bad index.

| Check | Enforced by | Status |
|-------|-------------|--------|
| All loaders emit the common schema (§1) | `normalize.validate_metadata` (closed schema — unknown fields raise) | ✅ automated |
| Child/parent chunking applied (§2) | `split_parents` / `split_children` | ✅ automated |
| Vectors L2-normalized (§3) | `normalize_L2=True` | ✅ automated |
| `IndexFlatIP` built and saved (§4–5) | `distance_strategy=MAX_INNER_PRODUCT`; `verify_build` asserts `index.d == 1536` | ✅ automated |
| `documents.jsonl` + `parents.jsonl` written | `persist()` | ✅ automated |
| Chunk IDs unique | `verify_build` raises on duplicates | ✅ automated |
| Vector count == chunk count | `verify_build` raises on mismatch | ✅ automated |
| Golden-set chunk IDs still resolve | `_warn_on_golden_drift` — **warns**, does not raise | ⚠️ advisory |
| Hybrid + rerank returns cited chunks | Manual / `run_eval.py --stage 1` | ⚠️ needs a corpus |

**Navigation and menu text is stripped at load time** by hyperlink density
(`ingestion/load_html._strip_link_heavy_blocks`). Menu text is poison in a RAG
corpus: the TNAU page's nav chunk — *"ABOUT US CONTACT US State Level Schemes
Tamil Nadu"* — scored **0.997** against "Tamil Nadu schemes" while containing no
answer at all, so it displaced the real scheme sections and the groundedness
gate then rejected whatever the model wrote from it.

**Non-English source documents** are rejected at load time by `assert_english`
(a page whose alphabetic characters are <50% Latin is skipped with a warning),
so Indic text never reaches the OpenAI embedder.
