# RAG App — Chat with your PDF

A Retrieval-Augmented Generation app. Upload a PDF, and it is chunked, embedded,
and stored in a FAISS vector database. Once the database is ready, ask questions
and get answers grounded in the PDF's contents.

**Stack:** Streamlit · LangChain · OpenAI · FAISS · LangSmith (tracing)

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate            # macOS/Linux
# venv\Scripts\Activate.ps1         # Windows PowerShell

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your OpenAI API key (never commit this)
export OPENAI_API_KEY="your-openai-key"        # macOS/Linux
# $env:OPENAI_API_KEY="your-openai-key"         # Windows PowerShell

# 4. (Optional) Enable LangSmith tracing — just set the key
export LANGSMITH_API_KEY="your-langsmith-key"   # macOS/Linux
# $env:LANGSMITH_API_KEY="your-langsmith-key"    # Windows PowerShell

# 5. Run
streamlit run app.py
```

## Tracing with LangSmith

Tracing turns on automatically when `LANGSMITH_API_KEY` is set — no code changes
needed. Every query is logged to your LangSmith project, where you can inspect
the retrieved chunks, the exact prompt sent to the model, latency, and token
cost. The sidebar shows whether tracing is on and which project it writes to.

Get a key from [smith.langchain.com](https://smith.langchain.com) → Settings →
API Keys. To turn tracing off even with a key set, use
`export LANGSMITH_TRACING=false`.

The app opens at http://localhost:8501.

## How it works

1. Upload a PDF.
2. The app shows a **Building vector database...** state while it reads, splits,
   embeds, and indexes the document.
3. Once the index is ready, the query box is enabled.
4. Ask a question — the retriever pulls the most relevant chunks and the OpenAI
   model answers using only that context. Source passages are shown for
   transparency.

## Notes

- The API key is read from the `OPENAI_API_KEY` environment variable — it is
  never hard-coded.
- The index is cached against a hash of the uploaded file, so typing in the
  query box does **not** rebuild the database. Only a new file triggers a
  rebuild.
- If the PDF has no extractable text (e.g. a scanned document with no OCR
  layer), the app says so clearly instead of building an empty index.

## Configuration (optional environment variables)

| Variable                  | Default                  | Purpose                       |
|---------------------------|--------------------------|-------------------------------|
| `OPENAI_MODEL`            | `gpt-4o-mini`            | Chat model                    |
| `OPENAI_EMBEDDING_MODEL`  | `text-embedding-3-small` | Embedding model               |
| `CHUNK_SIZE`              | `1000`                   | Characters per chunk          |
| `CHUNK_OVERLAP`           | `150`                    | Overlap between chunks        |
| `TOP_K`                   | `4`                      | Chunks retrieved per question |
| `LANGSMITH_API_KEY`       | *(unset)*                | Enables LangSmith tracing     |
| `LANGSMITH_PROJECT`       | `rag-demo`               | LangSmith project name        |
| `LANGSMITH_TRACING`       | `true` when key is set   | Set to `false` to opt out     |