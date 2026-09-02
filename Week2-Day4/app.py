"""
app.py — RAG (Retrieval-Augmented Generation) app.

Upload a PDF, the app splits it into chunks, embeds them, and stores them in a
FAISS vector database. Once the database is ready, ask questions and get answers
grounded in the PDF's contents.

Stack: Streamlit (UI) + LangChain (retrieval) + OpenAI (embeddings + chat model)
       + FAISS (vector store) + LangSmith (tracing/observability).

The OpenAI key is read from the OPENAI_API_KEY environment variable. It is never
hard-coded and must never be committed to source control.

LangSmith tracing turns on automatically when LANGSMITH_API_KEY is set in the
environment. Every retrieval + generation run is then logged to your LangSmith
project so you can inspect prompts, retrieved chunks, latency, and token cost.

Run with:
    streamlit run app.py
"""

import os
import hashlib
import tempfile

import streamlit as st

# Load variables from a local .env file (OPENAI_API_KEY, LANGSMITH_*, etc.) into
# the environment before anything reads them. Safe no-op if python-dotenv isn't
# installed or there's no .env file.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain


# --------------------------------------------------------------------------- #
# Configuration (all overridable via environment variables)
# --------------------------------------------------------------------------- #
CHAT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 200))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 100))
TOP_K = int(os.environ.get("TOP_K", 4))


# --------------------------------------------------------------------------- #
# Observability — LangSmith tracing
# --------------------------------------------------------------------------- #
def configure_langsmith():
    """Enable LangSmith tracing when an API key is present.

    LangChain auto-traces every run once the tracing env vars are set, so all we
    do here is detect the key and normalise both the new (LANGSMITH_*) and legacy
    (LANGCHAIN_*) variable names so any LangChain version picks them up. Setting
    LANGSMITH_TRACING=false explicitly opts out even when a key is present.

    Returns (enabled: bool, project_name: str | None, endpoint: str | None).
    """
    api_key = os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")
    tracing_flag = (
        os.environ.get("LANGSMITH_TRACING")
        or os.environ.get("LANGCHAIN_TRACING_V2")
        or ""
    ).lower()

    # No key, or an explicit opt-out, means no tracing.
    if not api_key or tracing_flag == "false":
        return False, None, None

    project = (
        os.environ.get("LANGSMITH_PROJECT")
        or os.environ.get("LANGCHAIN_PROJECT")
        or "rag-pdf-app"
    )
    # Region endpoint matters — e.g. APAC users must post to the APAC host or
    # tracing silently fails. Default to the US host if none is provided.
    endpoint = (
        os.environ.get("LANGSMITH_ENDPOINT")
        or os.environ.get("LANGCHAIN_ENDPOINT")
        or "https://api.smith.langchain.com"
    )

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"      # legacy alias
    os.environ["LANGSMITH_API_KEY"] = api_key
    os.environ["LANGCHAIN_API_KEY"] = api_key         # legacy alias
    os.environ["LANGSMITH_PROJECT"] = project
    os.environ["LANGCHAIN_PROJECT"] = project         # legacy alias
    os.environ["LANGSMITH_ENDPOINT"] = endpoint
    os.environ["LANGCHAIN_ENDPOINT"] = endpoint       # legacy alias
    return True, project, endpoint


# --------------------------------------------------------------------------- #
# Core pipeline
# --------------------------------------------------------------------------- #
def build_vectorstore(file_bytes: bytes):
    """Load the PDF, split it, embed the chunks, and return a FAISS store.

    Returns None if the PDF has no extractable text (e.g. a scanned document
    with no OCR layer), so the caller can show a clear message instead of
    building an empty, useless index.
    """
    # PyPDFLoader needs a file path, so write the uploaded bytes to a temp file.
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        docs = PyPDFLoader(tmp_path).load()
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # Guard: PDF with no readable text (scanned images, empty file, etc.)
    if not "".join(doc.page_content for doc in docs).strip():
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    if not chunks:
        return None

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return FAISS.from_documents(chunks, embeddings)


def build_rag_chain(vectorstore):
    """Wire the retriever + chat model into a LangChain retrieval chain."""
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    system_prompt = (
        "You are an assistant that answers questions about an uploaded PDF. "
        "Use ONLY the retrieved context below to answer. If the answer is not "
        "contained in the context, say you could not find it in the document — "
        "do not make anything up. Keep answers concise and grounded.\n\n"
        "Context:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{input}")]
    )
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, combine_docs_chain)


# --------------------------------------------------------------------------- #
# Streamlit UI
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Chat with your PDF", page_icon="📄")
st.title("📄 Chat with your PDF")
st.caption("Upload a PDF, wait for the vector database to build, then ask questions.")

# Fail fast and clearly if the key is missing — everything downstream needs it.
if not os.environ.get("OPENAI_API_KEY"):
    st.error(
        "OPENAI_API_KEY is not set. Set it as an environment variable before "
        "running:\n\n"
        "  macOS/Linux:  export OPENAI_API_KEY=\"your-key\"\n"
        "  PowerShell:   $env:OPENAI_API_KEY=\"your-key\""
    )
    st.stop()

# Turn on LangSmith tracing if a key is present, and show its status.
tracing_enabled, langsmith_project, langsmith_endpoint = configure_langsmith()
with st.sidebar:
    st.subheader("Observability")
    if tracing_enabled:
        st.success(f"LangSmith tracing on\nProject: {langsmith_project}")
        st.caption(f"Endpoint: {langsmith_endpoint}")
    else:
        st.info("LangSmith tracing off")
        st.caption("Set LANGSMITH_API_KEY to enable tracing.")

# session_state persists across Streamlit reruns. We key the built index on a
# hash of the file's contents so that typing in the query box (which reruns the
# whole script) does NOT trigger a rebuild — only a genuinely new file does.
if "file_hash" not in st.session_state:
    st.session_state.file_hash = None
    st.session_state.vectorstore = None

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    # Build only when the uploaded file is new (different content hash).
    if file_hash != st.session_state.file_hash:
        with st.status("Building vector database...", expanded=True) as status:
            st.write("Reading and splitting the PDF...")
            st.write("Creating embeddings and indexing...")
            vectorstore = build_vectorstore(file_bytes)

            if vectorstore is None:
                status.update(label="No readable text found", state="error")
                st.error(
                    "This PDF has no extractable text. It is likely a scanned "
                    "document — run it through OCR and try again."
                )
                # Reset so the query box stays disabled.
                st.session_state.file_hash = None
                st.session_state.vectorstore = None
            else:
                st.session_state.file_hash = file_hash
                st.session_state.vectorstore = vectorstore
                status.update(label="Vector database ready ✅", state="complete")

# Query box is only enabled once a database is ready.
ready = st.session_state.vectorstore is not None

if ready:
    st.success("Database ready. Ask a question about your PDF below.")

question = st.text_input(
    "Ask a question about the PDF",
    placeholder="e.g. What are the key findings?",
    disabled=not ready,
)

if ready and question:
    with st.spinner("Thinking..."):
        chain = build_rag_chain(st.session_state.vectorstore)
        # A named, tagged run makes this query easy to find and filter in
        # LangSmith. The metadata is attached to the trace, not the prompt.
        result = chain.invoke(
            {"input": question},
            config={
                "run_name": "pdf_rag_query",
                "tags": ["rag", "pdf", "streamlit"],
                "metadata": {
                    "chat_model": CHAT_MODEL,
                    "embedding_model": EMBEDDING_MODEL,
                    "top_k": TOP_K,
                    "file_hash": st.session_state.file_hash[:12],
                },
            },
        )

    st.markdown("### Answer")
    st.write(result["answer"])

    # Show the source chunks the answer was grounded in — useful for trust.
    with st.expander("Show source passages"):
        for i, doc in enumerate(result.get("context", []), start=1):
            page = doc.metadata.get("page")
            label = f"Passage {i}" + (f" (page {page + 1})" if page is not None else "")
            st.markdown(f"**{label}**")
            st.write(doc.page_content)
            st.divider()