# LangChain Basics — from first model call to a hardened RAG chatbot

A hands-on project for learning the core building blocks of GenAI apps with
LangChain, one concept at a time. Everything is built around a single theme —
a **Customer Feedback / Support Assistant** for a fictional SaaS product,
"NimbusCRM" — so the pieces connect instead of floating around as disconnected
snippets.

Written for **LangChain 1.x** (the current major line) and **OpenAI**
(`gpt-4.1-nano`, the cheapest option — fine for all of this).

## The one idea to hold onto

You build small components (a prompt, a model, a parser) and snap them together
with the pipe operator `|` into a **chain**. Data flows left to right, exactly
like a shell pipe:

```
{"feedback": "..."}  ->  prompt  ->  model  ->  parser  ->  clean result
```

Once that clicks, everything else — structured output, RAG, guardrails — is
just fancier components plugged into the same pipe.

## What's inside

Two runnable programs:

| File | What it is | Run with |
|------|------------|----------|
| `app.py` | A guided tour of the 6 core concepts. Runs top to bottom and prints each result so you can *watch* each idea work. | `python app.py` |
| `rag_chat.py` | A conversational RAG chatbot with a Streamlit web UI, hardened against prompt injection. The "real app" the tour builds toward. | `streamlit run rag_chat.py` |

Start with `app.py` to learn the primitives, then move to `rag_chat.py` to see
them assembled into something usable.

## Setup

You need Python 3.10+ and an OpenAI API key.

```bash
# 1. (optional) a clean virtual environment
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt          # for app.py
pip install -r requirements-rag.txt      # additionally, for rag_chat.py

# 3. add your key
cp .env.example .env                      # then edit .env and paste your key
```

Your `.env` should contain:

```
OPENAI_API_KEY=sk-your-key-here
```

Get a key from https://platform.openai.com/api-keys. Everything here costs a
fraction of a cent to run on `gpt-4.1-nano`.

## Part 1 — the concept tour (`app.py`)

Run `python app.py` and it walks through six steps in order:

1. **The model** — your direct line to the LLM. `model.invoke("...")` returns a
   message object; the text lives in `.content`. This is GenAI with zero
   framework around it.
2. **Prompt templates** — a form letter with blanks (`{feedback}`). Write the
   instructions once, fill the blanks later, reuse everywhere.
3. **Your first chain** — `prompt | model | parser`. The pipe wires components;
   the output parser turns the model's reply into a plain string.
4. **Structured output** — instead of free text, the model fills a *shape* you
   define (with Pydantic). You get back typed fields — `sentiment`, `urgency`,
   `churn_risk` — ready to drop into a database or CRM ticket.
5. **Composing chains** — one chain's output (plus your own Python logic) drives
   the next. Here the analysis picks the *tone* of an auto-drafted reply:
   LLM → your code → LLM.
6. **Streaming** — the same chain, called with `.stream()` instead of
   `.invoke()`, for the live "typing" effect you see in chat UIs.

Try changing `SAMPLE_FEEDBACK` at the top and re-running — the structured fields
and the drafted reply change with it.

## Part 2 — the RAG chatbot (`rag_chat.py`)

Run `streamlit run rag_chat.py` and a chat window opens in your browser. Ask
about plans, billing, support, or security; try a follow-up like "what about
Enterprise?"; expand **Sources used** under any reply to see which chunks were
retrieved.

**What RAG adds.** The tour app answered from the model's own general knowledge.
RAG (Retrieval-Augmented Generation) makes it answer from *your* documents: it
finds the few chunks most relevant to the question and pastes them into the
prompt as context, so the bot can talk about your product — and stays honest,
because it's told to answer only from what it was given.

The retrieval pipeline (in `build_retriever`):

- **Document** — LangChain's standard text container.
- **Splitting** — long text is chopped into chunks so retrieval returns just the
  relevant piece.
- **Embeddings** — each chunk becomes a vector (numbers capturing its meaning);
  similar meaning = nearby vectors.
- **Vector store** — holds the vectors and searches them. This project uses
  `InMemoryVectorStore` (zero setup); swapping to **FAISS** or **Chroma** is a
  one-line change (see the commented lines — the rest of the code is identical).
- **Retriever** — given a question, returns the `k` most relevant chunks.

**What makes it conversational.** A follow-up like "what about Enterprise?" has
no subject, so it retrieves badly on its own. `condense_question` runs a small
chain that rewrites it into a standalone question ("How much is the NimbusCRM
Enterprise plan?") using the chat history, *then* searches.

**Memory.** The conversation lives in `st.session_state`. A `MEMORY_WINDOW`
sends only the last few messages to the model (to control cost and context size)
while the UI still displays the whole chat.

### Security: defense in depth

A single blocklist is easily bypassed (e.g. "ignore **all** previous
instructions" defeats a match on "ignore previous instructions"). So the app
layers several independent defenses, assuming any one can fail:

- **Layer 1 — rule-based input filter** (`check_input`): fast and free; regex +
  text normalisation so filler words and punctuation can't split the match.
  Still beatable by rewording — it's the cheap first pass.
- **Layer 2 — LLM classifier** (`llm_guard`, toggle with `USE_LLM_GUARD`): a
  tiny model call that judges intent, generalising beyond a fixed word list.
- **Layer 3 — hardened prompt** (`ANSWER_PROMPT`): instructs the model to treat
  user text and retrieved context as *data to answer about*, never as commands.
- **Layer 4 — limited blast radius**: the bot has no tools, so even a successful
  injection can only make it say something silly — it can't touch data, send
  email, or spend money. This is the most important layer.

**Honest note:** prompt injection is an open problem — none of this makes it
impossible. The goal is to make it *hard and harmless*, not "solved". Security
scales with what the model can *do*, so keeping its powers minimal matters more
than any filter.


## Switching models or providers

Both apps set the model in one place at the top:

```python
MODEL_NAME = "openai:gpt-4.1-nano"
```

The string is `"provider:model-name"`. To switch, change this line, install that
provider's package, and set its API key:

- `"openai:gpt-4.1-mini"` — a bit smarter, still cheap
- `"anthropic:claude-haiku-4-5"` — needs `langchain-anthropic`, `ANTHROPIC_API_KEY`
- `"google_genai:gemini-2.0-flash"` — needs `langchain-google-genai`, `GOOGLE_API_KEY`

Heads up: many older tutorials use `gpt-4o-mini`, which OpenAI is deprecating.
If a model name errors, get a current one from
https://platform.openai.com/docs/models.

## Ideas to try

- Edit `KNOWLEDGE_BASE` in `rag_chat.py` to your own facts and re-run.
- Set `MEMORY_WINDOW = 2`, have a long chat, then ask about the first turn — the
  bot will have "forgotten" it. That's the cost/memory tradeoff made visible.
- Set `k = 1` on the retriever and ask something that needs two chunks, to feel
  why `k` matters.
- Swap `InMemoryVectorStore` for FAISS or Chroma using the commented lines.
- Load real files instead of hardcoded strings (a document loader replaces the
  `KNOWLEDGE_BASE` list; the pipeline downstream stays the same).

## Where this goes next

- **Agents** (`create_agent` in LangChain 1.x) — the model decides which tools
  to call, and your retriever becomes one tool among several. This is where the
  "limit the blast radius / confirm before acting" security patterns become
  essential, because the model can now *act*.
- **Better retrieval** — similarity thresholds, metadata filtering, hybrid
  (keyword + semantic) search, and re-ranking.
- **Persistent memory** — store conversations in a database keyed by user, so
  they survive and resume across sessions.

## Version note

Built against **LangChain 1.x**. Older 0.x tutorials use different import paths;
the pipe/chain idea is the same, but package layouts moved, so prefer the 1.x
docs at https://docs.langchain.com/oss/python.