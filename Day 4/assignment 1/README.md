# Basic FastAPI App

**Gen AI Architect Program · Assignment 4**

A small web API built with [FastAPI](https://fastapi.tiangolo.com/) that demonstrates endpoints, path parameters, query parameters, and automatic interactive documentation.

---

## What it does

| Endpoint | Type | Example | Response |
|---|---|---|---|
| `GET /` | Root | `/` | `{"message": "Hello, FastAPI"}` |
| `GET /greet/{name}` | Path parameter | `/greet/Jo` | `{"message": "Hello, Jo! Welcome to FastAPI."}` |
| `GET /square?number=5` | Query parameter | `/square?number=9` | `{"number": 9.0, "square": 81.0}` |

FastAPI also generates interactive API docs automatically at `/docs` — no extra work needed.

---

## Setup

### Step 1 — Create a project folder
```bash
mkdir fastapi-basic-app
cd fastapi-basic-app
```

### Step 2 — Create a virtual environment
```bash
python -m venv venv
```

### Step 3 — Activate the virtual environment
```bash
# Windows (Command Prompt)
venv\Scripts\activate.bat

# Windows (PowerShell)
venv\Scripts\Activate.ps1

# macOS / Linux
source venv/bin/activate
```
You will see `(venv)` at the start of your terminal prompt when it is active.

### Step 4 — Install dependencies
```bash
pip install fastapi uvicorn
```

---

## Run

```bash
uvicorn main:app --reload
```

The `--reload` flag restarts the server automatically whenever you save a change to `main.py`.

---

## Try it

Once the server is running, open these URLs in your browser:

| URL | What you see |
|---|---|
| http://127.0.0.1:8000 | Root JSON response |
| http://127.0.0.1:8000/greet/Jo | Greeting with your name |
| http://127.0.0.1:8000/square?number=9 | Square of 9 |
| http://127.0.0.1:8000/docs | Interactive API docs |

The `/docs` page (Swagger UI) lets you call every endpoint directly from the browser — no extra tool needed.

---

## Stop the server

Press `Ctrl + C` in the terminal to stop Uvicorn, then deactivate the virtual environment:

```bash
deactivate
```

---

## How FastAPI works — the key ideas

**Decorator = endpoint.** `@app.get("/")` tells FastAPI to handle HTTP GET requests to `/` with the function below it.

**Type hints = automatic validation.** When you write `def greet(name: str)` or `def square(number: float)`, FastAPI reads those types and validates the incoming value automatically. If a caller passes a non-number to `/square`, FastAPI returns a clear 422 error without any extra code.

**Path vs query parameters.** A name inside curly braces (`/greet/{name}`) is a path parameter — part of the URL itself. A name without braces (`number` in `/square`) is a query parameter — passed after the `?` in the URL.

**Docs are free.** FastAPI reads your function names, type hints, and docstrings and builds the interactive `/docs` page from them automatically. Clear names and good docstrings make the docs useful with zero extra effort.

---

## Files

| File | Purpose |
|---|---|
| `main.py` | The complete FastAPI application |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |