# Calculator API — FastAPI

A simple and powerful calculator with a JSON API and a small web UI. Expressions
are evaluated **safely** (no `eval()`): the input is parsed into an AST and only a
whitelist of math operations, functions, and constants is allowed.

## Features

- Arithmetic: `+ - * / // % **` and parentheses
- Scientific functions: `sqrt, cbrt, sin, cos, tan, asin, acos, atan, log, log10,
  log2, exp, factorial, floor, ceil, degrees, radians, gcd, hypot, min, max, ...`
- Constants: `pi, e, tau`
- Web UI with a live-preview paper tape that keeps a running history
- Interactive API docs at `/docs`

## Run

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Then open:
- **http://127.0.0.1:8000** — the calculator UI
- **http://127.0.0.1:8000/docs** — interactive API docs

## API

| Method | Path | Body / Query | Example result |
|---|---|---|---|
| POST | `/calculate` | `{"expression": "2 + 3 * sqrt(16)"}` | `{"result": 14.0}` |
| GET  | `/calculate` | `?expression=factorial(6)` | `{"result": 720}` |
| POST | `/add` `/subtract` `/multiply` `/divide` `/power` | `{"a": 5, "b": 2}` | `{"result": ...}` |
| GET  | `/functions` | — | list of supported functions |
| GET  | `/health` | — | `{"status": "ok"}` |

Errors (bad syntax, unknown function, division by zero) return HTTP 400 with a
clear `detail` message.

## Why it's safe

User input never reaches Python's `eval()`. `safe_eval()` parses the expression
with `ast.parse(..., mode="eval")` and walks the tree, permitting only numbers,
whitelisted operators, whitelisted function names, and known constants. Attempts
like `__import__("os")`, `open(...)`, or attribute access are rejected.

## Files

| File | Purpose |
|---|---|
| `main.py` | FastAPI app + safe evaluator |
| `static/index.html` | Web UI (paper-tape calculator) |
| `requirements.txt` | Dependencies |
