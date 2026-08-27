"""
Assignment 4 - Basic FastAPI App
=================================
A small web API built with FastAPI that demonstrates:
  - A root GET / endpoint returning a welcome message
  - A path parameter endpoint  GET /greet/{name}
  - A query parameter endpoint GET /square?number=5
  - Automatic interactive docs at /docs

Run:
    uvicorn main:app --reload

Then open:
    http://127.0.0.1:8000        → root endpoint
    http://127.0.0.1:8000/docs   → interactive API docs
"""

from fastapi import FastAPI

# Create the FastAPI app instance
app = FastAPI(
    title="Basic FastAPI App",
    description="Assignment 4 — a simple API built with FastAPI.",
    version="1.0.0",
)


# ------------------------------------------------------------------
# Root endpoint
# GET /
# ------------------------------------------------------------------
@app.get("/")
def root():
    """Return a welcome message."""
    return {"message": "Hello, Social Eagle! Welcome to FastAPI."}


# ------------------------------------------------------------------
# Path parameter endpoint
# GET /greet/{name}
# ------------------------------------------------------------------
@app.get("/greet/{name}")
def greet(name: str):
    """
    Greet a person by name.

    - **name**: the name to greet (supplied in the URL path)

    Example: /greet/Jo  →  {"message": "Hello, Jo! Welcome to FastAPI."}
    """
    return {"message": f"Hello, {name}! Welcome to FastAPI."}


# ------------------------------------------------------------------
# Query parameter endpoint
# GET /square?number=5
# ------------------------------------------------------------------
@app.get("/square")
def square(number: float):
    """
    Return the square of a number.

    - **number**: the value to square (supplied as a query parameter)

    Example: /square?number=5  →  {"number": 5.0, "square": 25.0}
    """
    return {"number": number, "square": number ** 2}