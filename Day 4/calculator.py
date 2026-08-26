"""
Simple & Powerful Calculator API — FastAPI
==========================================
A calculator backend that safely evaluates math expressions and exposes
both a JSON API and a small web UI.

Features
--------
- Safe expression evaluation (no eval()): parses the expression into an AST
  and only allows a whitelist of math operations and functions.
- Basic arithmetic: + - * / // % ** and parentheses.
- Scientific functions: sqrt, sin, cos, tan, log, log10, exp, factorial, ...
- Constants: pi, e, tau.
- Dedicated /calculate endpoint plus quick binary endpoints (add, subtract...).

Run
---
    pip install fastapi uvicorn
    uvicorn main:app --reload

Then open http://127.0.0.1:8000  (web UI)
API docs at        http://127.0.0.1:8000/docs
"""

import ast
import math
import operator
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(
    title="Calculator API",
    description="A simple and powerful calculator built with FastAPI.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Safe expression evaluator
# ---------------------------------------------------------------------------
# We never use Python's eval() on user input. Instead we parse the expression
# into an Abstract Syntax Tree and walk it, allowing only known-safe nodes.

# Binary operators we permit
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

# Unary operators (e.g. -5, +3)
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Whitelisted functions the user may call
_FUNCTIONS = {
    "sqrt": math.sqrt,
    "cbrt": lambda x: math.copysign(abs(x) ** (1 / 3), x),
    "abs": abs,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
    "factorial": math.factorial,
    "exp": math.exp,
    "log": math.log,        # log(x) natural, or log(x, base)
    "log10": math.log10,
    "log2": math.log2,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "degrees": math.degrees,
    "radians": math.radians,
    "gcd": math.gcd,
    "hypot": math.hypot,
    "pow": math.pow,
    "min": min,
    "max": max,
}

# Whitelisted constants
_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
}


def _eval_node(node):
    """Recursively evaluate a whitelisted AST node."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    # Numbers: 3, 4.5
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric constants are allowed.")

    # Names: pi, e, tau
    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise ValueError(f"Unknown name: '{node.id}'")

    # Binary operations: a + b
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BIN_OPS:
            raise ValueError(f"Operator not allowed: {op_type.__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return _BIN_OPS[op_type](left, right)

    # Unary operations: -a
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise ValueError(f"Unary operator not allowed: {op_type.__name__}")
        return _UNARY_OPS[op_type](_eval_node(node.operand))

    # Function calls: sqrt(2), log(8, 2)
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Invalid function call.")
        fname = node.func.id
        if fname not in _FUNCTIONS:
            raise ValueError(f"Unknown function: '{fname}'")
        args = [_eval_node(a) for a in node.args]
        return _FUNCTIONS[fname](*args)

    raise ValueError("Unsupported expression element.")


def safe_eval(expression: str) -> float:
    """Parse and safely evaluate a math expression string."""
    if not expression or not expression.strip():
        raise ValueError("Expression is empty.")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        raise ValueError("Invalid syntax in expression.")
    result = _eval_node(tree)
    return result


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class ExpressionRequest(BaseModel):
    expression: str


class BinaryRequest(BaseModel):
    a: float
    b: float


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
@app.post("/calculate")
def calculate(req: ExpressionRequest):
    """Evaluate a full math expression, e.g. '2 + 3 * sqrt(16)'."""
    try:
        result = safe_eval(req.expression)
    except ZeroDivisionError:
        raise HTTPException(status_code=400, detail="Division by zero.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not evaluate: {e}")
    return {"expression": req.expression, "result": result}


@app.get("/calculate")
def calculate_get(expression: str):
    """Same as POST /calculate but via query string: /calculate?expression=2%2B2"""
    return calculate(ExpressionRequest(expression=expression))


# Quick binary-operation endpoints
@app.post("/add")
def add(req: BinaryRequest):
    return {"result": req.a + req.b}


@app.post("/subtract")
def subtract(req: BinaryRequest):
    return {"result": req.a - req.b}


@app.post("/multiply")
def multiply(req: BinaryRequest):
    return {"result": req.a * req.b}


@app.post("/divide")
def divide(req: BinaryRequest):
    if req.b == 0:
        raise HTTPException(status_code=400, detail="Division by zero.")
    return {"result": req.a / req.b}


@app.post("/power")
def power(req: BinaryRequest):
    return {"result": req.a ** req.b}


@app.get("/functions")
def list_functions():
    """List every function and constant the calculator supports."""
    return {
        "functions": sorted(_FUNCTIONS.keys()),
        "constants": sorted(_CONSTANTS.keys()),
        "operators": ["+", "-", "*", "/", "//", "%", "**", "()"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Web UI (served from /static, index at /)
# ---------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")
