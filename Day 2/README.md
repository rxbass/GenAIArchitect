# Student Grade System

A simple command-line Python program that takes a mark (0–100) and prints the matching letter grade. Built as the CAIE course assignment — pure Python, standard library only, no frameworks.

## Grading scale

| Mark range | Grade |
|------------|-------|
| 90 – 100   | A     |
| 80 – 89    | B     |
| 70 – 79    | C     |
| 60 – 69    | D     |
| Below 60   | E     |

Boundaries are inclusive — a mark of exactly 90 is an A, exactly 80 is a B, and so on.

## How to run

```bash
# 1. Create and activate a virtual environment
python -m venv venv

# macOS / Linux
source venv/bin/activate
# Windows (PowerShell)
venv\Scripts\Activate.ps1

# 2. Run the program
python grade_system.py
```

You'll be prompted to enter a mark. Example run:

```
Enter your mark (0-100): 85
Mark: 85 -> Grade: B
```

## How invalid input is handled

The program guards against bad input two ways. Non-numeric input (e.g. `abc`) is caught with a `try/except` block — `int()` raises a `ValueError`, which is caught and answered with a friendly message instead of crashing. Numbers outside the valid range (below 0 or above 100) are caught by a range check that runs after the conversion succeeds, printing an out-of-range message. Only a valid whole number between 0 and 100 proceeds to grading.

## Files

- `grade_system.py` — the program (single file, standard library only)
