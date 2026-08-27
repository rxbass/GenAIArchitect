# Student Grade System

**Gen AI Architect Program · Student Grade Assignment**

An interactive student grade calculator built with [Streamlit](https://streamlit.io/). Enter a mark using the slider and instantly see the letter grade, a colour-coded result, and a motivational message.

---

## What it does

- Drag the **slider** to set a mark between 0 and 100
- Instantly shows the **letter grade** (A–E) with no button needed
- Displays a **progress bar** that fills up with the score
- Shows a **colour-coded message** with an emoji for each grade band
- Includes a **grading scale reference table** at the bottom

---

## Grading scale

| Mark | Grade | Message |
|---|---|---|
| 90 – 100 | A 🏆 | Excellent! Outstanding performance. |
| 80 – 89 | B 🥈 | Great job! Above average performance. |
| 70 – 79 | C 👍 | Good work! Average performance. |
| 60 – 69 | D 📚 | You passed, but there's room to improve. |
| Below 60 | E 💪 | Don't give up! Keep studying and try again. |

---

## Setup

**Step 1 — Create a project folder**
```bash
mkdir grade-system
cd grade-system
```

**Step 2 — Create and activate a virtual environment**
```bash
python -m venv venv

# Windows (Command Prompt)
venv\Scripts\activate.bat

# Windows (PowerShell)
venv\Scripts\Activate.ps1

# macOS / Linux
source venv/bin/activate
```

**Step 3 — Install Streamlit**
```bash
pip install streamlit
```

---

## Run

```bash
streamlit run grade_system_app.py
```

Then open **http://localhost:8501** in your browser.

To stop the app press `Ctrl + C` in the terminal, then deactivate the virtual environment:

```bash
deactivate
```

---

## Files

| File | Purpose |
|---|---|
| `grade_system_app.py` | The Streamlit app |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

---

## Requirements

```
streamlit
```

Or install from a `requirements.txt`:

```bash
pip install -r requirements.txt
```
