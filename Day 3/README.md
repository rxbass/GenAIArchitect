# Daily Report Bot — PyAutoGUI Automation

**Gen AI Architect Program · Assignment 1**

A desktop automation bot that prepares a daily status report on its own. It drives Chrome and Microsoft Excel exactly the way a person would — moving the mouse, clicking, typing, and copying data on screen — using [PyAutoGUI](https://pyautogui.readthedocs.io/).

---

## Scenario

As an operations executive, you need to prepare a daily status report every morning. This bot automates the whole routine: it opens a browser, fetches a live value from a public website, records it in Excel alongside the current date/time and a short comment, saves the file with today's date, and captures a screenshot of the result.

---

## What the bot does

The bot performs this sequence unattended:

1. Opens Chrome and navigates to a public website (here: [Groww share-market page](https://groww.in/share-market-today)).
2. Searches for a stock (HDFC Bank) and copies the live price from the page.
3. Opens Microsoft Excel.
4. Creates a new row with three things: today's date & time, the fetched data, and a short comment.
5. Saves the Excel file with today's date in the filename, e.g. `daily_report_2026-08-25.xlsx`.
6. Takes a screenshot of the final Excel sheet and saves it.

---

## Requirements

| Requirement | Status |
|---|---|
| PyAutoGUI as the core automation library | Yes |
| Date & time generated automatically at run time | Yes — via `datetime.now()` |
| Filename includes current date in `YYYY-MM-DD` format | Yes |
| Saves both the Excel file and a screenshot | Yes |
| All code in a single file | Yes — `daily_report_bot.py` |

---

## Setup

1. **Create and activate a virtual environment** (recommended):

   ```bash
   python -m venv myenv
   # Windows
   myenv\Scripts\activate
   # macOS / Linux
   source myenv/bin/activate
   ```

2. **Install dependencies:**

   ```bash
   pip install pyautogui pillow
   ```

   `pillow` is required by PyAutoGUI for taking screenshots. On Windows, PyAutoGUI also pulls in `pyscreeze`, `pymsgbox`, and `pygetwindow` automatically.

---

## How to run

```bash
python daily_report_bot.py
```

Once started, the script takes over your mouse and keyboard, so **don't touch the machine while it runs.** Give it a clear screen and let it work through the steps.

---

## Configuration

Because PyAutoGUI clicks at fixed screen positions, a few coordinates are specific to *your* display resolution and layout. Before your first run, tune these values near the top of each step:

| What | Variable / line | How to find it |
|---|---|---|
| Search bar position | `moveTo(1200, 180 ...)` | Hover the mouse over the target and run `print(pyautogui.position())` |
| Stock price location | `moveTo(360, 500 ...)` | Same method |
| First Excel cell | `moveTo(700, 300 ...)` | Same method |
| Save directory | `save_dir = r'D:\...'` | Set to a real folder on your machine |

To discover a coordinate:

```python
import pyautogui, time
time.sleep(5)              # 5 seconds to position your mouse
print(pyautogui.position())
```

---

## Safety features

- **Failsafe:** `pyautogui.FAILSAFE = True` — slam the mouse into the **top-left corner** of the screen at any time to abort the script instantly.
- **Pacing:** `pyautogui.PAUSE = 1.0` forces a one-second gap after every action, giving apps time to respond and giving you time to intervene.
- **Modifier reset:** `reset_modifiers()` releases any stuck Alt/Ctrl/Shift keys before critical steps (like saving), preventing keystrokes from going to the wrong place.
- **Folder safety:** `os.makedirs(..., exist_ok=True)` creates the output folder if it doesn't exist — the Save As dialog won't create missing folders on its own.

---

## Notes & known limitations

- **Timing sensitivity.** GUI automation depends on apps opening and gaining focus in time. If a window loads slower than the `time.sleep()` allows, a keystroke can miss. The waits are set generously; increase them on a slower machine.
- **Coordinate fragility.** Hardcoded positions break if the screen resolution, display scaling, or page layout changes. Keep display scaling at 100% while testing.
- **The copy step is the fragile part.** Drag-selecting a value from a modern JavaScript website is inherently unreliable — if the value sits inside a link or button rather than plain text, the copy may come up empty. This is a property of the site, not a bug in the code.
- **`#####` in Excel** is not an error — it means a column is too narrow to display a value. The optional AutoFit step (`Alt → H → O → I`) widens columns to fix this.
- **Admin focus.** On Windows, if a target window runs as Administrator but the script does not (or vice versa), Windows silently blocks the automated input. Run your terminal as Administrator if input isn't registering.


---

## Files

| File | Purpose |
|---|---|
| `daily_report_bot.py` | The complete automation script |
| `daily_report_YYYY-MM-DD.xlsx` | Generated report (created at run time) |
| `screenshot_daily_report_YYYY-MM-DD.png` | Screenshot of the final sheet (created at run time) |
| `README.md` | This file |