# Gen AI Architect Program — Automation Assignments

**learn · build · operate**

This repository contains two automation assignments that each drive real desktop / browser software the way a person would.

| # | Assignment | Core tool | File |
|---|---|---|---|
| 1 | Daily Report Bot | PyAutoGUI | `daily_report_bot.py` |
| 2 | WhatsApp Sender + Smart Data Extractor | Playwright | `playwright_assign.py` |

---

# Assignment 1 — Daily Report Bot (PyAutoGUI)

A desktop automation bot that prepares a daily status report on its own. It drives Chrome and Microsoft Excel exactly the way a person would — moving the mouse, clicking, typing, and copying data on screen — using [PyAutoGUI](https://pyautogui.readthedocs.io/).

## Scenario

As an operations executive, you need to prepare a daily status report every morning. This bot automates the whole routine: it opens a browser, fetches a live value from a public website, records it in Excel alongside the current date/time and a short comment, saves the file with today's date, and captures a screenshot of the result.

## What the bot does

1. Opens Chrome and navigates to a public website (here: [Groww share-market page](https://groww.in/share-market-today)).
2. Searches for a stock (HDFC Bank) and copies the live price from the page.
3. Opens Microsoft Excel.
4. Creates a new row with three things: today's date & time, the fetched data, and a short comment.
5. Saves the Excel file with today's date in the filename, e.g. `daily_report_2026-08-25.xlsx`.
6. Takes a screenshot of the final Excel sheet and saves it.

## Requirements

| Requirement | Status |
|---|---|
| PyAutoGUI as the core automation library | Yes |
| Date & time generated automatically at run time | Yes — via `datetime.now()` |
| Filename includes current date in `YYYY-MM-DD` format | Yes |
| Saves both the Excel file and a screenshot | Yes |
| All code in a single file | Yes — `daily_report_bot.py` |

## How to run

```bash
pip install pyautogui pillow
python daily_report_bot.py
```

Once started, the script takes over your mouse and keyboard, so **don't touch the machine while it runs.** Give it a clear screen and let it work through the steps.

## Configuration

Because PyAutoGUI clicks at fixed screen positions, a few coordinates are specific to *your* display resolution and layout. Before your first run, tune these values:

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

## Safety features

- **Failsafe:** `pyautogui.FAILSAFE = True` — slam the mouse into the **top-left corner** of the screen at any time to abort the script instantly.
- **Pacing:** `pyautogui.PAUSE = 1.0` forces a one-second gap after every action.
- **Modifier reset:** `reset_modifiers()` releases any stuck Alt/Ctrl/Shift keys before critical steps (like saving).
- **Folder safety:** `os.makedirs(..., exist_ok=True)` creates the output folder if it doesn't exist.

## Notes & known limitations

- **Timing sensitivity.** GUI automation depends on apps opening and gaining focus in time. If a window loads slower than the `time.sleep()` allows, a keystroke can miss. Increase the waits on a slower machine.
- **Coordinate fragility.** Hardcoded positions break if the screen resolution, display scaling, or page layout changes. Keep display scaling at 100% while testing.
- **The copy step is the fragile part.** Drag-selecting a value from a modern JavaScript website is inherently unreliable — if the value sits inside a link or button rather than plain text, the copy may come up empty. This is a property of the site, not a bug in the code.
- **`#####` in Excel** is not an error — it means a column is too narrow to display a value. The optional AutoFit step (`Alt → H → O → I`) widens columns to fix this.
- **Admin focus.** On Windows, if a target window runs as Administrator but the script does not (or vice versa), Windows silently blocks the automated input. Run your terminal as Administrator if input isn't registering.

---

# Assignment 2 — WhatsApp Sender + Smart Data Extractor (Playwright)

A [Playwright](https://playwright.dev/python/)-powered bot that drives **WhatsApp Web** in a real browser to send personalized messages to a list of contacts and extract recent chat data back out.

## Scenario

You want to automate daily customer and business communication. The bot logs in to WhatsApp Web, reads a contact list from Excel, sends each person a personalized message, confirms it was sent, takes a screenshot, extracts the last few messages from the chat, and saves a full report.

> **Use responsibly.** Only message people who have agreed to hear from you. WhatsApp actively bans automation that spams — the built-in random delays are there both to look human and to protect your account. Keep the contact list small and consenting.

## What the bot does

1. Logs in to WhatsApp Web (scan the QR code manually on the first run; the session is then remembered).
2. Reads contacts from `contacts.xlsx` (columns: **Name**, **Phone** with country code, **Message** — an optional template).
3. For each contact: opens the chat by number, sends a personalized message (`{name}` is replaced with the real name), waits for the message to be sent, and takes a screenshot.
4. **Smart data extraction:** opens the chat and extracts the last 3 messages.
5. Saves a dated report as both JSON and Excel.

## Must-implement checklist

| Requirement | How it's met |
|---|---|
| Proper waits (`wait_for_selector`, `wait_for_timeout`) | Used for login, message box, delivery ticks, and message rows |
| Search a contact by number or name | Reaches chats via the `wa.me` deep link by phone number |
| Type and send a message reliably | `get_message_box()` tries multiple selectors + placeholder text |
| Random delays (2–5 s) between actions | `human_delay()` helper sleeps a random interval |
| Error handling (e.g. contact not found) | Invalid / not-on-WhatsApp numbers are caught and logged, run continues |
| Save the sent status for each contact | Recorded per contact in both reports |
| All code in a single file | `playwright_assign.py` |

## Reports produced

At the end of a run the bot saves two dated files:

- `whatsapp_report_YYYY-MM-DD.json` — full details per contact (status, timestamp, extracted messages, screenshot path, any error).
- `whatsapp_report_YYYY-MM-DD.xlsx` — a summary table.

Screenshots are saved per contact in a `screenshots/` folder.

## Setup

```bash
# create & activate a virtual environment first (see below), then:
pip install playwright openpyxl pandas
playwright install chromium
```

> **Note:** `playwright install` downloads the actual browser binaries, which is a **separate step** from `pip install playwright`. Anyone setting up from `requirements.txt` still needs to run `playwright install chromium` afterwards. Do **not** `pip install chromium` — the PyPI package by that name is an empty placeholder and unrelated to Playwright.

## Prepare your contacts

Create `contacts.xlsx` with these columns:

| Name | Phone | Message |
|---|---|---|
| Alice | +919876543210 | Hi {name}, hope you are doing well! |
| Bob | +919812345678 | *(blank — a default template is used)* |

- **Phone** must include the country code (e.g. `+91...`).
- **Message** is optional; `{name}` is replaced with the contact's name. If blank, a default template is used.

## How to run

```bash
python playwright_assign.py
```

On the first run, scan the QR code with your phone. The session is stored in the `whatsapp_session/` folder, so subsequent runs log in automatically.

## Notes & known limitations

- **Selector drift is the main gotcha.** WhatsApp Web changes its HTML periodically, so element selectors (IDs, `aria-label`s, class names like `message-in` / `message-out`) can shift after an update. The script uses several fallback selectors and a polling loop for login, but if a step can't find something, an outdated selector is the likeliest cause. Two helper scripts — `diagnose_whatsapp.py` and `diagnose_whatsapp2.py` — print the live DOM so you can find the current selector and update the script.
- **Test with one contact first.** Keep `contacts.xlsx` to a single consenting number for your first successful run, so you're debugging one clean case rather than a whole list.
- **Delivery-tick confirmation is cosmetic.** If the message sends but the script reports "delivery tick not detected," the message still went through — only the tick selector didn't match. It won't stop the run or the reports.
- **Visible browser required.** WhatsApp Web runs in a non-headless window; don't close it mid-run.
- **No official API.** This is unavoidable with WhatsApp automation — everything is driven through the web interface, which is why selector maintenance is part of the deal.

---

# Shared setup — virtual environment

Both assignments run inside a Python virtual environment to stay isolated from the rest of your system.

```bash
python -m venv venv

# Windows (Command Prompt)
venv\Scripts\activate.bat
# Windows (PowerShell)
venv\Scripts\Activate.ps1
# macOS / Linux
source venv/bin/activate
```

When activated you'll see `(venv)` at the start of your prompt. Type `deactivate` when finished.

---

# Files

| File | Assignment | Purpose |
|---|---|---|
| `daily_report_bot.py` | 1 | PyAutoGUI daily-report automation script |
| `playwright_assign.py` | 2 | Playwright WhatsApp bot |
| `contacts.xlsx` | 2 | Input contact list (Name, Phone, Message) |
| `diagnose_whatsapp.py` | 2 | Helper: inspects the WhatsApp Web DOM to find login selectors |
| `diagnose_whatsapp2.py` | 2 | Helper: finds search-box and message-box selectors |
| `requirements.txt` | both | Python dependencies |
| `daily_report_YYYY-MM-DD.xlsx` | 1 | Generated report (run time) |
| `screenshot_daily_report_YYYY-MM-DD.png` | 1 | Screenshot of the final sheet (run time) |
| `whatsapp_report_YYYY-MM-DD.json` | 2 | Full run report (run time) |
| `whatsapp_report_YYYY-MM-DD.xlsx` | 2 | Summary report (run time) |
| `screenshots/` | 2 | Per-contact sent-message screenshots (run time) |
| `README.md` | — | This file |

---

*Build the automation logic responsibly. For Assignment 2, only message people who have agreed to hear from you.*