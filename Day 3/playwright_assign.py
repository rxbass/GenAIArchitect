"""
Assignment 2 - WhatsApp Message Sender + Smart Data Extractor
=============================================================
A Playwright-powered bot that drives WhatsApp Web in a real browser to:
  1. Log in to WhatsApp Web (scan the QR code manually on the first run).
  2. Read contacts from contacts.xlsx (columns: Name, Phone, Message).
  3. For each contact: search by number, send a personalized message
     ({name} is replaced with the real name), wait for it to be sent,
     and take a screenshot.
  4. Extract the last 3 messages from the chat (smart data extraction).
  5. Save a dated report as both JSON and Excel.

Use responsibly: only message people who have agreed to hear from you.
Human-like random delays are built in to reduce the risk of a ban.

Setup:
    pip install playwright openpyxl pandas
    playwright install chromium

Run:
    python playwright_assign.py
"""

import os
import re
import json
import time
import random
from datetime import datetime

import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONTACTS_FILE = "contacts.xlsx"
USER_DATA_DIR = "whatsapp_session"      # persists login so you scan the QR only once
SCREENSHOT_DIR = "screenshots"
TODAY = datetime.now().strftime("%Y-%m-%d")
JSON_REPORT = f"whatsapp_report_{TODAY}.json"
XLSX_REPORT = f"whatsapp_report_{TODAY}.xlsx"

DEFAULT_TEMPLATE = "Hi {name}, this is an automated hello from my Playwright bot."

# Timeouts (milliseconds)
LOGIN_TIMEOUT = 120_000     # 2 minutes to scan the QR on first run
ELEMENT_TIMEOUT = 30_000    # wait up to 30s for chat elements


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def human_delay(min_s=2, max_s=5):
    """Sleep a random human-like interval to avoid looking robotic."""
    delay = random.uniform(min_s, max_s)
    time.sleep(delay)


def personalize(template, name):
    """Replace {name} in the template with the actual contact name."""
    if not template or (isinstance(template, float) and pd.isna(template)):
        template = DEFAULT_TEMPLATE
    return str(template).replace("{name}", str(name))


def load_contacts(path):
    """Read contacts from Excel. Returns a list of dicts."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' not found. Create it with columns: Name, Phone, Message."
        )
    df = pd.read_excel(path)
    # Normalise column names to be forgiving about case/spacing
    df.columns = [str(c).strip().capitalize() for c in df.columns]
    required = {"Name", "Phone"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"contacts.xlsx is missing column(s): {missing}")
    if "Message" not in df.columns:
        df["Message"] = None
    return df.to_dict(orient="records")


def clean_phone(phone):
    """Keep digits and a leading +, strip spaces/dashes."""
    phone = str(phone).strip()
    phone = re.sub(r"[^\d+]", "", phone)
    return phone


# ---------------------------------------------------------------------------
# Core WhatsApp actions
# ---------------------------------------------------------------------------
def wait_for_login(page):
    """Wait until the main chat UI is visible (QR scanned)."""
    print("Waiting for WhatsApp Web to load...")
    print("If prompted, scan the QR code with your phone (you have 2 minutes).")

    # These selectors are confirmed present on a logged-in WhatsApp Web session.
    # We poll each second so slow rendering doesn't cause a false timeout, and
    # so one selector missing doesn't abort the whole wait.
    login_signals = [
        '#pane-side',
        '#side',
        'div[aria-label="Chat list"]',
        '[data-testid="chat-list"]',
    ]

    deadline = time.time() + (LOGIN_TIMEOUT / 1000)
    while time.time() < deadline:
        for sel in login_signals:
            try:
                if page.locator(sel).count() > 0:
                    print("Logged in successfully.\n")
                    return
            except Exception:
                continue
        page.wait_for_timeout(1000)  # re-check in 1 second

    raise PlaywrightTimeoutError("Login not detected within the timeout.")


def open_chat_by_phone(page, phone):
    """
    Open a chat directly via the wa.me deep link, which is the most
    reliable way to reach a number without fiddling with the search box.
    Returns True if the chat opened, False if the number is not on WhatsApp.
    """
    url = f"https://web.whatsapp.com/send?phone={phone.lstrip('+')}"
    page.goto(url)

    # Two outcomes: the message box appears (success) OR an
    # "invalid/not on WhatsApp" dialog appears (failure).
    message_box = (
        'footer div[contenteditable="true"][aria-label="Type a message"], '
        'div[contenteditable="true"][data-tab="10"], '
        'footer div[contenteditable="true"]'
    )
    invalid_dialog = (
        'div:has-text("phone number shared via url is invalid"), '
        'div:has-text("Phone number shared via url is invalid")'
    )

    try:
        page.wait_for_selector(f"{message_box}, {invalid_dialog}", timeout=ELEMENT_TIMEOUT)
    except PlaywrightTimeoutError:
        return False

    # If the invalid dialog is showing, the number isn't reachable.
    if page.locator(
        'div:has-text("phone number shared via url is invalid"), '
        'div:has-text("Phone number shared via url is invalid")'
    ).count() > 0:
        # Dismiss the dialog if there's an OK button
        try:
            page.get_by_role("button", name=re.compile("OK", re.I)).click(timeout=3000)
        except Exception:
            pass
        return False

    return True


def get_message_box(page):
    """Return the 'Type a message' input locator, or None if not found."""
    # Try the placeholder-text approach first (most stable across versions).
    try:
        loc = page.get_by_role("textbox", name=re.compile("Type a message", re.I))
        if loc.count() > 0:
            return loc.first
    except Exception:
        pass

    selectors = [
        'footer div[contenteditable="true"][aria-label="Type a message"]',
        'div[contenteditable="true"][data-tab="10"]',
        'div[contenteditable="true"][aria-label="Type a message"]',
        'footer div[contenteditable="true"]',
    ]
    for sel in selectors:
        loc = page.locator(sel)
        if loc.count() > 0:
            return loc.first
    return None


def send_message(page, text):
    """Type into the message box and send. Returns True on success."""
    box = get_message_box(page)
    if box is None:
        return False

    box.click()
    human_delay(1, 2)
    # Type in a human-ish way rather than pasting all at once
    box.type(text, delay=random.randint(20, 60))
    human_delay(1, 2)
    page.keyboard.press("Enter")
    return True


def confirm_sent(page):
    """
    Confirm the last outgoing message actually left by looking for a
    delivery tick (sent / delivered / read) on the most recent bubble.
    Returns True if a tick is found within the timeout.
    """
    # Outgoing message ticks carry aria-labels like "Sent", "Delivered", "Read".
    tick = (
        '//div[@aria-label=" Sent "] | //div[@aria-label=" Delivered "] | '
        '//div[@aria-label=" Read "] | '
        '//span[@data-icon="msg-check"] | //span[@data-icon="msg-dblcheck"]'
    )
    try:
        page.wait_for_selector(tick, timeout=15_000)
        return True
    except PlaywrightTimeoutError:
        return False


def extract_last_messages(page, count=3):
    """
    Extract the last `count` messages in the currently open chat.
    Returns a list of {direction, text} dicts, oldest-to-newest.
    """
    messages = []
    try:
        page.wait_for_selector('//div[@role="row"]', timeout=ELEMENT_TIMEOUT)
    except PlaywrightTimeoutError:
        return messages

    # Each message bubble contains a span with class 'selectable-text'.
    # Incoming vs outgoing is distinguished by the container class.
    rows = page.locator('//div[contains(@class,"message-in") or contains(@class,"message-out")]')
    total = rows.count()
    if total == 0:
        return messages

    start = max(0, total - count)
    for i in range(start, total):
        row = rows.nth(i)
        # Direction from class name
        cls = row.get_attribute("class") or ""
        direction = "incoming" if "message-in" in cls else "outgoing"
        # Text content
        text_loc = row.locator('//span[contains(@class,"selectable-text")]')
        if text_loc.count() > 0:
            text = text_loc.first.inner_text().strip()
        else:
            text = ""  # could be media/sticker with no text
        if text:
            messages.append({"direction": direction, "text": text})

    return messages


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)

    print("Loading contacts...")
    contacts = load_contacts(CONTACTS_FILE)
    print(f"Loaded {len(contacts)} contact(s).\n")

    report = []

    with sync_playwright() as p:
        # Persistent context keeps you logged in between runs (scan QR once).
        context = p.chromium.launch_persistent_context(
            USER_DATA_DIR,
            headless=False,          # WhatsApp Web needs a visible browser
            args=["--start-maximized"],
            viewport=None,
        )
        page = context.pages[0] if context.pages else context.new_page()
        page.goto("https://web.whatsapp.com")

        try:
            wait_for_login(page)
        except PlaywrightTimeoutError:
            print("Login timed out. Please re-run and scan the QR code faster.")
            context.close()
            return

        for idx, contact in enumerate(contacts, start=1):
            name = str(contact.get("Name", "")).strip()
            phone = clean_phone(contact.get("Phone", ""))
            template = contact.get("Message")
            message = personalize(template, name)

            entry = {
                "name": name,
                "phone": phone,
                "message": message,
                "status": "pending",
                "sent_at": None,
                "screenshot": None,
                "last_messages": [],
                "error": None,
            }

            print(f"[{idx}/{len(contacts)}] {name} ({phone})")

            try:
                opened = open_chat_by_phone(page, phone)
                if not opened:
                    entry["status"] = "contact_not_found"
                    entry["error"] = "Number not on WhatsApp or invalid."
                    print("  -> Contact not found / not on WhatsApp. Skipping.\n")
                    report.append(entry)
                    human_delay()
                    continue

                human_delay(2, 4)

                sent = send_message(page, message)
                if not sent:
                    entry["status"] = "failed"
                    entry["error"] = "Could not find the message box."
                    print("  -> Failed: message box not found.\n")
                    report.append(entry)
                    human_delay()
                    continue

                # Confirm the message actually went out
                if confirm_sent(page):
                    entry["status"] = "sent"
                    entry["sent_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print("  -> Message sent.")
                else:
                    entry["status"] = "sent_unconfirmed"
                    entry["sent_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print("  -> Sent, but delivery tick not detected.")

                human_delay(1, 2)

                # Screenshot of the sent message / chat
                shot_path = os.path.join(
                    SCREENSHOT_DIR, f"{name.replace(' ', '_')}_{TODAY}.png"
                )
                page.screenshot(path=shot_path)
                entry["screenshot"] = shot_path
                print(f"  -> Screenshot saved: {shot_path}")

                # Smart data extraction: last 3 messages
                human_delay(1, 2)
                entry["last_messages"] = extract_last_messages(page, count=3)
                print(f"  -> Extracted {len(entry['last_messages'])} recent message(s).\n")

            except Exception as e:
                entry["status"] = "error"
                entry["error"] = str(e)
                print(f"  -> Unexpected error: {e}\n")

            report.append(entry)
            human_delay()  # 2-5s pause between contacts

        context.close()

    # -----------------------------------------------------------------------
    # Save reports
    # -----------------------------------------------------------------------
    # JSON (full detail)
    with open(JSON_REPORT, "w", encoding="utf-8") as f:
        json.dump(
            {"date": TODAY, "total": len(report), "results": report},
            f, indent=2, ensure_ascii=False,
        )
    print(f"JSON report saved: {JSON_REPORT}")

    # Excel (summary)
    summary_rows = []
    for r in report:
        summary_rows.append({
            "Name": r["name"],
            "Phone": r["phone"],
            "Status": r["status"],
            "Sent At": r["sent_at"] or "",
            "Message": r["message"],
            "Last Messages": " | ".join(
                f"[{m['direction']}] {m['text']}" for m in r["last_messages"]
            ),
            "Screenshot": r["screenshot"] or "",
            "Error": r["error"] or "",
        })
    pd.DataFrame(summary_rows).to_excel(XLSX_REPORT, index=False)
    print(f"Excel report saved: {XLSX_REPORT}")

    # Quick console summary
    sent = sum(1 for r in report if r["status"].startswith("sent"))
    print(f"\nDone. {sent}/{len(report)} message(s) sent.")


if __name__ == "__main__":
    main()