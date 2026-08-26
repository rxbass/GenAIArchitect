"""
Diagnostic #2: find the SEARCH BOX and (after opening a chat) the
MESSAGE BOX selectors for your WhatsApp Web version.

It opens WhatsApp, waits, prints search-box candidates, then opens the
first chat in your list and prints message-box candidates.
"""

import time
from playwright.sync_api import sync_playwright

USER_DATA_DIR = "whatsapp_session"

with sync_playwright() as p:
    context = p.chromium.launch_persistent_context(
        USER_DATA_DIR, headless=False, args=["--start-maximized"], viewport=None,
    )
    page = context.pages[0] if context.pages else context.new_page()
    page.goto("https://web.whatsapp.com")
    print("Waiting 30s for load...")
    time.sleep(30)

    print("\n--- SEARCH BOX candidates ---")
    search_candidates = [
        'div[contenteditable="true"][data-tab="3"]',
        'div[aria-label="Search input textbox"]',
        'div[title="Search input textbox"]',
        'div[contenteditable="true"][role="textbox"]',
        '[data-testid="chat-list-search"]',
        'div[aria-label="Search"]',
    ]
    for sel in search_candidates:
        try:
            print(f"  {'FOUND' if page.locator(sel).count() else '  -- '} ({page.locator(sel).count()})  {sel}")
        except Exception as e:
            print(f"  ERR  {sel}: {e}")

    # Print contenteditable elements and their attributes
    print("\n--- All contenteditable=true elements & their key attributes ---")
    info = page.evaluate("""() => {
        return Array.from(document.querySelectorAll('[contenteditable="true"]')).map(e => ({
            ariaLabel: e.getAttribute('aria-label'),
            dataTab: e.getAttribute('data-tab'),
            title: e.getAttribute('title'),
            role: e.getAttribute('role'),
        }));
    }""")
    for i in info:
        print("  ", i)

    print("\n--- Opening first chat to inspect the message box ---")
    try:
        page.locator('#pane-side div[role="listitem"]').first.click()
        time.sleep(4)
    except Exception as e:
        print("  Could not click first chat:", e)

    print("\n--- MESSAGE BOX candidates ---")
    msg_candidates = [
        'footer div[contenteditable="true"][aria-label="Type a message"]',
        'div[contenteditable="true"][data-tab="10"]',
        'div[contenteditable="true"][aria-label="Type a message"]',
        'footer div[contenteditable="true"]',
    ]
    for sel in msg_candidates:
        try:
            print(f"  {'FOUND' if page.locator(sel).count() else '  -- '} ({page.locator(sel).count()})  {sel}")
        except Exception as e:
            print(f"  ERR  {sel}: {e}")

    print("\n--- contenteditable elements AFTER opening chat ---")
    info2 = page.evaluate("""() => {
        return Array.from(document.querySelectorAll('[contenteditable="true"]')).map(e => ({
            ariaLabel: e.getAttribute('aria-label'),
            dataTab: e.getAttribute('data-tab'),
        }));
    }""")
    for i in info2:
        print("  ", i)

    print("\nDone — copy everything above.")
    time.sleep(3)
    context.close()