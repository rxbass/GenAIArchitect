"""
Diagnostic: find the real login selector for YOUR WhatsApp Web version.
Run this, wait for WhatsApp Web to appear, then read the output.
It will NOT time out on you — it waits 40s then reports what it sees.
"""

import time
from playwright.sync_api import sync_playwright

USER_DATA_DIR = "whatsapp_session"

with sync_playwright() as p:
    context = p.chromium.launch_persistent_context(
        USER_DATA_DIR,
        headless=False,
        args=["--start-maximized"],
        viewport=None,
    )
    page = context.pages[0] if context.pages else context.new_page()
    page.goto("https://web.whatsapp.com")

    print("Waiting 40 seconds for the page to settle (scan QR if needed)...")
    time.sleep(40)

    print("\n--- Checking candidate login selectors ---")
    candidates = [
        "#pane-side",
        'div[aria-label="Chat list"]',
        'div[title="Search input textbox"]',
        'div[contenteditable="true"][data-tab="3"]',
        'header',
        '[data-testid="chat-list"]',
        '[data-testid="chatlist-header"]',
        'div[role="textbox"]',
        '#side',
    ]
    for sel in candidates:
        try:
            count = page.locator(sel).count()
            print(f"  {'FOUND ' if count else '  --  '} ({count})  {sel}")
        except Exception as e:
            print(f"  ERROR   {sel}: {e}")

    print("\n--- Page title ---")
    print(" ", page.title())

    print("\n--- All elements with an id (first 25) ---")
    ids = page.evaluate("""() => Array.from(document.querySelectorAll('[id]')).map(e => e.id).filter(Boolean).slice(0,25)""")
    for i in ids:
        print("  #" + i)

    print("\n--- All aria-labels on divs (first 25) ---")
    labels = page.evaluate("""() => Array.from(document.querySelectorAll('div[aria-label]')).map(e => e.getAttribute('aria-label')).filter(Boolean).slice(0,25)""")
    for l in labels:
        print("  aria-label:", l)

    print("\nDone. Leave the browser open or close it — copy the output above and share it.")
    time.sleep(5)
    context.close()