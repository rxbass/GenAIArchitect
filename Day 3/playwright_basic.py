from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)  # Set headless=False to see the browser
    page = browser.new_page()
    page.goto("https://groww.in/share-market-today")
    # Add your automation steps here
    page.screenshot(path="playwright_screenshot.png")  # Example: take a screenshot
    browser.close()