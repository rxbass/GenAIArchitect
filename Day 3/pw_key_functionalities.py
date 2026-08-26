from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)  # Set headless=False to see the browser
    page = browser.new_page()

    #Navigation
    page.goto("https://groww.in/share-market-today")
    page.screenshot(path="playwright_grow_screenshot.png")  # Example: take a screenshot

    #clicking
    page.click("text=Infosys")  # Example: click on the Infosys link
    page.screenshot(path="playwright_infosys_screenshot.png")  # Example: take a screenshot
    page.click("div[class='search-placeholder contentSecondary']")  # Example: click on the first element with the specified class

    #typing
    page.type("input", "HDFC Bank")  # Example: type "HDFC Bank" into the search box
    page.press("input", "Enter")  # Example: press Enter after typing
    page.screenshot(path="playwright_hdfc_screenshot.png")  # Example: take a screenshot

    #waiting for elements
    page.wait_for_selector("text=HDFC Bank")  # Example: wait for the HDFC Bank link to appear
    page.screenshot(path="playwright_hdfc_wait_screenshot.png")  # Example: take a screenshot

    #Extracting the data
    title = page.title()  # Example: get the page title
    print(f"Page title: {title}")  # Print the page title
    browser.close()