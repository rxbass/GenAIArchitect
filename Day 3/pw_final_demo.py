from playwright.sync_api import sync_playwright
from datetime import datetime

print("Starting Playwright automation script...")
print("Starting the script at:", datetime.now())

#Daily weather report bot
#Chromium -> weather site -> extract weather report -> screen shot -> close the browser    
with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)  # Set headless=False to see the browser
    page = browser.new_page()

    #Navigation
    page.goto("https://www.bbc.com/weather/1264527")  # Example: navigate to the BBC weather page for a specific location
    page.wait_for_load_state("networkidle")  # Wait for the page to load completely
    page.screenshot(path="playwright_weather_report_screenshot.png")  # Example: take a screenshot

    browser.close()