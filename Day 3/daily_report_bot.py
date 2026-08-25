# Steps
# 1. Open Chrome
# 2. Navigate to https://groww.in/share-market-today
# 3. Use mouse operation to click the search bar and search for a stock - HDFC Bank
# 4. Use mouse operations to copy the stock information
# 5. Open Excel and create a new row: today's date & time, the fetched data, a short comment
# 6. Save the Excel file with today's date in the filename, e.g. daily_report_2025-06-17.xlsx
# 7. Take a screenshot of the final Excel sheet and save it

import pyautogui
import time
import os
from datetime import datetime

pyautogui.FAILSAFE = True   # move mouse to top-left corner to abort
pyautogui.PAUSE = 1.0       # forced pause after each pyautogui call


def reset_modifiers():
    """Release any modifier keys that might be stuck down."""
    for key in ('alt', 'ctrl', 'shift'):
        pyautogui.keyUp(key)


print("Starting the automation script...")
time.sleep(1)

# -------------------------------------------------------------------
# Step 1: Open Chrome
# -------------------------------------------------------------------
print("Step 1: Opening Chrome...")
pyautogui.hotkey('win', 's')          # open Windows search
time.sleep(1)
pyautogui.typewrite('chrome', interval=0.1)
time.sleep(1)
pyautogui.press('enter')
time.sleep(1)                         # give Chrome time to fully open

# -------------------------------------------------------------------
# Step 2: Navigate to the Groww website
# -------------------------------------------------------------------
print("Step 2: Navigating to the Groww website...")
pyautogui.hotkey('ctrl', 'l')         # focus the address bar
time.sleep(1)
pyautogui.typewrite('https://groww.in/share-market-today', interval=0.05)
time.sleep(1)
pyautogui.press('enter')
time.sleep(2)                         # let the page load

# -------------------------------------------------------------------
# Step 3: Click the search bar and search for HDFC Bank
# -------------------------------------------------------------------
print("Step 3: Searching for HDFC Bank...")
pyautogui.moveTo(1200, 180, duration=1)   # <-- adjust to your search bar position
pyautogui.click()
time.sleep(1)
pyautogui.typewrite('HDFC Bank', interval=0.1)
time.sleep(2)                          # let suggestions appear
pyautogui.press('enter')
time.sleep(2)                          # let the stock page load

# -------------------------------------------------------------------
# Step 4: Copy the stock information (drag-select the price, then copy)
# -------------------------------------------------------------------
print("Step 4: Copying the stock information...")
pyautogui.moveTo(360, 500, duration=1)     # <-- adjust to the price location
pyautogui.dragTo(450, 500, duration=1)     # drag-select the value
time.sleep(0.5)
pyautogui.hotkey('ctrl', 'c')
time.sleep(0.5)

# -------------------------------------------------------------------
# Step 5: Open Excel and build the new row
# -------------------------------------------------------------------
print("Step 5: Opening Excel and adding a row...")
pyautogui.hotkey('win', 's')
time.sleep(1)
pyautogui.typewrite('excel', interval=0.1)
time.sleep(1)
pyautogui.press('enter')
time.sleep(6)                          # give Excel time to open a blank workbook

# Column A: date & time
pyautogui.moveTo(700, 300, duration=1)     # <-- adjust to cell A1
pyautogui.click()
time.sleep(0.5)
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
pyautogui.typewrite(now, interval=0.05)
pyautogui.press('tab')                 # move to column B

# Column B: paste the fetched stock data
pyautogui.hotkey('ctrl', 'v')
time.sleep(0.5)
pyautogui.press('tab')                 # move to column C

# Column C: your own short comment
pyautogui.typewrite('HDFC Bank daily price', interval=0.03)
pyautogui.press('enter')
time.sleep(0.5)

pyautogui.hotkey('ctrl', 'a')   # select all cells
time.sleep(0.3)
pyautogui.hotkey('alt', 'h')
time.sleep(0.3)
pyautogui.press('o')
time.sleep(0.3)
pyautogui.press('i')
time.sleep(0.5)
pyautogui.click()
# -------------------------------------------------------------------
# Step 6: Save the file with today's date in the filename
# -------------------------------------------------------------------
print("Step 6: Saving the Excel file...")
reset_modifiers()                      # make sure no key is stuck before F12
time.sleep(1)

fileName = datetime.now().strftime("%Y-%m-%d")
save_dir = r'D:\ai\GenAIArchitect\Day 3'
os.makedirs(save_dir, exist_ok=True)   # create the folder if it doesn't exist

pyautogui.press('f12')                 # open the classic Save As dialog directly
time.sleep(3)                          # give the dialog time to appear
full_path = os.path.join(save_dir, f'daily_report_{fileName}.xlsx')
pyautogui.typewrite(full_path, interval=0.05)
time.sleep(1)
pyautogui.press('enter')
time.sleep(2)
pyautogui.press('enter')               # confirm overwrite if the file already exists
time.sleep(2)

# -------------------------------------------------------------------
# Step 7: Take a screenshot of the final Excel sheet
# -------------------------------------------------------------------
print("Step 7: Taking a screenshot...")
screenshot = pyautogui.screenshot()
screenshot_path = os.path.join(save_dir, f'screenshot_daily_report_{fileName}.png')
screenshot.save(screenshot_path)
print(f"Screenshot saved to: {screenshot_path}")

print("Done!")