# Steps
# 1. Open chrome
# 2. Navigate to a https://groww.in/share-market-today  website
# 3. Use mouse operation to click on the search bar and search for a specific stock - HDFC bank limited
# 4. Use mouse operations to copy the stock information 
# 5. Open excel sheet and Create a new row containing three things: today's date & time, the fetched data, and your own short comment 
# 6. Save the excel file with today's date in the filename, e.g. daily_report_2025-06-17.xlsx
# 7. Take a screenshot of the final Excel sheet and save it


import pyautogui
import time
from datetime import datetime
import pyscreeze
pyautogui.FAILSAFE = True  # Enable failsafe feature (move mouse to top-left corner to abort)
pyautogui.PAUSE = 1.0  # Add a pause after each

print("Starting the automation script...")
time.sleep(1)  # Wait for 1 second before starting

print("Step 1: Opening Chrome...")
time.sleep(1)  # Wait for 1 second

pyautogui.hotkey('win', 's')  # Open Windows search
time.sleep(1)  # Wait for 1 second  

pyautogui.typewrite('chrome', interval=0.1)  # Type 'chrome' with a 0.1 second delay between each character
time.sleep(1)  # Wait for 1 second

pyautogui.press('enter')  # Press the Enter key to open Chrome

print("Step 2: Navigating to the groww website...")
time.sleep(1)  # Wait for 3 seconds to allow Chrome to open
pyautogui.typewrite('https://groww.in/share-market-today', interval=0.1)
time.sleep(1)  # Wait for 1 second
pyautogui.press('enter')  # Press the Enter key to navigate to the website
time.sleep(3)  # Wait for 3 seconds to load the website

print("Step 3: Use mouse operation to click on the search bar and search for a specific stock - HDFC bank limited")
pyautogui.moveTo(1200, 180, duration=1)  # Move the mouse to the search bar
pyautogui.click()  # Click the mouse to focus on the search bar
pyautogui.typewrite('HDFC Bank', interval=0.1)  # Type the stock name
pyautogui.press('enter')  # Press the Enter key to search for the stock

print("Step 4: Use mouse operations to copy the stock information")
pyautogui.moveTo(360, 500, duration=1)                 # move to start point
pyautogui.dragTo(450, 500, duration=1)        # click, hold, drag to (360, 500)
pyautogui.hotkey('ctrl', 'c')  # Copy the selected text

print("Step 5: Open excel sheet and Create a new row containing three things: today's date & time, the fetched data, and your own short comment")
pyautogui.hotkey('win', 's')  # Open Windows search
time.sleep(1)  # Wait for 1 second
pyautogui.typewrite('excel', interval=0.1)  # Type 'excel' with a 0.1 second delay between each character
time.sleep(1)  # Wait for 1 second
pyautogui.press('enter')  # Press the Enter key to open Excel
time.sleep(3)  # Wait for 3 seconds to allow Excel to open
#Create a new row containing three things: today's date & time, the fetched data, and your own short comment 
pyautogui.moveTo(700, 300, duration=1)  # Move the mouse to the first cell of the new row
pyautogui.click()  # Click the mouse to focus on the cell
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
pyautogui.typewrite(now, interval=0.3)  # Type the current date and time
pyautogui.press('tab')  # Move to the next cell
pyautogui.hotkey('ctrl', 'v')  # Paste the copied stock information 
# Open font size box: Alt, H, F, S  then type the size and Enter
pyautogui.hotkey('alt', 'h')
time.sleep(0.3)
pyautogui.press('f')
pyautogui.press('s')
time.sleep(0.3)
pyautogui.typewrite('12', interval=0.1)   # font size 14
pyautogui.press('enter')
pyautogui.keyDown('alt')
time.sleep(0.3)
pyautogui.press('tab')
pyautogui.press('enter')  # Press the Enter key to search for the stock

pyautogui.moveTo(360, 450, duration=1)                 # move to start point
pyautogui.dragTo(360, 500, duration=1)        # click, hold, drag to (360, 500)
pyautogui.keyUp('alt')
pyautogui.hotkey('ctrl', 'c')  # Copy the selected text
time.sleep(0.5)

pyautogui.keyDown('alt')
pyautogui.press('tab') 
pyautogui.keyUp('alt')
time.sleep(0.5)
pyautogui.press('tab')   # press again to move further along
pyautogui.hotkey('ctrl', 'v')  # Paste the copied stock information 
time.sleep(1)
# Open font size box: Alt, H, F, S  then type the size and Enter
pyautogui.hotkey('alt', 'h')
time.sleep(0.3)
pyautogui.press('f')
pyautogui.press('s')
time.sleep(0.3)
pyautogui.typewrite('12', interval=0.1)   # font size 14
pyautogui.press('enter')
pyautogui.keyDown('alt')
pyautogui.keyUp('alt')
time.sleep(0.3)
print("Step 6: Save the excel file with today's date in the filename, e.g. daily_report_2025-06-17.xlsx")
time.sleep(1)  # Wait for 1 second
fileName = datetime.now().strftime("%Y-%m-%d")
pyautogui.press('f12')
print("Step 7: Entering the file name...")
time.sleep(2)
pyautogui.typewrite(rf'D:\ai\GenAIArchitect\Day 3\daily_report_{fileName}.xlsx', interval=0.05)
time.sleep(1)  # Wait for 1 second
pyautogui.press('enter')  # Press the Enter key to save the file

print("Step 7: Take a screenshot of the final Excel sheet and save it")
screenshot = pyautogui.screenshot()  # Take a screenshot of the entire screen
#convert the current date and time to a string format suitable for a filename
screenshot.save(f'screenshot_daily_report_{fileName}.png')  # Save the screenshot to a file
