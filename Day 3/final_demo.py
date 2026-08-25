# Steps
# 1. Open chrome
# 2. Navigate to a weather website
# 3. Copy the weather information
# 4. Open a notepad and paste the weather information into a text file
# 5. Save the text file with a specific name

import pyautogui
import time

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

print("Step 2: Navigating to the weather website...")
time.sleep(1)  # Wait for 3 seconds to allow Chrome to open
pyautogui.typewrite('https://weather.com/in/tamil-nadu/chennai/locality/nangainallur/today', interval=0.1)
time.sleep(1)  # Wait for 1 second
pyautogui.press('enter')  # Press the Enter key to navigate to the website
time.sleep(10)  # Wait for 10 seconds to load the website and copy the weather information
pyautogui.hotkey('ctrl', 'a')  # Select all text in website
print("Step 3: Copying weather information...")
time.sleep(1)  # Wait for 1 second
pyautogui.hotkey('ctrl', 'c')  # Copy the selected text
print("Step 4: Opening Notepad...")
time.sleep(1)  # Wait for 1 second
pyautogui.hotkey('win', 's')  # Open Windows search
time.sleep(1)  # Wait for 1 second
pyautogui.typewrite('notepad', interval=0.1)  # Type 'notepad' with a 0.1 second delay between each character
time.sleep(1)  # Wait for 1 second
pyautogui.press('enter')
time.sleep(1)  # Wait for 1 second
#click new tab to open new notepad
pyautogui.hotkey('ctrl', 'n')  # Open a new Notepad window
print("Step 5: Pasting weather information into Notepad...")
time.sleep(1)  # Wait for 1 second
pyautogui.hotkey('ctrl', 'v')  # Paste the copied text into Notepad
print("Step 6: Saving the text file...")
time.sleep(1)  # Wait for 1 second
pyautogui.hotkey('ctrl', 's')  # Open the Save dialog
print("Step 7: Entering the file name...")
time.sleep(1)  # Wait for 1 second
pyautogui.typewrite(r'D:\ai\GenAIArchitect\Day 3\weather_info.txt', interval=0.1)  # Type the file name to save in the current directory
time.sleep(1)  # Wait for 1 second
pyautogui.press('enter')  # Press the Enter key to save the file
print("Step 8: File saved successfully!")
