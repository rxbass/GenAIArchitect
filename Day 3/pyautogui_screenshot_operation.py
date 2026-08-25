import pyautogui
import pyscreeze

pyautogui.FAILSAFE = True  # Enable failsafe feature (move mouse to top-left corner to abort)
pyautogui.PAUSE = 1.0  # Add a pause after each



screenshot = pyautogui.screenshot()  # Take a screenshot of the entire screen
screenshot.save('screenshot.png')  # Save the screenshot to a file