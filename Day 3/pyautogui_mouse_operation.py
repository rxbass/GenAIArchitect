import pyautogui
import time
pyautogui.FAILSAFE = True  # Enable failsafe feature (move mouse to top-left corner to abort)
pyautogui.PAUSE = 1.0  # Add a pause after each

pyautogui.moveTo(100, 100, duration=1)  # Move the mouse to (100, 100) over 1 second

pyautogui.click(100, 100)  # Click the mouse at (100, 100)
pyautogui.doubleClick(100, 100)  # Double-click the mouse at (100, 100)
pyautogui.rightClick(100, 100)  # Right-click the mouse at (100, 100)   
pyautogui.leftClick(100, 100)  # Left-click the mouse at (100, 100)
#move to scroll area and scroll down
pyautogui.moveTo(1540, 200, duration=1)  # Move the mouse to (1500, 500) over 1 second
pyautogui.scroll(-500)  # Scroll down 500 units
pyautogui.scroll(500)  # Scroll up 500 units