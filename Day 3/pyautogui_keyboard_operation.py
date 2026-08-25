import  pyautogui
import time

pyautogui.FAILSAFE = True  # Enable failsafe feature (move mouse to top-left corner to abort)
pyautogui.PAUSE = 1.0  # Add a pause after each

pyautogui.typewrite('Hello, World!', interval=0.1)  # Type 'Hello, World!' with a 0.1 second delay between each character

#hot keys
pyautogui.hotkey('cmd', 'c')  # Press Command + C (copy)


#single key

#pyautogui.press('enter')  # Press the Enter key
pyautogui.press('tab')  # Press the Tab key
pyautogui.press('backspace')  # Press the Backspace key

#hold keys
pyautogui.keyDown('shift')  # Hold down the Shift key
pyautogui.press('a')  # Press the 'a' key while Shift is held
pyautogui.keyUp('shift')  # Release the Shift key
