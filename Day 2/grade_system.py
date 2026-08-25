# grade_system.py

def get_grade(mark):
    # 90-100 -> "A", 80-89 -> "B", 70-79 -> "C", 60-69 -> "D", below 60 -> "E"
    # remember: boundaries are inclusive (exactly 90 is an A)
    if mark >= 90:
        return "A"
    elif mark >= 80:
        return "B"
    elif mark >= 70:
        return "C"
    elif mark >= 60:
        return "D"
    else:
        return "E"

# --- main program ---
user_input = input("Enter your mark (0-100): ")

try:
    mark = int(user_input)          # might throw ValueError
    if mark < 0 or mark > 100:
        print("Mark is out of range. Please enter a number between 0 and 100.")
    else:
        grade = get_grade(mark)
        print(f"Mark: {mark} -> Grade: {grade}")
except ValueError:
    print("That is not a valid number. Please enter a number between 0 and 100.")