import streamlit as st

st.title("🎓 Student Grade System")

def get_grade(mark):
    if mark >= 90:
        return "A", "Excellent! Outstanding performance.", "🏆", "green"
    elif mark >= 80:
        return "B", "Great job! Above average performance.", "🥈", "blue"
    elif mark >= 70:
        return "C", "Good work! Average performance.", "👍", "orange"
    elif mark >= 60:
        return "D", "You passed, but there's room to improve.", "📚", "orange"
    else:
        return "E", "Don't give up! Keep studying and try again.", "💪", "red"

st.write("Enter your mark below to find out your grade.")

mark = st.slider("Select your mark", min_value=0, max_value=100, value=50)

st.write(f"**Your mark:** {mark} / 100")
st.progress(mark)

grade, message, emoji, color = get_grade(mark)

st.markdown(f"## {emoji} Grade: **{grade}**")

if color == "green":
    st.success(message)
elif color == "blue":
    st.info(message)
elif color == "red":
    st.error(message)
else:
    st.warning(message)

# Grade reference table
st.markdown("---")
st.markdown("### 📋 Grading Scale")

col1, col2 = st.columns(2)
with col1:
    st.markdown("| Mark | Grade |")
    st.markdown("|------|-------|")
    st.markdown("| 90 – 100 | A |")
    st.markdown("| 80 – 89  | B |")
    st.markdown("| 70 – 79  | C |")
    st.markdown("| 60 – 69  | D |")
    st.markdown("| Below 60 | E |")