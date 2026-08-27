import streamlit as st

st.set_page_config(page_title="Calculator", layout="centered")
st.title("🧮 Calculator")

# Initialize state
if "display" not in st.session_state:
    st.session_state.display = "0"
if "reset_next" not in st.session_state:
    st.session_state.reset_next = False

def press(btn):
    val = st.session_state.display

    if btn == "C":
        st.session_state.display = "0"
        st.session_state.reset_next = False

    elif btn == "DEL":
        st.session_state.display = val[:-1] if len(val) > 1 else "0"

    elif btn == "=":
        try:
            result = eval(val)
            # Show int if result is whole number
            if isinstance(result, float) and result.is_integer():
                st.session_state.display = str(int(result))
            else:
                st.session_state.display = str(result)
        except:
            st.session_state.display = "Error"
        st.session_state.reset_next = True

    else:
        if st.session_state.reset_next:
            st.session_state.display = btn
            st.session_state.reset_next = False
        elif val == "0" and btn not in ("+", "-", "*", "/", "."):
            st.session_state.display = btn
        else:
            st.session_state.display += btn

# Display screen
st.markdown(f"""
<div style="background:#1e1e1e; color:#00ff88; font-size:2rem; font-weight:bold;
            padding:16px 20px; border-radius:10px; text-align:right;
            margin-bottom:12px; letter-spacing:1px; min-height:65px;
            border: 1px solid #333;">
    {st.session_state.display}
</div>
""", unsafe_allow_html=True)

# Button grid — unique keys using index to avoid conflicts
rows = [
    ["7", "8", "9", "/"],
    ["4", "5", "6", "*"],
    ["1", "2", "3", "-"],
    ["0", ".", "=", "+"],
    ["C", "(", ")", "DEL"],
]

key_map = {
    "/": "divide", "*": "multiply", "-": "minus",
    "+": "plus", "=": "equals", ".": "dot",
    "(": "lparen", ")": "rparen", "C": "clear", "DEL": "delete"
}

for r, row in enumerate(rows):
    cols = st.columns(4)
    for c, btn in enumerate(row):
        safe_key = f"k_{r}_{c}_{key_map.get(btn, btn)}"
        label = "⌫" if btn == "DEL" else btn
        cols[c].button(label, key=safe_key,
                       use_container_width=True,
                       on_click=press, args=(btn,))