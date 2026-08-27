import streamlit as st

st.title("My First Streamlit App")
st.header("Welcome to Streamlit")
st.subheader("This is a subheader")
st.write("This is a simple Streamlit app demonstrating basic functionalities.")
st.markdown("You can use **Markdown** to format your text. For example, you can make text **bold**, *italic*, or create lists:\n\n- Item 1\n- Item 2\n- Item 3")    

#getting data from user
name = st.text_input("Enter your name:")    
st.write(f"Hello, {name}! Welcome to Streamlit.")

age = st.slider("Select your age:", 0, 100, 25)
st.write(f"You are {age} years old.")

#button
if st.button("Click me!"):
    st.write("Button clicked!")

#selectbox & Check box
option = st.selectbox("Choose an option:", ["Option 1", "Option 2", "Option 3"])
st.write(f"You selected: {option}") 

checkbox = st.checkbox("Check me!")
if checkbox:
    st.write("Checkbox is checked!")    

#notification messages

st.success("This is a success message!")
st.info("This is an info message.")
st.warning("This is a warning message!")    
st.error("This is an error message!")
st.exception("This is an exception message!")
st.stop()  # Stop the execution of the app here for demonstration purposes
