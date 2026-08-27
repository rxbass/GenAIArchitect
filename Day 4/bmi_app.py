import streamlit as st

st.title("BMI Calculator")

weight = st.number_input("Weight (kg)", min_value=1.0, value=70.0)
height = st.number_input("Height (cm)", min_value=1.0, value=170.0)

if st.button("Calculate"):
    height_m = height / 100
    bmi = weight / (height_m ** 2)

    st.write(f"Your BMI is: **{round(bmi)}**")

    if bmi < 18.5:
        st.warning("Underweight")
    elif bmi < 25:
        st.success("Normal weight")
    elif bmi < 30:
        st.warning("Overweight")
    else:
        st.error("Obese")