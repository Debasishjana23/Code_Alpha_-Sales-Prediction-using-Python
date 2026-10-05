import streamlit as st
import numpy as np
import joblib


# Page Configuration

st.set_page_config(
    page_title="Sales Prediction App",
    page_icon="📈",
    layout="centered"
)


# Load Trained Model

@st.cache_resource
def load_model():
    model = joblib.load("lr_model.pkl")
    return model

model = load_model()


# App Title & Description

st.title("📈 Sales Prediction Using Machine Learning")
st.write(
    "Predict product sales based on advertising expenditure across "
    "TV, Radio, and Newspaper platforms."
)

st.markdown("---")


# User Inputs

st.subheader("Enter Advertising Spend ")

tv = st.number_input(
    "TV Advertising Spend",
    min_value=0.0,
    max_value=500.0,
    value=100.0,
    step=1.0
)

radio = st.number_input(
    "Radio Advertising Spend",
    min_value=0.0,
    max_value=100.0,
    value=20.0,
    step=1.0
)

newspaper = st.number_input(
    "Newspaper Advertising Spend",
    min_value=0.0,
    max_value=200.0,
    value=30.0,
    step=1.0
)


# Prediction

if st.button("Predict Sales"):
    input_data = np.array([[tv, radio, newspaper]])
    prediction = model.predict(input_data)[0]

    st.markdown("---")
    st.subheader("📊 Prediction Result")
    st.success(f"Predicted Sales: **{prediction:.2f} units**")

    st.caption(
        "Prediction is based on a trained Linear Regression model using historical advertising data."
    )


# Footer

st.markdown("---")
st.caption("Built with cares using Python & Streamlit")