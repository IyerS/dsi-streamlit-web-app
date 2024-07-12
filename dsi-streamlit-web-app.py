# Import Libraries

import streamlit as st
import pandas as pd
import joblib

# load our model pipeline object
model = joblib.load("model.joblib")

# add title and instruction
st.title("Purchase prediction model")
st.subheader("Enter customer information and submit for likelyhood to purchase")

# age input form
age = st.number_input(
    label = "01. Enter the customer's age",
    min_value = 18,
    max_value = 100,
    value = 35)

# gender input form
gender = st.radio(
    label = "02. Enter the customer's gender",
    options = ["M", "F"])

# Credit Score input form
credit_score = st.number_input(
    label = "03. Enter the customer's credit score",
    min_value = 0,
    max_value = 1000,
    value = 500)

# Submit input to model
if st.button("Submit for prediction"):
    
    # store the data in a df for prediction
    new_df = pd.DataFrame({"age": [age], "gender": [gender], "credit_score": [credit_score]})
    
    # apply model pipeline to the input data and extract probability prediction
    pred_proba = model.predict_proba(new_df)[0][1]
    
    # Output the Prediction
    st.subheader(f"Based on the customer attributes, the model predicts the customers probability of purchase is {pred_proba:.0%}")



