import streamlit as st
import pandas as pd 
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from predict import predict_single

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.title("Customer Churn Risk Analyzer")
st.markdown("""
            This dashboard uses a machine learning model trained on customer subscription data to estimate churn risk and 
            identify customers who may benefit from retention efforts. 
            By entering key customer information, you can get a prediction of whether they are likely to churn and receive actionable insights on how to retain them. 
            """)
st.write("This app predicts whether a customer is likely to churn and shows retention strategies.")

st.header("Enter Customer Information")

gender = st.selectbox("Gender", {"Male", "Female"})
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
monthly = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
total = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=2000.0)
payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
cltv = st.number_input("Customer Lifetime Value", min_value=0.0, max_value=10000.0, value=5000.0)
country = st.selectbox("Country", ["United States"])
state = st.selectbox("State", ["California"])
city = st.text_input("City", "Los Angeles")
zip_code = st.number_input("Zip Code", min_value=0, value=90001)
count = st.number_input("Count", min_value=1, value=1)
multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
longitude = st.number_input("Longitude", value=-118.2437)


input_df = pd.DataFrame([{
    "Count": count,
    "Country": country,
    "State": state,
    "City": city,
    "Zip Code": zip_code,
    "Longitude": longitude,
    "Gender": gender,
    "Senior Citizen": senior,
    "Partner": partner,
    "Dependents": dependents,
    "Tenure Months": tenure,
    "Phone Service": phone_service,
    "Multiple Lines": multiple_lines,
    "Internet Service": internet_service,
    "Online Security": online_security,
    "Online Backup": online_backup,
    "Device Protection": device_protection,
    "Tech Support": tech_support,
    "Streaming TV": streaming_tv,
    "Streaming Movies": streaming_movies,
    "Contract": contract,
    "Paperless Billing": paperless,
    "Payment Method": payment_method,
    "Monthly Charges": monthly,
    "Total Charges": total,
    "CLTV": cltv
}])

if st.button("Predict Churn"):
        pred, prob = predict_single(input_df)
        
        st.subheader("Prediction Result")
        st.write(f"Predicted Churn: **{'Yes' if pred == 1 else 'No'}**")
        st.write(f"Churn Probability: **{prob:.2%}**")
        
        if prob > 0.7: 
                st.error("High risk of churn! Consider offering retention incentives.")
        elif prob > 0.4: 
                st.warning("Moderate risk of churn. Monitor customer closely.")
        else: 
                st.success("Low churn risk.")