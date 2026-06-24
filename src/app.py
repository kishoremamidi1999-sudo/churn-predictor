# app.py - Churn Predictor Web Application
import streamlit as st
import pickle
import numpy as np

# Load saved model and scaler
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model  = pickle.load(open(os.path.join(BASE_DIR, "models", "best_model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "models", "scaler.pkl"),     "rb"))

# Page title
st.title("📉 Customer Churn Predictor")
st.write("Fill in the customer details below and click**Predict** to see if they are likely to churn.")
 
# Input Fields

st.header("Customer Information")

col1, col2 = st.columns(2)
with col1:
    gender           = st.selectbox("Gender", 
["Male", "Female"])
    senior_citizen   = st.selectbox("Senior Citizen", 
["No", "Yes"])
    partner          = st.selectbox("Has Partner", 
["No", "Yes"])
    dependents       = st.selectbox("Has Dependents", 
["No", "Yes"])
    tenure           = st.slider("Tenure (months)", 
0, 72, 120)
    phone_service    = st.selectbox("Phone Service",
["No", "Yes"])
    multiple_lines   = st.selectbox("Multiple Lines",
["No", "Yes", "No phone service"])
    
with col2:
    internet_service    = st.selectbox("Internet" \
"Service",  ["DSL", "Fiber optic", "No"])
    online_security     = st.selectbox("Online" \
 "Security",  ["No", "Yes", "No internet service"])
    online_backup       = st.selectbox("Online Backup",
["No", "Yes", "No internet service"])
    device_protection   = st.selectbox("Device Protection",
["No", "Yes", "No internet service"])
    tech_support        = st.selectbox("Tech Support",
["No", "Yes", "No internet service"])
    streaming_tv        = st.selectbox("Streaming TV",
["No", "Yes", "No internet service"])
    streaming_movies    = st.selectbox("Streaming Movies",
["No", "Yes", "No internet service"])
    
    st.header("Billing Information")

    col3, col4 = st.columns(2)
with col3:
    contract            = st.selectbox("Contract Type",
["Month-to-month", "One year", "Two year"])
    paperless_billing   = st.selectbox("Paperless Billing",
["No", "Yes"])
    payment_method      = st.selectbox("Payment Method",
[        
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
with col4:
    monthly_charges  = st.number_input("Monthly Charges ($)",
0.0, 200.0, 65.0)
    total_charges    = st.number_input("Total Charges ($)",
0.0, 10000.0, 500.0)

# Encode inputs (same order as training data)
def encode(val, mapping):
    return mapping[val]

gender_map    = {"Male": 1, "Female": 0}
yesno_map     = {"No": 0, "Yes": 1}
multiline_map = {"No": 0, "Yes": 1, "No phone service": 2}
internet_map  = {"DSL": 0, "Fiber optic": 1, "No": 2}
inet_feat_map = {"No": 0, "Yes": 1, "No internet service": 2}
contract_map  = {"Month-to-month": 0, "One year": 1, "Two year": 2}
payment_map   ={
    "Bank transfer (automatic)": 0,
    "Credit card (automatic)":   1,
    "Electronic check":          2,
    "Mailed check":              3
}
input_data = np.array([[
    encode(gender,            gender_map),
    encode(senior_citizen,    yesno_map),
    encode(partner,           yesno_map),
    encode(dependents,        yesno_map),
    tenure,
    encode(phone_service,     yesno_map),
    encode(multiple_lines,    multiline_map),
    encode(internet_service,  internet_map),
    encode(online_security,   inet_feat_map),
    encode(online_backup,     inet_feat_map),
    encode(device_protection, inet_feat_map),
    encode(tech_support,      inet_feat_map),
    encode(streaming_tv,      inet_feat_map),
    encode(streaming_movies,  inet_feat_map),
    encode(contract,          contract_map),
    encode(paperless_billing, yesno_map),
    encode(payment_method,    payment_map),
    monthly_charges,
    total_charges
    ]])

# Scale and Predict

if st.button("🔍 Predict"):
    input_scaled = scaler.transform(input_data)
    prediction   = model.predict(input_scaled)[0]
    probability  = model.predict_proba(input_scaled)[0][1]

    st.divider()

    if prediction == 1:
        st.error(f"⚠️ This customer is **likely to churn** - {probability:.1%} risk")
    else:
        st.success(f"✅ This customer is **not likely to stay** - only {probability:.1%} churn risk")
        
        st.caption(f"Model confidence: {max(probability, 1 - probability):.1%}")
        
