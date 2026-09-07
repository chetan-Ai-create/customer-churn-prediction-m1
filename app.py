import streamlit as st
import pandas as pd
import joblib

# Load trained model and preprocessor
model = joblib.load("churn_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# Page title
st.title("📊 Customer Churn Prediction")
st.write("Predict whether a customer is likely to churn.")

st.subheader("Customer Information")

# Create two columns
col1, col2 = st.columns(2)

# ---------------- COLUMN 1 ----------------
with col1:
    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )


# ---------------- COLUMN 2 ----------------
with col2:
    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    senior_citizen = st.number_input(
        "Senior Citizen",
        min_value=0,
        max_value=1,
        value=0
    )


# ---------------- NUMERICAL INPUTS ----------------
tenure = st.number_input(
    "Tenure (months)",
    min_value=0,
    max_value=100,
    value=12
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=840.0
)


# ---------------- PREDICTION ----------------
if st.button("🔮 Predict Churn"):

    customer_data = {
        "gender": gender,
        "Partner": partner,
        "Dependents": dependents,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "SeniorCitizen": senior_citizen,
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    # Convert input into DataFrame
    customer_df = pd.DataFrame([customer_data])

    # Apply the same preprocessing used during training
    processed = preprocessor.transform(customer_df)

    # Make prediction
    prediction = model.predict(processed)[0]

    # Get churn probability
    probability = model.predict_proba(processed)[0][1]

    # Display result
    if prediction == 1:
        st.error("⚠️ Customer is likely to CHURN")
    else:
        st.success("✅ Customer is likely to STAY")

    # Display probability
    st.metric(
        "Churn Probability",
        f"{probability:.2%}"
    )

    # Risk level
    if probability >= 0.70:
        st.error("🔴 High Churn Risk")
    elif probability >= 0.40:
        st.warning("🟡 Medium Churn Risk")
    else:
        st.success("🟢 Low Churn Risk")