# ================== IMPORTS ==================
import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ================== CONFIG ==================
st.set_page_config(page_title="Churn AI", layout="wide")

# ================== LOAD ==================
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ================== HEADER ==================
st.markdown("""
<h1 style='text-align: center;'>📊 Customer Churn Intelligence</h1>
<p style='text-align: center; color: gray;'>
AI-powered churn prediction with explainability
</p>
""", unsafe_allow_html=True)

# ================== SIDEBAR ==================
st.sidebar.header("Customer Details")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Partner", ["No", "Yes"])
dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly = st.sidebar.slider("Monthly Charges", 0, 150, 70)
total = st.sidebar.slider("Total Charges", 0, 10000, 2000)

phone = st.sidebar.selectbox("Phone Service", ["No", "Yes"])
multi = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

security = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
backup = st.sidebar.selectbox("Online Backup", ["No", "Yes", "No internet service"])
device = st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"])
support = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])

tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
movies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])

payment = st.sidebar.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
)

# ================== ENCODING ==================
def encode(val):
    return 1 if val == "Yes" else 0

def encode_gender(val):
    return 1 if val == "Male" else 0

# ================== PREDICT ==================
if st.button("🚀 Predict Churn"):

    with st.spinner("Analyzing customer..."):

        input_data = np.array([[
            encode_gender(gender),
            encode(senior),
            encode(partner),
            encode(dependents),
            tenure,
            encode(phone),
            0 if multi == "No" else 1,
            0 if internet == "DSL" else (1 if internet == "Fiber optic" else 2),
            encode(security),
            encode(backup),
            encode(device),
            encode(support),
            encode(tv),
            encode(movies),
            0 if contract == "Month-to-month" else (1 if contract == "One year" else 2),
            encode(paperless),
            ["Electronic check", "Mailed check", "Bank transfer", "Credit card"].index(payment),
            monthly,
            total
        ]])

        input_scaled = scaler.transform(input_data)

        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0]

        confidence = max(prob)

    st.markdown("---")

    # ================== RESULT ==================
    col1, col2 = st.columns(2)

    with col1:
        if confidence > 0.75:
            st.error("🔴 High Risk Customer")
        elif confidence > 0.5:
            st.warning("🟠 Medium Risk Customer")
        else:
            st.success("🟢 Low Risk Customer")

        st.metric("Confidence", f"{confidence:.2f}")
        st.progress(int(confidence * 100))

    with col2:
        prob_df = pd.DataFrame({
            "Category": ["No Churn", "Churn"],
            "Probability": prob
        })
        st.bar_chart(prob_df.set_index("Category"))

    # ================== SUMMARY ==================
    st.markdown(f"""
    ### 📌 Summary
    - Prediction: **{"Churn" if pred==1 else "No Churn"}**
    - Confidence: **{confidence:.2f}**
    - Monthly Charges: **{monthly}**
    - Tenure: **{tenure} months**
    """)

    # ================== FEATURE IMPORTANCE ==================
    st.markdown("### 📊 Top Influencing Factors")

    feature_names = [
        'gender','SeniorCitizen','Partner','Dependents','tenure',
        'PhoneService','MultipleLines','InternetService','OnlineSecurity',
        'OnlineBackup','DeviceProtection','TechSupport','StreamingTV',
        'StreamingMovies','Contract','PaperlessBilling','PaymentMethod',
        'MonthlyCharges','TotalCharges'
    ]

    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    st.dataframe(importance_df.head(5))

    # ================== RULE-BASED EXPLANATION ==================
    st.markdown("### 🧠 Key Risk Factors")

    reasons = []

    if monthly > 80:
        reasons.append("High monthly charges")

    if tenure < 12:
        reasons.append("Low tenure")

    if contract == "Month-to-month":
        reasons.append("Short-term contract")

    if internet == "Fiber optic":
        reasons.append("Fiber users show higher churn")

    if reasons:
        for r in reasons:
            st.write(f"- {r}")
    else:
        st.write("No strong churn signals detected")

    # ================== BUSINESS RECOMMENDATION ==================
    st.markdown("### 💡 Recommendation")

    if pred == 1:
        st.warning("""
        📉 High churn probability detected  

        Recommended actions:
        - Offer personalized discounts  
        - Promote long-term contracts  
        - Improve customer support  
        """)
    else:
        st.info("Customer is likely to stay. Maintain engagement.")