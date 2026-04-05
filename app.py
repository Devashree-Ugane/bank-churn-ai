import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os

# ================================
# SESSION STATE INIT
# ================================
if "predicted" not in st.session_state:
    st.session_state.predicted = False

if "input_data" not in st.session_state:
    st.session_state.input_data = None

if "prediction_prob" not in st.session_state:
    st.session_state.prediction_prob = None

if "prediction" not in st.session_state:
    st.session_state.prediction = None

# ================================
# LOAD MODEL
# ================================
with open("models/pipeline.pkl", "rb") as file:
    pipeline = pickle.load(file)

# ================================
# LOAD DATA STATS
# ================================
try:
    with open("models/data_stats.json", "r") as f:
        data_stats = json.load(f)
except:
    data_stats = None

st.title("Bank Customer Churn Prediction")

# ================================
# USER INPUT
# ================================
st.sidebar.header("Customer Information")

def user_input_features():
    country = st.sidebar.selectbox("Country", ["France", "Spain", "Germany"])
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    credit_score = st.sidebar.number_input("Credit Score", 300, 850, 650)
    age = st.sidebar.number_input("Age", 18, 100, 30)
    tenure = st.sidebar.number_input("Tenure", 0, 10, 3)
    balance = st.sidebar.number_input("Balance", 0.0, 200000.0, 50000.0)
    products_number = st.sidebar.number_input("Products", 1, 4, 1)
    credit_card = st.sidebar.selectbox("Has Credit Card", ["Yes", "No"])
    active_member = st.sidebar.selectbox("Active Member", ["Yes", "No"])
    estimated_salary = st.sidebar.number_input("Salary", 0.0, 200000.0, 50000.0)

    data = {
        "credit_score": credit_score,
        "country": country,
        "gender": gender,
        "age": age,
        "tenure": tenure,
        "balance": balance,
        "products_number": products_number,
        "credit_card": 1 if credit_card == "Yes" else 0,
        "active_member": 1 if active_member == "Yes" else 0,
        "estimated_salary": estimated_salary
    }

    return pd.DataFrame(data, index=[0])

input_data = user_input_features()

# ================================
# CLEAN FEATURE NAME
# ================================
def clean_feature_name(name):
    name = name.replace("num__", "").replace("cat__", "")
    return name.replace("_", " ").title()

# ================================
# PREDICT BUTTON
# ================================
if st.button("Predict"):

    prediction_prob = pipeline.predict_proba(input_data)[0][1]
    prediction = pipeline.predict(input_data)[0]

    st.session_state.predicted = True
    st.session_state.input_data = input_data
    st.session_state.prediction_prob = prediction_prob
    st.session_state.prediction = prediction

    input_data["prediction"] = prediction
    input_data["probability"] = prediction_prob

    try:
        existing = pd.read_csv("data/user_inputs.csv")
        updated = pd.concat([existing, input_data], ignore_index=True)
    except:
        updated = input_data

    updated.to_csv("data/user_inputs.csv", index=False)

# ================================
# SHOW RESULTS
# ================================
if st.session_state.predicted:

    input_data = st.session_state.input_data
    prediction_prob = st.session_state.prediction_prob
    prediction = st.session_state.prediction

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Prediction")
        if prediction == 1:
            st.error("⚠️ Customer is likely to churn")
        else:
            st.success("✅ Customer will stay")

    with col2:
        st.subheader("Churn Probability")
        st.write(f"### {prediction_prob:.2%}")

    # ================================
    # WHAT-IF SIMULATOR (4 FEATURES)
    # ================================
    st.divider()
    st.subheader("What-If Simulator")

    new_balance = st.slider("Balance", 0, 200000, int(input_data['balance'][0]))
    new_age = st.slider("Age", 18, 100, int(input_data['age'][0]))
    new_credit = st.slider("Credit Score", 300, 850, int(input_data['credit_score'][0]))
    new_products = st.slider("Number of Products", 1, 4, int(input_data['products_number'][0]))

    simulated_data = input_data.copy()
    simulated_data['balance'] = new_balance
    simulated_data['age'] = new_age
    simulated_data['credit_score'] = new_credit
    simulated_data['products_number'] = new_products

    new_prob = pipeline.predict_proba(simulated_data)[0][1]

    st.write(f"### New Churn Probability: {new_prob:.2%}")

    if new_prob > prediction_prob:
        st.warning("⚠️ Risk increased after changes")
    else:
        st.success("✅ Risk decreased after changes")

    # ================================
    # FEATURE IMPORTANCE
    # ================================
    st.divider()
    st.subheader("Top Factors Affecting Churn")

    model = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]

    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_

    feat_imp = pd.Series(importances, index=feature_names)
    feat_imp = feat_imp.sort_values(ascending=False).head(10)

    feat_imp.index = [clean_feature_name(i) for i in feat_imp.index]

    st.bar_chart(feat_imp)

    # ================================
    # DATA DRIFT
    # ================================
    st.divider()
    st.subheader("Data Quality Check")

    if data_stats:
        drift_flag = False
        for col in input_data.columns:
            if col in data_stats and isinstance(data_stats[col], dict):
                mean = data_stats[col].get("mean", 0)
                std = data_stats[col].get("std", 0)
                val = input_data[col].iloc[0]

                if std > 0 and abs(val - mean) > 3 * std:
                    drift_flag = True

        if drift_flag:
            st.warning("⚠️ Unusual data detected!")
        else:
            st.success("✅ Data looks normal")

    # ================================
    # RECOMMENDATIONS
    # ================================
    st.divider()
    st.subheader("Recommended Next Steps")

    if prediction_prob < 0.3:
        st.write("Customer is low risk. Maintain engagement.")
    else:
        if input_data['active_member'].iloc[0] == 0:
            st.write("• Increase engagement (inactive customer)")
        if input_data['products_number'].iloc[0] == 1:
            st.write("• Cross-sell more products")
        if input_data['balance'].iloc[0] < 1000:
            st.write("• Encourage higher balance")
        if input_data['credit_score'].iloc[0] < 600:
            st.write("• Offer financial advisory")

# ================================
# RETRAIN BUTTON
# ================================
st.divider()
st.subheader("Model Maintenance")

if st.button("Retrain Model"):
    with st.spinner("Retraining model..."):
        try:
            os.system("python train.py")
            st.success("✅ Model retrained successfully!")
        except Exception as e:
            st.error(f"❌ Retraining failed: {e}")