"""
Streamlit Online Fraud Detection Dashboard for Local Vendors
"""

import pickle
import pandas as pd
import streamlit as st
import numpy as np

# --- 1. CONFIGURATION & UI SETUP ---
st.set_page_config(page_title="Vendor Fraud Detection", page_icon="🛡️", layout="wide")

st.title("🛡️ Online Transaction Fraud Detection")
st.markdown("Upload your daily transaction CSV to detect potentially fraudulent activities.")

# --- 2. MODEL LOADING ---
@st.cache_resource
def load_models():
    """Loads the pre-trained ML model, scaler, and expected feature columns."""
    try:
        with open('models/xgb_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('models/column_names.pkl', 'rb') as f:
            expected_cols = pickle.load(f)
        return model, scaler, expected_cols
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

model, scaler, expected_cols = load_models()

# --- 3. INSTRUCTIONS ---
with st.expander("📖 How to use this tool", expanded=True):
    st.markdown("""
    ### Steps to run an Audit:
    1. Export your transaction report as a CSV.
    2. Upload it below and click **Run Fraud Detection**.
    3. Our pre-trained ML engine (XGBoost) will analyze the historical patterns to detect fraud.

    *Note: The CSV must contain transaction details like Amount, Balances, and Type (CASH_OUT, PAYMENT, etc.).*
    """)

st.divider()

# --- 4. FILE UPLOAD & PROCESSING ---
uploaded_file = st.file_uploader("Upload Transaction CSV", type=["csv"])

if uploaded_file is not None and model is not None:
    if st.button("🔍 Run Fraud Detection", use_container_width=True):
        with st.spinner("Analyzing transactions using AI..."):
            try:
                # Read the CSV file
                df = pd.read_csv(uploaded_file)
                
                # Keep a copy of the original data to display later
                results_df = df.copy()

                # --- 5. FEATURE ENGINEERING ---
                # Create dummy variables for the 'type' column if it exists
                if 'type' in df.columns:
                    type_dummies = pd.get_dummies(df['type'], prefix='type')
                    df = pd.concat([df, type_dummies], axis=1)

                # Ensure all expected columns are present, fill missing with 0
                for col in expected_cols:
                    if col not in df.columns:
                        df[col] = 0

                # Select only the columns the model expects in the correct order
                X = df[expected_cols]

                # --- 6. DATA SCALING ---
                # Scale features using the pre-trained scaler
                X_scaled = scaler.transform(X)

                # --- 7. INFERENCE (PREDICTION) ---
                # Predict fraud probabilities using XGBoost
                # Predict returns [prob_normal, prob_fraud], we take the 2nd column
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_scaled)[:, 1]
                else:
                    # Fallback if predict_proba is not available
                    probs = model.predict(X_scaled)

                # Assign risk levels based on probabilities
                results_df["Fraud_Probability"] = (probs * 100).round(2)
                results_df["Risk_Level"] = np.where(
                    results_df["Fraud_Probability"] >= 75, "HIGH",
                    np.where(results_df["Fraud_Probability"] >= 40, "MEDIUM", "LOW")
                )

                # Move Risk_Level and Fraud_Probability to the front of the dataframe for visibility
                cols = ["Risk_Level", "Fraud_Probability"] + [c for c in results_df.columns if c not in ["Risk_Level", "Fraud_Probability"]]
                results_df = results_df[cols]

                # --- 8. RESULTS DISPLAY ---
                st.subheader("Detection Results")
                
                n_total = len(results_df)
                n_high = (results_df["Risk_Level"] == "HIGH").sum()
                
                m1, m2 = st.columns(2)
                m1.metric("Total Transactions Scanned", n_total)
                m2.metric("High Risk Anomalies", int(n_high))

                # Highlight rows based on risk level
                def highlight_risk(row):
                    if row.get("Risk_Level") == "HIGH":
                        return ["background-color: #5c1a1a"] * len(row)
                    if row.get("Risk_Level") == "MEDIUM":
                        return ["background-color: #5c4a1a"] * len(row)
                    return [""] * len(row)

                st.dataframe(
                    results_df.style.apply(highlight_risk, axis=1),
                    use_container_width=True,
                    hide_index=True,
                )

                # Provide a download button for the results
                st.download_button(
                    "Download Results (CSV)",
                    results_df.to_csv(index=False),
                    file_name="fraud_detection_results.csv",
                    mime="text/csv",
                )

            except Exception as e:
                st.error(f"Error processing the file: {e}")
