"""
Streamlit Online Fraud Detection Dashboard for Local Vendors
"""

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
import numpy as np
import xgboost

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MAX_UPLOAD_SIZE_MB = 10

# --- 1. CONFIGURATION & UI SETUP ---
st.set_page_config(page_title="Vendor Fraud Detection", page_icon="🛡️", layout="wide")

st.title("🛡️ Online Transaction Fraud Detection")
st.markdown(
    "Upload an anonymized transaction CSV to identify transactions that may "
    "need additional verification."
)

st.info(
    "This dashboard provides risk indicators only. A HIGH risk result does not "
    "confirm fraud and must be verified through the official payment provider."
)

# --- 2. MODEL LOADING ---
@st.cache_resource
def load_models():
    """Loads the pre-trained ML model, scaler, and expected feature columns."""
    with open(MODEL_DIR / "xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODEL_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(MODEL_DIR / "column_names.pkl", "rb") as f:
        expected_cols = pickle.load(f)

    if not expected_cols:
        raise ValueError("The model feature list is empty.")

    return model, scaler, list(expected_cols)

try:
    model, scaler, expected_cols = load_models()
except (OSError, EOFError, ValueError, pickle.UnpicklingError) as error:
    model, scaler, expected_cols = None, None, None
    st.error(f"Unable to load the fraud-detection model files: {error}")

# --- 3. INSTRUCTIONS ---
with st.expander("📖 How to use this tool", expanded=True):
    st.markdown("""
    ### Steps to run an Audit:
    1. Export an anonymized transaction report as a CSV.
    2. Upload it below and click **Run Fraud Detection**.
    3. Review the risk indicators and verify unusual transactions through your payment provider.

    *The CSV should contain transaction details such as Amount, Balances, and Type
    (CASH_OUT, PAYMENT, etc.). Do not upload names, phone numbers, account numbers,
    payment IDs, or other unnecessary personal information.*
    """)

st.divider()

# --- 4. FILE UPLOAD & PROCESSING ---
uploaded_file = st.file_uploader("Upload Transaction CSV", type=["csv"])

if uploaded_file is not None and model is not None:
    if uploaded_file.size > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        st.error(f"Please upload a CSV smaller than {MAX_UPLOAD_SIZE_MB} MB.")
        st.stop()

    if st.button("🔍 Run Fraud Detection", use_container_width=True):
        with st.spinner("Analyzing transactions using AI..."):
            try:
                # Read the CSV file
                df = pd.read_csv(uploaded_file)
                if df.empty:
                    raise ValueError("The uploaded CSV does not contain any rows.")
                if df.columns.duplicated().any():
                    duplicate_columns = df.columns[df.columns.duplicated()].tolist()
                    raise ValueError(
                        f"Duplicate column names are not supported: {duplicate_columns}"
                    )

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
                X = df[expected_cols].apply(pd.to_numeric, errors="coerce")
                invalid_columns = X.columns[X.isna().any()].tolist()
                if invalid_columns:
                    raise ValueError(
                        "These model input columns contain missing or non-numeric "
                        f"values: {invalid_columns}"
                    )

                # --- 6. DATA SCALING ---
                # The saved scaler may have been fitted before one-hot columns
                # were added. Scale only the columns recorded by that scaler,
                # then pass the complete model feature matrix to the model.
                scaler_columns = getattr(scaler, "feature_names_in_", None)
                if scaler_columns is not None:
                    scaler_columns = list(scaler_columns)
                    missing_scaler_columns = [
                        column for column in scaler_columns if column not in X.columns
                    ]
                    if missing_scaler_columns:
                        raise ValueError(
                            "The uploaded data is missing scaler input columns: "
                            f"{missing_scaler_columns}"
                        )
                    scaled_values = scaler.transform(X[scaler_columns])
                    model_input = X.copy()
                    model_input.loc[:, scaler_columns] = scaled_values
                    X_scaled = model_input.to_numpy()
                elif getattr(scaler, "n_features_in_", len(expected_cols)) == len(expected_cols):
                    X_scaled = scaler.transform(X.to_numpy())
                else:
                    raise ValueError(
                        "The saved scaler feature schema is incompatible with the "
                        "model feature list."
                    )

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
                st.caption(
                    "Risk levels are screening indicators. Confirm unusual activity "
                    "using your bank or payment provider before taking action."
                )

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

            except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as error:
                st.error(f"Unable to read the CSV file: {error}")
            except (ValueError, KeyError, TypeError) as error:
                st.error(f"Unable to analyse the uploaded data: {error}")
