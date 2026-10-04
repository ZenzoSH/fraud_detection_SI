"""
Streamlit Online Fraud Detection Dashboard for Local Vendors
"""

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
import numpy as np
import xgboost
from app_helpers import risk_summary

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MAX_UPLOAD_SIZE_MB = 10
REQUIRED_NUMERIC_COLUMNS = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]
TRANSACTION_TYPE_COLUMNS = [
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER",
]
VERIFICATION_STATUSES = [
    "Not Reviewed",
    "Verified Successful",
    "Payment Pending",
    "Payment Failed",
    "Payment Reversed",
    "Requires Provider Support",
    "No Issue Found",
]

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


def show_actionable_error(title, problem, fix, example=None):
    """Show a known problem together with steps the user can take to fix it."""
    st.error(f"{title}: {problem}")
    with st.expander("How to fix this", expanded=True):
        st.markdown(fix)
        if example:
            st.code(example, language="text")


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
    show_actionable_error(
        "Model files could not be loaded",
        str(error),
        "Confirm that `models/xgb_model.pkl`, `models/scaler.pkl`, and "
        "`models/column_names.pkl` exist. Install the versions from "
        "`requirements.txt`, then restart Streamlit.",
    )

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
        show_actionable_error(
            "File is too large",
            f"The upload is larger than {MAX_UPLOAD_SIZE_MB} MB.",
            "Export only the columns needed for analysis, remove unnecessary "
            "rows, or split the file into smaller CSV files.",
        )
        st.stop()

    if st.button("🔍 Run Fraud Detection", use_container_width=True):
        with st.spinner("Analyzing transactions using AI..."):
            try:
                # Read the CSV file
                df = pd.read_csv(uploaded_file)
                if df.empty:
                    show_actionable_error(
                        "CSV contains no data rows",
                        "The file has headers but no transaction records.",
                        "Add at least one transaction row below the header, save "
                        "the file as CSV, and upload it again. You can use the "
                        "files in `sample_data/` as formatting examples.",
                        ", ".join(REQUIRED_NUMERIC_COLUMNS + ["type"]),
                    )
                    st.stop()
                if df.columns.duplicated().any():
                    duplicate_columns = df.columns[df.columns.duplicated()].tolist()
                    show_actionable_error(
                        "Duplicate column names",
                        f"These headers appear more than once: {duplicate_columns}",
                        "Open the CSV in a spreadsheet editor, rename or remove the "
                        "duplicate headers, save it again as CSV, and re-upload it.",
                    )
                    st.stop()

                missing_numeric_columns = [
                    column for column in REQUIRED_NUMERIC_COLUMNS if column not in df.columns
                ]
                if missing_numeric_columns:
                    raise ValueError(
                        "Required transaction columns are missing: "
                        f"{missing_numeric_columns}"
                    )

                has_raw_type = "type" in df.columns
                has_encoded_type = any(
                    column in df.columns for column in TRANSACTION_TYPE_COLUMNS
                )
                if not has_raw_type and not has_encoded_type:
                    raise ValueError(
                        "No transaction type column was found. Add a `type` column "
                        "or the expected type indicator columns."
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
                results_df["Verification_Status"] = "Not Reviewed"
                results_df["Action_Taken"] = ""

                summary = risk_summary(results_df)
                summary_columns = st.columns(5)
                summary_columns[0].metric("Transactions Scanned", summary["total"])
                summary_columns[1].metric("LOW Risk", summary["low"])
                summary_columns[2].metric("MEDIUM Risk", summary["medium"])
                summary_columns[3].metric("HIGH Risk", summary["high"])
                summary_columns[4].metric("Needs Review", summary["review_count"])
                st.caption(
                    f"Total amount: {summary['total_amount']:,.2f} | "
                    f"LOW/MEDIUM/HIGH amount requiring review: "
                    f"{summary['review_amount']:,.2f}"
                )
                st.caption(
                    "Risk levels are screening indicators. Confirm unusual activity "
                    "using your bank or payment provider before taking action."
                )

                st.subheader("Risk explanation")
                st.markdown(
                    "- **LOW (below 40):** Continue normal payment verification.\n"
                    "- **MEDIUM (40 to below 75):** Perform additional checks.\n"
                    "- **HIGH (75 or above):** Review immediately through the official "
                    "payment provider.\n\n"
                    "These are model-generated screening indicators, not confirmed "
                    "probabilities of fraud."
                )

                st.subheader("Review and filter results")
                filter_columns = st.columns(3)
                selected_risks = filter_columns[0].multiselect(
                    "Risk levels",
                    ["HIGH", "MEDIUM", "LOW"],
                    default=["HIGH", "MEDIUM", "LOW"],
                )
                selected_types = filter_columns[1].multiselect(
                    "Transaction types",
                    sorted(results_df["type"].dropna().unique()),
                    default=sorted(results_df["type"].dropna().unique()),
                )
                sort_by = filter_columns[2].selectbox(
                    "Sort results by",
                    ["Fraud_Probability", "amount", "step"],
                )
                filtered_df = results_df[
                    results_df["Risk_Level"].isin(selected_risks)
                    & results_df["type"].isin(selected_types)
                ].sort_values(sort_by, ascending=False)

                # Highlight rows based on risk level
                def highlight_risk(row):
                    if row.get("Risk_Level") == "HIGH":
                        return ["background-color: #5c1a1a"] * len(row)
                    if row.get("Risk_Level") == "MEDIUM":
                        return ["background-color: #5c4a1a"] * len(row)
                    return [""] * len(row)

                edited_df = st.data_editor(
                    filtered_df.style.apply(highlight_risk, axis=1),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Verification_Status": st.column_config.SelectboxColumn(
                            "Verification Status",
                            options=VERIFICATION_STATUSES,
                            required=True,
                        ),
                        "Action_Taken": st.column_config.TextColumn(
                            "Action Taken",
                            help="Optional: record the verification action without "
                            "entering customer or account information.",
                        ),
                    },
                    disabled=[
                        column
                        for column in filtered_df.columns
                        if column not in {"Verification_Status", "Action_Taken"}
                    ],
                    key="results_editor",
                )

                # Provide a download button for the results
                st.download_button(
                    "Download Results (CSV)",
                    edited_df.to_csv(index=False),
                    file_name="fraud_detection_results.csv",
                    mime="text/csv",
                )

                st.subheader("Review checklist")
                checklist_columns = st.columns(2)
                checklist_items = [
                    "Checked the official payment-provider status",
                    "Confirmed the amount matches the sale",
                    "Confirmed the transaction is successful",
                    "Checked that the payment is not pending or reversed",
                    "Contacted official support if required",
                    "Recorded the verification result without private customer data",
                ]
                for index, item in enumerate(checklist_items):
                    checklist_columns[index % 2].checkbox(item, key=f"check_{index}")

                st.subheader("What to do next")
                st.markdown(
                    """
                    Use this report to **prioritise verification**. A risk level is
                    only a screening indicator and does not confirm fraud.

                    1. **Review HIGH rows first.** Open each transaction in the
                       official bank or payment-provider application and confirm
                       that the amount, status, and transaction reference match.
                    2. **Check MEDIUM rows next.** Verify the payment status,
                       amount, transaction type, and balance changes before
                       completing or settling the sale.
                    3. **Continue normal checks for LOW rows.** Confirm that the
                       payment is successful in the official account; LOW does not
                       guarantee that a transaction is safe.
                    4. **Do not rely on screenshots or this dashboard alone.**
                       If a payment is missing, pending, reversed, or inconsistent,
                       pause delivery and contact the payment provider through an
                       official support channel.
                    5. **Keep a secure review record.** Record the transaction
                       reference, verification result, and follow-up action without
                       publishing customer or account information.
                    """
                )
                st.warning(
                    "Never accuse a customer, issue a refund, or hand over goods "
                    "based only on a model result. Verify the payment independently."
                )

                with st.expander("Anonymous user feedback"):
                    st.caption(
                        "This optional feedback is for improving the project. Do not "
                        "enter names, phone numbers, account numbers, or transaction IDs."
                    )
                    feedback_clarity = st.radio(
                        "Was the dashboard easy to understand?",
                        ["Yes", "Partly", "No"],
                        horizontal=True,
                    )
                    feedback_usefulness = st.radio(
                        "Would this review workflow be useful for your shop?",
                        ["Yes", "Partly", "No"],
                        horizontal=True,
                    )
                    feedback_improvement = st.selectbox(
                        "Which improvement would help most?",
                        [
                            "Simpler explanations",
                            "More filters",
                            "Better transaction reports",
                            "Payment-provider integration",
                            "Training or awareness material",
                        ],
                    )
                    feedback_notes = st.text_area(
                        "Additional feedback (optional)",
                        max_chars=500,
                        help="Keep this anonymous and do not enter transaction details.",
                    )
                    if st.button("Save anonymous feedback"):
                        st.success(
                            "Thank you. Feedback captured for this session only: "
                            f"{feedback_clarity}, {feedback_usefulness}, "
                            f"{feedback_improvement}."
                        )

            except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as error:
                show_actionable_error(
                    "CSV could not be read",
                    str(error),
                    "Open the file in a spreadsheet editor and export it as "
                    "comma-separated CSV using UTF-8 encoding. Ensure every row "
                    "has the same number of columns and does not contain broken "
                    "quotes or unsupported separators.",
                )
            except (ValueError, KeyError, TypeError) as error:
                message = str(error)
                if "Required transaction columns are missing" in message:
                    fix = (
                        "Add all required numeric columns to the CSV. Use the exact "
                        "lowercase headers shown below; do not rename them."
                    )
                    example = ", ".join(REQUIRED_NUMERIC_COLUMNS)
                elif "No transaction type column" in message:
                    fix = (
                        "Add a `type` column containing values such as PAYMENT, "
                        "TRANSFER, CASH_OUT, or DEBIT. Alternatively, provide the "
                        "four encoded type columns shown below."
                    )
                    example = "type\nPAYMENT\nTRANSFER\nCASH_OUT"
                elif "missing or non-numeric" in message:
                    fix = (
                        "Find the listed columns in your CSV and replace text such "
                        "as `missing`, `N/A`, currency symbols, or blank cells with "
                        "valid numbers. Save the corrected file and upload it again."
                    )
                    example = "amount,oldbalanceOrg\n250.00,1000.00"
                elif "feature schema is incompatible" in message:
                    fix = (
                        "The saved model and scaler do not belong to the same "
                        "feature version. Restore the model files from this "
                        "repository together and restart the app."
                    )
                    example = None
                elif "feature names" in message or "n_features" in message:
                    fix = (
                        "The model artifacts and preprocessing schema do not match. "
                        "Make sure all files in `models/` came from the same project "
                        "version, then restart Streamlit. Do not edit the pickle "
                        "files manually."
                    )
                    example = None
                else:
                    fix = (
                        "Check that the CSV uses the expected headers and numeric "
                        "values. If the problem continues, use one of the synthetic "
                        "files in `sample_data/` to confirm that the application "
                        "environment is configured correctly."
                    )
                    example = None
                show_actionable_error("Unable to analyse the uploaded data", message, fix, example)
