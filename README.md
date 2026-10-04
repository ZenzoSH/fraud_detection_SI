# 🛡️ Online Transaction Fraud Detection

An AI-powered dashboard designed to help local vendors review potentially suspicious online transactions. This application takes an anonymized transaction CSV and runs an on-the-fly analysis using a pre-trained XGBoost model to assign risk indicators to every transaction.

## 🚀 Features
- **Machine Learning Integration:** Uses a robust, historical-data-trained XGBoost model.
- **Automated Feature Engineering:** Dynamically processes vendor CSVs, generates necessary dummy variables, and scales data using pre-trained StandardScaler weights.
- **Interactive UI:** Built entirely in Streamlit for an intuitive, easy-to-use merchant experience.
- **One-Click Export:** Download the flagged anomalies and risk-scored dataset instantly.
- **Risk Summary:** Shows LOW, MEDIUM, HIGH, and review-required counts plus amounts.
- **Review Workflow:** Provides filters, sorting, verification status, action notes, and a checklist.
- **Community Feedback:** Collects optional anonymous usability feedback for project improvement.
- **Input Protection:** Validates file size, empty files, duplicate columns, and non-numeric model inputs.

## 📁 Project Structure
```text
fraud_detection_SI/
│
├── app.py                  # Main Streamlit application and ML logic
├── requirements.txt        # Python dependencies
├── sample_data/             # Synthetic CSVs for local testing only
├── .gitignore              # Git ignore file (excludes venv, pycache, etc.)
└── models/                 # Directory containing the pre-trained ML files
    ├── xgb_model.pkl       # The XGBoost prediction engine
    ├── scaler.pkl          # Scikit-Learn standard scaler
    └── column_names.pkl    # Expected feature schema
```

## 🔐 Responsible use

- Use only anonymized or synthetic transaction data for demonstrations.
- Do not upload customer names, phone numbers, account numbers, payment IDs, or other unnecessary personal information.
- A risk level is not proof of fraud. Verify unusual activity through the official bank or payment provider.
- The dashboard is a screening and awareness tool; it does not replace payment-provider controls or human review.
- The current repository does not include model-training or evaluation metrics, so no accuracy claim should be made from this project alone.

---

## 💻 How to Recreate & Run Locally

Follow these steps to set up and run the dashboard on your own machine.

### 1. Clone the Repository
Open your terminal and clone the repository:
```bash
git clone https://github.com/ZenzoSH/fraud_detection_SI.git
cd fraud_detection_SI
```

### 2. Create a Virtual Environment
It is highly recommended to use a virtual environment to manage dependencies and avoid version conflicts (especially with NumPy and Scikit-Learn).
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

### 3. Install Dependencies
Install the required packages. Note that specific older versions (like `numpy<2.0.0` and `scikit-learn==1.2.2`) are pinned to ensure the pre-trained `.pkl` models load correctly without binary incompatibility errors.
```bash
pip install -r requirements.txt
```

### 4. Run the Application
Launch the Streamlit dashboard:
```bash
streamlit run app.py
```
The app will open automatically in your browser at `http://localhost:8501`.

## 🧪 Demonstration

For a safe internship demonstration, prepare a small synthetic CSV containing only
the feature columns expected by the model and no real customer information. Upload
it through the dashboard, review the risk indicators, and download the results.
Record the model version, dataset source, and evaluation metrics separately if they
become available; do not invent these values.

The repository includes synthetic test files in `sample_data/`:

- `sample_transactions.csv` — normal mixed transaction examples.
- `sample_suspicious_transactions.csv` — synthetic high-value patterns for review.
- `sample_invalid_transactions.csv` — invalid numeric values for testing validation errors.

These files contain no real customer, account, or payment information.

---

## ☁️ Deployment (Streamlit Community Cloud)
This project is configured to be instantly deployable to [Streamlit Community Cloud](https://share.streamlit.io/). 
1. Log in to Streamlit Cloud with your GitHub account.
2. Click **New app**.
3. Point it to this repository, select the `main` branch, and set the main file path to `app.py`.
4. Click **Deploy**. Streamlit will handle the rest!
