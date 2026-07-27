# 🛡️ Online Transaction Fraud Detection

An AI-powered dashboard designed to help local vendors detect potentially fraudulent online transactions. This application takes daily settlement CSVs and runs an on-the-fly analysis using a pre-trained XGBoost model to assign risk scores to every transaction.

## 🚀 Features
- **Machine Learning Integration:** Uses a robust, historical-data-trained XGBoost model.
- **Automated Feature Engineering:** Dynamically processes vendor CSVs, generates necessary dummy variables, and scales data using pre-trained StandardScaler weights.
- **Interactive UI:** Built entirely in Streamlit for an intuitive, easy-to-use merchant experience.
- **One-Click Export:** Download the flagged anomalies and risk-scored dataset instantly.

## 📁 Project Structure
```text
fraud_detection_SI/
│
├── app.py                  # Main Streamlit application and ML logic
├── requirements.txt        # Python dependencies
├── .gitignore              # Git ignore file (excludes venv, pycache, etc.)
└── models/                 # Directory containing the pre-trained ML files
    ├── xgb_model.pkl       # The XGBoost prediction engine
    ├── scaler.pkl          # Scikit-Learn standard scaler
    └── column_names.pkl    # Expected feature schema
```

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

---

## ☁️ Deployment (Streamlit Community Cloud)
This project is configured to be instantly deployable to [Streamlit Community Cloud](https://share.streamlit.io/). 
1. Log in to Streamlit Cloud with your GitHub account.
2. Click **New app**.
3. Point it to this repository, select the `main` branch, and set the main file path to `app.py`.
4. Click **Deploy**. Streamlit will handle the rest!
