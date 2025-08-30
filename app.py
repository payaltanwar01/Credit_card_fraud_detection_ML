import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from imblearn.over_sampling import SMOTE

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("💳 Credit Card Fraud Detection")

# ---------------- File Upload ----------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset loaded successfully!")
    st.write(df.head())

    # ---------------- Preprocessing ----------------
    scaler = StandardScaler()
    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_scaled = scaler.fit_transform(X)

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_scaled, y)

    st.info(f"Training set size after SMOTE: {X_res.shape[0]} samples")

    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42
    )

    # ---------------- Model Selection ----------------
    model_choice = st.selectbox("Select Model", ["Logistic Regression", "Random Forest"])

    if model_choice == "Logistic Regression":
        model = LogisticRegression(max_iter=1000, solver="liblinear", random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)

    # ---------------- Prediction ----------------
    if model_choice == "Logistic Regression":
        y_probs = model.predict_proba(X_test)[:, 1]
        threshold = st.sidebar.slider("Select Threshold", 0.0, 1.0, 0.5)
        y_pred = (y_probs >= threshold).astype(int)
    else:
        y_pred = model.predict(X_test)
        y_probs = model.predict_proba(X_test)[:, 1]

    # ---------------- Evaluation ----------------
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    st.subheader("📊 Model Performance Metrics")
    st.write(f"**Accuracy:** {acc:.4f}")
    st.write(f"**Precision:** {prec:.4f}")
    st.write(f"**Recall:** {rec:.4f}")
    st.write(f"**F1 Score:** {f1:.4f}")

    # ---------------- ROC Curve ----------------
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    st.subheader("📈 ROC Curve")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "r--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    st.pyplot(fig)

    # ---------------- Real-time Alert ----------------
    st.subheader("⚡ Real-time Transaction Alert")
    new_txn_amount = st.number_input("Enter Transaction Amount", min_value=0.0, value=100.0)
    new_txn_features = np.array([new_txn_amount] + [0] * (X.shape[1] - 1)).reshape(1, -1)
    new_txn_scaled = scaler.transform(new_txn_features)

    if model_choice == "Logistic Regression":
        prob = model.predict_proba(new_txn_scaled)[0][1]
        alert_threshold = st.slider("Alert Threshold", 0.0, 1.0, 0.5)
        alert = "⚠️ Fraudulent Transaction!" if prob >= alert_threshold else "✅ Transaction Safe"
    else:
        pred = model.predict(new_txn_scaled)[0]
        alert = "⚠️ Fraudulent Transaction!" if pred == 1 else "✅ Transaction Safe"

    st.write(alert)

else:
    st.info("📂 Please upload a CSV file to begin.")
