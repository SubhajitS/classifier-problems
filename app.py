import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from utils import compute_metrics, TARGET_COL, LEAKAGE_COLS

st.set_page_config(page_title="Teen Mental Health Classifier", layout="wide")

MODEL_REGISTRY = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/svm.joblib"
}

@st.cache_resource
def get_artifacts():
    preprocessor = joblib.load("model/preprocessor.joblib")
    label_encoder = joblib.load("model/label_encoder.joblib")
    return preprocessor, label_encoder

st.title("Teen Digital Wellbeing Multi-Class Classifier")
st.markdown("Evaluate classification models on the **Teen Mental Health** dataset.")

# Sidebar Controls
st.sidebar.header("Configuration")

# 1. Model Selection Dropdown
selected_model_name = st.sidebar.selectbox("Choose Classification Model", list(MODEL_REGISTRY.keys()))

# 2. Dataset Upload Option
uploaded_file = st.sidebar.file_uploader("Upload Test CSV (must contain target column)", type=["csv"])

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    st.sidebar.success("Custom test file uploaded!")
elif os.path.exists("test_data.csv"):
    test_df = pd.read_csv("test_data.csv")
    st.sidebar.info("Using default 'test_data.csv'")
else:
    st.error("No test data found. Upload a test CSV or run prepare_data.py first.")
    st.stop()

# Clean input columns
test_df.columns = test_df.columns.str.strip().str.lower().str.replace(" ", "_")
for col in LEAKAGE_COLS:
    if col in test_df.columns:
        test_df = test_df.drop(columns=[col])

if TARGET_COL not in test_df.columns:
    st.error(f"Uploaded CSV must contain the target column: '{TARGET_COL}'")
    st.stop()

preprocessor, label_encoder = get_artifacts()
model_path = MODEL_REGISTRY[selected_model_name]

if not os.path.exists(model_path):
    st.error(f"Model file `{model_path}` not found. Please run training scripts first.")
    st.stop()

model = joblib.load(model_path)

# Prepare Features and True Labels
X_test_raw = test_df.drop(columns=[TARGET_COL])
y_test_raw = test_df[TARGET_COL].astype(str).str.strip()

X_test = preprocessor.transform(X_test_raw)
y_test = label_encoder.transform(y_test_raw)

# Generate Predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

metrics = compute_metrics(y_test, y_pred, y_proba)

# 3. Display Evaluation Metrics
st.subheader(f"Evaluation Metrics: {selected_model_name}")
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
col2.metric("AUC Score", f"{metrics['AUC']:.4f}")
col3.metric("Precision", f"{metrics['Precision']:.4f}")
col4.metric("Recall", f"{metrics['Recall']:.4f}")
col5.metric("F1 Score", f"{metrics['F1']:.4f}")
col6.metric("MCC Score", f"{metrics['MCC']:.4f}")

st.markdown("---")

# 4. Confusion Matrix and Classification Report
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Confusion Matrix")
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_, ax=ax)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    st.pyplot(fig)

with col_right:
    st.subheader("Classification Report")
    from sklearn.metrics import classification_report
    report_dict = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose()
    st.dataframe(report_df.style.format("{:.3f}"))