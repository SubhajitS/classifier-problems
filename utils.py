import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix
)

TARGET_COL = "digital_wellbeing_flag"
LEAKAGE_COLS = []

def load_preprocessor():
    preprocessor = joblib.load("model/preprocessor.joblib")
    label_encoder = joblib.load("model/label_encoder.joblib")
    return preprocessor, label_encoder

def compute_metrics(y_true, y_pred, y_proba):
    """Calculates all 6 required evaluation metrics for multi-class classification."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    try:
        auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
    except Exception:
        auc = np.nan

    return {
        "Accuracy": round(float(acc), 4),
        "AUC": round(float(auc), 4),
        "Precision": round(float(prec), 4),
        "Recall": round(float(rec), 4),
        "F1": round(float(f1), 4),
        "MCC": round(float(mcc), 4)
    }

def print_evaluation_summary(model_name, metrics, y_true, y_pred, target_names):
    print(f"\n{'='*20} {model_name} Evaluation {'='*20}")
    for k, v in metrics.items():
        print(f"{k:12s}: {v}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))