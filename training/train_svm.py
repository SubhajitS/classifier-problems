import joblib
import pandas as pd
from sklearn.svm import SVC
from utils import load_preprocessor, compute_metrics, print_evaluation_summary, TARGET_COL


def train():
    preprocessor, label_encoder = load_preprocessor()

    train_df = pd.read_csv("model/train_data_internal.csv")
    test_df = pd.read_csv("test_data.csv")

    X_train = preprocessor.transform(train_df.drop(columns=[TARGET_COL]))
    y_train = label_encoder.transform(train_df[TARGET_COL])
    X_test = preprocessor.transform(test_df.drop(columns=[TARGET_COL]))
    y_test = label_encoder.transform(test_df[TARGET_COL])

    model = SVC(kernel="rbf", probability=True, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    metrics = compute_metrics(y_test, y_pred, y_proba)
    print_evaluation_summary("Support Vector Machine", metrics, y_test, y_pred, label_encoder.classes_)

    joblib.dump(model, "model/svm.joblib")
    print("Model saved to model/svm.joblib")


if __name__ == "__main__":
    train()