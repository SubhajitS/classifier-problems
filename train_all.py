import subprocess
import joblib
import pandas as pd
from utils import load_preprocessor, compute_metrics, TARGET_COL

models_info = [
    ("Logistic Regression", "training.train_logistic_regression", "model/logistic_regression.joblib"),
    ("Decision Tree", "training.train_decision_tree", "model/decision_tree.joblib"),
    ("KNN", "training.train_knn", "model/knn.joblib"),
    ("Naive Bayes", "training.train_naive_bayes", "model/naive_bayes.joblib"),
    ("Support Vector Machine", "training.train_svm", "model/svm.joblib"),
]

def run_all():
    print("--> Preparing data split and artifacts...")
    subprocess.run(["python", "-m","training.prepare_data"], check=True)

    for name, script, _ in models_info:
        print(f"\n--> Executing {script}...")
        subprocess.run(["python", "-m", script], check=True)

    # Compile Evaluation Table
    preprocessor, label_encoder = load_preprocessor()
    test_df = pd.read_csv("test_data.csv")
    X_test = preprocessor.transform(test_df.drop(columns=[TARGET_COL]))
    y_test = label_encoder.transform(test_df[TARGET_COL])

    summary_rows = []
    for name, _, path in models_info:
        model = joblib.load(path)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        m = compute_metrics(y_test, y_pred, y_proba)
        m["ML Model Name"] = name
        summary_rows.append(m)

    df_results = pd.DataFrame(summary_rows)[
        ["ML Model Name", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    ]

    print("\n" + "="*75)
    print("Step 5 Markdown Comparison Table for README.md:")
    print("="*75)
    print(df_results)
    #print(df_results.to_markdown(index=False))

if __name__ == "__main__":
    run_all()