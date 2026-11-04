import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from utils import TARGET_COL, LEAKAGE_COLS


def prepare():
    os.makedirs("../model", exist_ok=True)

    # 1. Load raw dataset
    df = pd.read_csv("Teen_Mental_Health.csv")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in: {df.columns.tolist()}")

    # 2. Exclude target leakage columns
    cols_to_drop = [c for c in LEAKAGE_COLS if c in df.columns]
    if cols_to_drop:
        print(f"Dropping target leakage columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

    df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip()

    # 3. Stratified 80/20 Train-Test Split
    train_df, test_df = train_test_split(
        df, test_size=0.20, random_state=42, stratify=df[TARGET_COL]
    )

    # Save test_data.csv to root for evaluation and Streamlit upload
    test_df.to_csv("test_data.csv", index=False)
    train_df.to_csv("../model/train_data_internal.csv", index=False)
    print(f"Saved test_data.csv ({len(test_df)} samples) and internal train split ({len(train_df)} samples)")

    # 4. Fit Feature Encoders & Scalers
    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]

    numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    print(f"Total features: {len(numeric_cols) + len(categorical_cols)} (Requirement: >= 12)")
    print(f"Numeric features ({len(numeric_cols)}): {numeric_cols}")
    print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
        ]
    )
    preprocessor.fit(X_train)

    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)

    # 5. Save preprocessing artifacts using joblib
    joblib.dump(preprocessor, "model/preprocessor.joblib")
    joblib.dump(label_encoder, "model/label_encoder.joblib")
    print("Preprocessors successfully saved to model/*.joblib")


if __name__ == "__main__":
    prepare()