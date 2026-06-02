import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler


# =========================
# 1. LABEL ENCODING
# =========================
def encode_target(df, target):
    print("\n===== LABEL ENCODING =====")

    df_copy = df.copy()

    le = LabelEncoder()

    encoded_col = target + "_encoded"
    df_copy[encoded_col] = le.fit_transform(df_copy[target])

    print("\nClass Mapping:")
    for i, class_name in enumerate(le.classes_):
        print(class_name, "->", i)

    return df_copy, le


# =========================
# 2. STANDARDIZATION
# =========================
def scale_features(df):
    print("\n===== FEATURE SCALING (STANDARDIZATION) =====")

    df_copy = df.copy()

    scaler = StandardScaler()

    num_cols = df_copy.select_dtypes(include=["int64", "float64"]).columns

    # remove encoded target if present
    num_cols = [c for c in num_cols if "encoded" not in c]

    df_copy[num_cols] = scaler.fit_transform(df_copy[num_cols])

    print("Standardization applied (Z-score scaling)")

    return df_copy, scaler


# =========================
# 3. SKEWNESS HANDLING (OPTIONAL)
# =========================
def handle_skewness(df):
    print("\n===== SKEWNESS HANDLING =====")

    df_copy = df.copy()

    num_cols = df_copy.select_dtypes(include=["float64", "int64"]).columns

    print("\nSkewness BEFORE:")
    print(df_copy[num_cols].skew())

    for col in num_cols:
        if df_copy[col].skew() > 1:
            df_copy[col] = np.log1p(df_copy[col])

    print("\nSkewness AFTER:")
    print(df_copy[num_cols].skew())

    return df_copy