import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# =========================
# 1. VISUAL DETECTION (BOXPLOT)
# =========================
def visualize_outliers(df):
    print("\n===== OUTLIER VISUALIZATION =====")

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns

    for col in num_cols:
        plt.figure(figsize=(5, 3))
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col}")
        plt.show()


# =========================
# 2. IQR DETECTION FUNCTION
# =========================
def detect_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower) | (df[col] > upper)]

    return lower, upper, outliers.shape[0]


# =========================
# 3. OUTLIER TREATMENT (CAPPING)
# =========================
def treat_outliers(df):
    print("\n===== OUTLIER DETECTION & TREATMENT =====")

    df_copy = df.copy()

    num_cols = df_copy.select_dtypes(include=["int64", "float64"]).columns

    for col in num_cols:
        lower, upper, count = detect_iqr(df_copy, col)

        print(f"\nColumn: {col}")
        print(f"Outliers found: {count}")
        print(f"Lower bound: {lower}")
        print(f"Upper bound: {upper}")

        # CAPPING
        df_copy[col] = np.where(df_copy[col] < lower, lower, df_copy[col])
        df_copy[col] = np.where(df_copy[col] > upper, upper, df_copy[col])

    print("\nOutliers treated using IQR + CAPPING method")

    return df_copy