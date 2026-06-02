import pandas as pd

def data_overview(df):
    print("\n===== DATASET OVERVIEW =====")
    print("Shape:", df.shape)
    print("\nColumns:\n", df.columns)
    print("\nData Types:\n", df.dtypes)


def detect_issues(df, target):
    print("\n===== DATA QUALITY CHECK =====")

    # missing values
    print("\nMissing Values:\n", df.isnull().sum())

    # duplicates
    print("\nDuplicate Rows:", df.duplicated().sum())

    # class imbalance
    if target in df.columns:
        print("\nClass Distribution:\n", df[target].value_counts())


def feature_analysis(df):
    print("\n===== FEATURE TYPES =====")

    num = df.select_dtypes(include=["int64", "float64"]).columns
    cat = df.select_dtypes(include=["object"]).columns

    print("Numerical Features:", list(num))
    print("Categorical Features:", list(cat))