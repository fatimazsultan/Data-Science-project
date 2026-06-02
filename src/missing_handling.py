import pandas as pd

# =========================
# 1. MISSING VALUE ANALYSIS
# =========================
def missing_analysis(df):
    print("\n===== MISSING VALUE ANALYSIS =====")

    missing = df.isnull().sum()
    print(missing[missing > 0])

    print("\nTotal Missing Values:", df.isnull().sum().sum())


# =========================
# 2. MCAR / MAR / MNAR EXPLANATION
# =========================
def missing_patterns():
    print("\n===== MISSING DATA PATTERNS =====")

    print("""
MCAR (Missing Completely At Random):
- Missing values occur randomly
- No relationship with any feature

MAR (Missing At Random):
- Missing depends on other observed variables
- Example: weight missing depends on fruit size

MNAR (Missing Not At Random):
- Missing depends on actual missing value
- Example: very bad fruit quality not recorded

In fruit dataset:
- Mostly MAR (sensor or measurement dependency)
""")


# =========================
# 3. IMPUTATION METHODS
# =========================
def impute_missing(df):
    df_copy = df.copy()

    # numerical columns
    num_cols = df_copy.select_dtypes(include=["int64", "float64"]).columns

    # categorical columns
    cat_cols = df_copy.select_dtypes(include=["object"]).columns

    print("\n===== IMPUTATION STARTED =====")

    # METHOD 1: Median for numerical
    for col in num_cols:
        df_copy[col] = df_copy[col].fillna(df_copy[col].median())

    # METHOD 2: Mode for categorical
    for col in cat_cols:
        df_copy[col] = df_copy[col].fillna(df_copy[col].mode()[0])

    print("Missing values handled using Median + Mode")

    return df_copy