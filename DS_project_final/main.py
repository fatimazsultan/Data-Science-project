import pandas as pd

from src.data_understanding import (
    data_overview,
    detect_issues,
    feature_analysis
)

from src.missing_handling import (
    missing_analysis,
    missing_patterns,
    impute_missing
)

from src.outlier_handling import (
    visualize_outliers,
    treat_outliers
)

from src.transformation import (
    encode_target,
    scale_features,
    handle_skewness
)

from src.split_data import split_data


# =========================
# LOAD DATA
# =========================
def load_data():
    df = pd.read_csv("data/raw/fruits.csv")
    df.columns = df.columns.str.lower()
    return df


# =========================
# MAIN PIPELINE
# =========================
def main():

    print("\n===== STEP 1: DATA UNDERSTANDING =====")

    df = load_data()

    target = "fruit_type"

    data_overview(df)
    feature_analysis(df)
    detect_issues(df, target)

    print("\n===== STEP 2: MISSING DATA HANDLING =====")

    missing_analysis(df)
    missing_patterns()
    df = impute_missing(df)

    print("\n===== STEP 3: OUTLIERS =====")

    visualize_outliers(df)
    df = treat_outliers(df)

    print("\n===== STEP 4: TRANSFORMATION =====")

    df, le = encode_target(df, target)
    df, scaler = scale_features(df)
    df = handle_skewness(df)

    print("\n===== STEP 5: DATA SPLITTING =====")

    split_data(df, target)

    print("\n===== PIPELINE COMPLETED =====")
    print("All preprocessing steps finished successfully ✅")


if __name__ == "__main__":
    main()

from src.feature_engineering import main as feature_eng_main

feature_eng_main()