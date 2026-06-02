import pandas as pd
from sklearn.model_selection import train_test_split


# =========================
# DATA SPLITTING FUNCTION
# =========================
def split_data(df, target):

    print("\n===== DATA SPLITTING =====")

    # Features (drop original + encoded target)
    X = df.drop(columns=[target, target + "_encoded"])

    # Target
    y = df[target + "_encoded"]

    # 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    # 15% validation, 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    print("Training shape:", X_train.shape)
    print("Validation shape:", X_val.shape)
    print("Test shape:", X_test.shape)

    # Save datasets
    X_train.to_csv("data/final/X_train.csv", index=False)
    X_val.to_csv("data/final/X_val.csv", index=False)
    X_test.to_csv("data/final/X_test.csv", index=False)

    y_train.to_csv("data/final/y_train.csv", index=False)
    y_val.to_csv("data/final/y_val.csv", index=False)
    y_test.to_csv("data/final/y_test.csv", index=False)

    print("\nDatasets saved in data/final folder")

    return X_train, X_val, X_test, y_train, y_val, y_test