import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def normalize_dataset(X_train, X_val, X_test):
    print("\n===== NORMALIZATION (MIN-MAX SCALING) =====")

    scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    print("Normalization completed successfully!")

    return X_train_scaled, X_val_scaled, X_test_scaled


def main():
    # Load datasets
    X_train = pd.read_csv("data/final/X_train.csv")
    X_val = pd.read_csv("data/final/X_val.csv")
    X_test = pd.read_csv("data/final/X_test.csv")

    # Drop fruit_id if exists (string column)
    if "fruit_id" in X_train.columns:
        X_train = X_train.drop(columns=["fruit_id"])
        X_val = X_val.drop(columns=["fruit_id"])
        X_test = X_test.drop(columns=["fruit_id"])
        print("fruit_id column removed (not useful for ML).")

    # Remove any other non-numeric columns automatically
    X_train = X_train.select_dtypes(include=["int64", "float64"])
    X_val = X_val.select_dtypes(include=["int64", "float64"])
    X_test = X_test.select_dtypes(include=["int64", "float64"])

    print("\nRemaining Columns for Scaling:", list(X_train.columns))

    # Normalize
    X_train_scaled, X_val_scaled, X_test_scaled = normalize_dataset(X_train, X_val, X_test)

    # Save scaled datasets
    X_train_scaled.to_csv("data/final/X_train_scaled.csv", index=False)
    X_val_scaled.to_csv("data/final/X_val_scaled.csv", index=False)
    X_test_scaled.to_csv("data/final/X_test_scaled.csv", index=False)

    print("\nScaled datasets saved successfully in data/final/")


if __name__ == "__main__":
    main()