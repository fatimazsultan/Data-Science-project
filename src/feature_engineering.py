import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ============================================
# LOAD DATA
# ============================================
def load_data():
    df = pd.read_csv("data/raw/fruits.csv")
    df.columns = df.columns.str.lower()
    return df


# ============================================
# FEATURE ENGINEERING (CREATE NEW FEATURES)
# ============================================
def feature_engineering(df):
    print("\n===== FEATURE ENGINEERING =====")

    df["sweetness_firmness_ratio"] = df["sweetness"] / (df["firmness"] + 0.0001)
    df["defect_density"] = df["defect_count"] / (df["size_cm"] + 0.0001)
    df["ripeness_sweetness_score"] = df["ripeness_level"] * df["sweetness"]

    print("New Features Added:")
    print(["sweetness_firmness_ratio", "defect_density", "ripeness_sweetness_score"])

    return df


# ============================================
# FEATURE SELECTION USING MUTUAL INFORMATION
# ============================================
def feature_selection(X_train, y_train, top_n=8):
    print("\n===== FEATURE SELECTION (MUTUAL INFORMATION) =====")

    mi_scores = mutual_info_classif(X_train, y_train, random_state=42)

    mi_df = pd.DataFrame({
        "Feature": X_train.columns,
        "MI_Score": mi_scores
    }).sort_values(by="MI_Score", ascending=False)

    print("\nTop Feature Scores:\n")
    print(mi_df.head(top_n))

    top_features = mi_df["Feature"].head(top_n).tolist()

    print("\nSelected Top Features:", top_features)

    return top_features


# ============================================
# BASELINE MODEL TRAINING
# ============================================
def train_model(X_train, X_test, y_train, y_test, title="Model"):
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy ({title}): {acc:.4f}")

    return acc


# ============================================
# PCA REDUCTION
# ============================================
def apply_pca(X_train, X_test):
    print("\n===== PCA DIMENSIONALITY REDUCTION =====")

    pca = PCA(n_components=0.95)  # Keep 95% variance

    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    print("Original Features:", X_train.shape[1])
    print("Reduced PCA Features:", X_train_pca.shape[1])

    return X_train_pca, X_test_pca


# ============================================
# MAIN FUNCTION
# ============================================
def main():
    print("\n===============================")
    print("   STEP 7.3 FEATURE ENGINEERING")
    print("===============================\n")

    df = load_data()
    target = "fruit_type"

    # Encode target
    le = LabelEncoder()
    df[target + "_encoded"] = le.fit_transform(df[target])

    # Feature Engineering
    df = feature_engineering(df)

    # Drop fruit_id (not useful + causes string error)
    if "fruit_id" in df.columns:
        df = df.drop(columns=["fruit_id"])

    # Prepare X and y
    X = df.drop(columns=[target, target + "_encoded"], errors="ignore")
    y = df[target + "_encoded"]

    # One-hot encode any categorical columns inside X
    X = pd.get_dummies(X, drop_first=True)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # ------------------------------
    # BASELINE PERFORMANCE
    # ------------------------------
    baseline_acc = train_model(X_train, X_test, y_train, y_test, title="Baseline (All Features)")

    # ------------------------------
    # FEATURE SELECTION
    # ------------------------------
    top_features = feature_selection(X_train, y_train, top_n=8)

    X_train_selected = X_train[top_features]
    X_test_selected = X_test[top_features]

    fs_acc = train_model(X_train_selected, X_test_selected, y_train, y_test, title="After Feature Selection")

    # ------------------------------
    # PCA PERFORMANCE
    # ------------------------------
    X_train_pca, X_test_pca = apply_pca(X_train, X_test)

    pca_acc = train_model(X_train_pca, X_test_pca, y_train, y_test, title="After PCA")

    # ------------------------------
    # FINAL COMPARISON
    # ------------------------------
    print("\n===============================")
    print("   FINAL PERFORMANCE COMPARISON")
    print("===============================")

    print(f"Baseline Accuracy: {baseline_acc:.4f}")
    print(f"Feature Selection Accuracy: {fs_acc:.4f}")
    print(f"PCA Accuracy: {pca_acc:.4f}")

    print("\n===== STEP 7.3 COMPLETED SUCCESSFULLY ✅ =====")


if __name__ == "__main__":
    main()