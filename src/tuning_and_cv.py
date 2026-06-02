import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# =========================
# LOAD DATA
# =========================
X_train = pd.read_csv("data/final/X_train.csv")
y_train = pd.read_csv("data/final/y_train.csv").values.ravel()

# Drop unnecessary string columns
drop_cols = ["fruit_id", "grade"]

for col in drop_cols:
    if col in X_train.columns:
        X_train = X_train.drop(columns=[col])

# Keep only numeric columns
X_train = X_train.select_dtypes(include=["int64", "float64"])

# =========================
# CLEAN DATA (IMPORTANT)
# =========================
def clean_data(X):
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    return X

X_train = clean_data(X_train)

# =========================
# STEP 1: HYPERPARAMETER TUNING (RANDOM FOREST)
# =========================
print("\n===== GRID SEARCH (RANDOM FOREST) =====")

rf = RandomForestClassifier(random_state=42)

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, 15, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

grid = GridSearchCV(
    rf,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
print("Best CV Score:", grid.best_score_)

best_rf = grid.best_estimator_

# =========================
# STEP 2: CROSS VALIDATION (ALL MODELS)
# =========================
print("\n===== CROSS VALIDATION (5-FOLD) =====")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest (Tuned)": best_rf
}

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5)

    print(f"\n{name}")
    print("Scores:", scores)
    print("Mean Accuracy:", scores.mean())

# =========================
# STEP 3: FEATURE IMPORTANCE (RF)
# =========================
print("\n===== FEATURE IMPORTANCE (RANDOM FOREST) =====")

importances = best_rf.feature_importances_
features = X_train.columns

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print(importance_df.head(10))

# Save importance
importance_df.to_csv("outputs/feature_importance.csv", index=False)

# =========================
# STEP 4: LOGISTIC COEFFICIENTS
# =========================
print("\n===== LOGISTIC REGRESSION COEFFICIENTS =====")

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

coef_df = pd.DataFrame(log_model.coef_, columns=X_train.columns)

print(coef_df.head())

coef_df.to_csv("outputs/logistic_coefficients.csv", index=False)