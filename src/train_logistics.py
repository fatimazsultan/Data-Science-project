import pandas as pd
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load data
X_train = pd.read_csv("data/final/X_train_scaled.csv")
X_val = pd.read_csv("data/final/X_val_scaled.csv")

y_train = pd.read_csv("data/final/y_train.csv").values.ravel()
y_val = pd.read_csv("data/final/y_val.csv").values.ravel()

# Data cleaning function
def clean_data(X):
    # Replace inf with NaN
    X = X.replace([float("inf"), float("-inf")], float("nan"))

    # Fill NaN with column median
    X = X.fillna(X.median())

    return X

# Clean data
X_train = clean_data(X_train)
X_val = clean_data(X_val)

# Model
model = LogisticRegression(max_iter=1000)

# Train
model.fit(X_train, y_train)

# Validate
y_pred = model.predict(X_val)

acc = accuracy_score(y_val, y_pred)
report = classification_report(y_val, y_pred)

print("Validation Accuracy:", acc)

# Save model
with open("models/logistic_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save outputs
with open("outputs/logistic_metrics.txt", "w") as f:
    f.write(f"Accuracy: {acc}\n\n")
    f.write(report)