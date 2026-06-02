import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, classification_report

model = pickle.load(open("models/random_forest.pkl", "rb"))

X_test = pd.read_csv("data/final/X_test_scaled.csv")
y_test = pd.read_csv("data/final/y_test.csv").values.ravel()

# Data cleaning function
def clean_data(X):
    # Replace infinite values
    X = X.replace([float("inf"), float("-inf")], float("nan"))

    # Fill NaN with median
    X = X.fillna(X.median())

    return X
# Clean test data
X_test = clean_data(X_test)    

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Test Accuracy:", acc)

with open("reports/random_forest_report.txt", "w") as f:
    f.write(f"Test Accuracy: {acc}\n\n")
    f.write(report)