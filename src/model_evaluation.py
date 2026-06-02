import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from sklearn.preprocessing import label_binarize


# ======================================
# CLEAN DATA FUNCTION
# ======================================
def clean_data(X):
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    return X


# ======================================
# LOAD TEST DATA
# ======================================
X_test = pd.read_csv("data/final/X_test_scaled.csv")
y_test = pd.read_csv("data/final/y_test.csv").values.ravel()

# Drop unnecessary columns
for col in ["fruit_id", "grade"]:
    if col in X_test.columns:
        X_test = X_test.drop(columns=[col])

X_test = X_test.select_dtypes(include=["int64", "float64"])
X_test = clean_data(X_test)


# ======================================
# LOAD MODELS
# ======================================
models = {
    "Logistic Regression": pickle.load(open("models/logistic_model.pkl", "rb")),
    "Decision Tree": pickle.load(open("models/decision_tree.pkl", "rb")),
    "Random Forest": pickle.load(open("models/random_forest.pkl", "rb"))
}


# ======================================
# ROC SETUP
# ======================================
classes = np.unique(y_test)
n_classes = len(classes)

y_test_bin = label_binarize(y_test, classes=classes)


# ======================================
# RESULTS STORAGE
# ======================================
results = []


# ======================================
# MAIN EVALUATION LOOP
# ======================================
for name, model in models.items():

    print(f"\n==============================")
    print(f"     {name} Evaluation")
    print(f"==============================")

    # -------------------------
    # PREDICTIONS
    # -------------------------
    y_pred = model.predict(X_test)

    # -------------------------
    # BASIC METRICS
    # -------------------------
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Classification Report
    report = classification_report(y_test, y_pred)

    print("\nAccuracy:", acc)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    print("\nClassification Report:\n")
    print(report)

    # Save detailed report
    with open(f"reports/{name.lower().replace(' ', '_')}_full_report.txt", "w") as f:
        f.write(f"Accuracy: {acc}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"F1 Score: {f1}\n\n")
        f.write(report)

    # Save results summary
    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    # -------------------------
    # CONFUSION MATRIX
    # -------------------------
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"{name} Confusion Matrix")
    plt.savefig(f"outputs/{name.lower().replace(' ', '_')}_confusion_matrix.png")
    plt.close()

    # -------------------------
    # ROC CURVE + AUC
    # -------------------------
    if hasattr(model, "predict_proba"):

        y_score = model.predict_proba(X_test)

        fpr = {}
        tpr = {}
        roc_auc = {}

        plt.figure()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

            plt.plot(
                fpr[i],
                tpr[i],
                label=f"Class {i} AUC = {roc_auc[i]:.2f}"
            )

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{name} ROC Curve")
        plt.legend()

        plt.savefig(f"outputs/{name.lower().replace(' ', '_')}_roc_curve.png")
        plt.close()

        # Save AUC
        with open(f"outputs/{name.lower().replace(' ', '_')}_auc.txt", "w") as f:
            for i in range(n_classes):
                f.write(f"Class {i}: {roc_auc[i]:.4f}\n")


# ======================================
# MODEL COMPARISON TABLE
# ======================================
comparison_df = pd.DataFrame(results)

comparison_df.to_csv("outputs/model_comparison.csv", index=False)

print("\n==============================")
print("      MODEL COMPARISON")
print("==============================")
print(comparison_df)