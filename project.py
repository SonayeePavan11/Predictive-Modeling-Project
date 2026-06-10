import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from pathlib import Path
from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)

print("=" * 70)
print("ADVANCED LOAN APPROVAL PREDICTION SYSTEM")
print("=" * 70)

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------

base_dir = Path(__file__).resolve().parent
csv_path = base_dir / "dataset.csv"

df = pd.read_csv(csv_path)

print("\nDataset Loaded Successfully")
print("\nShape:", df.shape)

print("\nFirst 5 Records")
print(df.head())

# ---------------------------------------------------
# DATA CLEANING
# ---------------------------------------------------

print("\nChecking Missing Values")
print(df.isnull().sum())

df.fillna(method='ffill', inplace=True)

# ---------------------------------------------------
# ENCODING
# ---------------------------------------------------

encoder = LabelEncoder()

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = encoder.fit_transform(df[col])

print("\nEncoded Dataset")
print(df.head())

# ---------------------------------------------------
# FEATURE SELECTION
# ---------------------------------------------------

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# ---------------------------------------------------
# TRAIN TEST SPLIT
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# ---------------------------------------------------
# DECISION TREE
# ---------------------------------------------------

dt_model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_pred)

print("\nDecision Tree Accuracy:",
      round(dt_accuracy * 100, 2), "%")

# ---------------------------------------------------
# RANDOM FOREST
# ---------------------------------------------------

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)

print("\nRandom Forest Accuracy:",
      round(rf_accuracy * 100, 2), "%")

# ---------------------------------------------------
# CROSS VALIDATION
# ---------------------------------------------------

cv_score = cross_val_score(
    rf_model,
    X,
    y,
    cv=5
)

print("\nCross Validation Scores")
print(cv_score)

print("\nAverage CV Accuracy:",
      round(cv_score.mean() * 100, 2), "%")

# ---------------------------------------------------
# CLASSIFICATION REPORT
# ---------------------------------------------------

print("\nClassification Report")

print(
    classification_report(
        y_test,
        rf_pred
    )
)

# ---------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------

cm = confusion_matrix(
    y_test,
    rf_pred
)

plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    cmap="Blues",
    fmt="d"
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig(base_dir / "confusion_matrix.png")
plt.show()

# ---------------------------------------------------
# ROC CURVE
# ---------------------------------------------------

y_prob = rf_model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)

auc = roc_auc_score(
    y_test,
    y_prob
)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {auc:.2f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.savefig(base_dir / "roc_curve.png")
plt.show()

print("\nAUC Score:", round(auc, 4))

# ---------------------------------------------------
# FEATURE IMPORTANCE
# ---------------------------------------------------

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance")
print(importance)

plt.figure(figsize=(8, 5))

sns.barplot(
    data=importance,
    x="Importance",
    y="Feature"
)

plt.title("Feature Importance")

plt.savefig(base_dir / "feature_importance.png")
plt.show()

# ---------------------------------------------------
# MODEL COMPARISON
# ---------------------------------------------------

comparison = pd.DataFrame({
    "Model": [
        "Decision Tree",
        "Random Forest"
    ],
    "Accuracy": [
        dt_accuracy * 100,
        rf_accuracy * 100
    ]
})

print("\nModel Comparison")
print(comparison)

plt.figure(figsize=(7, 5))

sns.barplot(
    data=comparison,
    x="Model",
    y="Accuracy"
)

plt.title("Model Accuracy Comparison")

plt.savefig(base_dir / "accuracy_comparison.png")
plt.show()

# ---------------------------------------------------
# SAVE BEST MODEL
# ---------------------------------------------------

if rf_accuracy >= dt_accuracy:
    best_model = rf_model
    best_name = "Random Forest"
else:
    best_model = dt_model
    best_name = "Decision Tree"

joblib.dump(
    best_model,
    base_dir / "best_model.pkl"
)

print("\nBest Model Saved :", best_name)

# ---------------------------------------------------
# SAMPLE PREDICTION
# ---------------------------------------------------

sample = pd.DataFrame(
    [[1, 1, 7500, 220, 0]],
    columns=X.columns
)

prediction = best_model.predict(sample)

if prediction[0] == 1:
    print("\nPrediction : Loan Approved")
else:
    print("\nPrediction : Loan Rejected")

print("\nProject Completed Successfully")
print("=" * 70)