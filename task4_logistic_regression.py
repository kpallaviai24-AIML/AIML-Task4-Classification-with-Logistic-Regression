# ==========================================================
# TASK 4 : CLASSIFICATION WITH LOGISTIC REGRESSION
# ==========================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve
)

# ==========================================================
# CREATE FOLDERS
# ==========================================================

os.makedirs("Output_Visualizations", exist_ok=True)
os.makedirs("Predictions", exist_ok=True)
os.makedirs("model", exist_ok=True)

print("=" * 60)
print("TASK 4 : LOGISTIC REGRESSION")
print("=" * 60)

# ==========================================================
# LOAD DATASET
# ==========================================================

df = pd.read_csv("dataset/breast_cancer.csv")

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# ==========================================================
# DATA PREPROCESSING
# ==========================================================

# Remove empty column
if "Unnamed: 32" in df.columns:
    df.drop("Unnamed: 32", axis=1, inplace=True)

# Remove ID column
if "id" in df.columns:
    df.drop("id", axis=1, inplace=True)

# Remove any remaining missing values
df.dropna(inplace=True)

# Encode diagnosis column
encoder = LabelEncoder()

df["diagnosis"] = encoder.fit_transform(
    df["diagnosis"]
)

print("\nDataset Shape After Cleaning:")
print(df.shape)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())
# ==========================================================
# TARGET DISTRIBUTION
# ==========================================================

plt.figure(figsize=(8,5))

sns.countplot(
    x="diagnosis",
    data=df
)

plt.title(
    "Target Distribution",
    fontsize=12,
    fontweight="bold"
)

plt.savefig(
    "Output_Visualizations/target_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================================
# CORRELATION HEATMAP
# ==========================================================

plt.figure(figsize=(14,10))

sns.heatmap(
    df.corr(),
    cmap="coolwarm"
)

plt.title(
    "Correlation Heatmap",
    fontsize=12,
    fontweight="bold"
)

plt.savefig(
    "Output_Visualizations/correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================================
# FEATURES & TARGET
# ==========================================================

X = df.drop(
    "diagnosis",
    axis=1
)

y = df["diagnosis"]

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)

# ==========================================================
# STANDARDIZATION
# ==========================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# ==========================================================
# LOGISTIC REGRESSION MODEL
# ==========================================================

model = LogisticRegression(
    max_iter=5000
)

model.fit(
    X_train,
    y_train
)

print("\nModel Training Completed")

# ==========================================================
# PREDICTIONS
# ==========================================================

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:,1]

# ==========================================================
# EVALUATION METRICS
# ==========================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_prob
)

print("\nModel Performance")
print("-" * 40)

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)
print("ROC AUC  :", roc_auc)

# ==========================================================
# CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title(
    "Confusion Matrix",
    fontsize=12,
    fontweight="bold"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig(
    "Output_Visualizations/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================================
# ROC CURVE
# ==========================================================

fpr, tpr, _ = roc_curve(
    y_test,
    y_prob
)

plt.figure(figsize=(8,5))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {roc_auc:.4f}"
)

plt.plot(
    [0,1],
    [0,1],
    "--"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve",
    fontsize=12,
    fontweight="bold"
)

plt.legend()

plt.savefig(
    "Output_Visualizations/roc_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================================
# PRECISION RECALL CURVE
# ==========================================================

precisions, recalls, _ = precision_recall_curve(
    y_test,
    y_prob
)

plt.figure(figsize=(8,5))

plt.plot(
    recalls,
    precisions
)

plt.xlabel(
    "Recall"
)

plt.ylabel(
    "Precision"
)

plt.title(
    "Precision Recall Curve",
    fontsize=12,
    fontweight="bold"
)

plt.savefig(
    "Output_Visualizations/precision_recall_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================================
# SIGMOID FUNCTION
# ==========================================================

x = np.linspace(
    -10,
    10,
    100
)

sigmoid = 1 / (
    1 + np.exp(-x)
)

plt.figure(figsize=(8,5))

plt.plot(
    x,
    sigmoid
)

plt.title(
    "Sigmoid Function",
    fontsize=12,
    fontweight="bold"
)

plt.xlabel("x")
plt.ylabel("sigmoid(x)")

plt.savefig(
    "Output_Visualizations/sigmoid_function.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================================
# THRESHOLD ANALYSIS
# ==========================================================

thresholds = np.arange(
    0.1,
    1.0,
    0.1
)

acc_scores = []

for threshold in thresholds:

    temp_pred = (
        y_prob >= threshold
    ).astype(int)

    acc_scores.append(
        accuracy_score(
            y_test,
            temp_pred
        )
    )

plt.figure(figsize=(8,5))

plt.plot(
    thresholds,
    acc_scores,
    marker="o"
)

plt.title(
    "Threshold Analysis",
    fontsize=12,
    fontweight="bold"
)

plt.xlabel(
    "Threshold"
)

plt.ylabel(
    "Accuracy"
)

plt.savefig(
    "Output_Visualizations/threshold_analysis.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================================
# PROBABILITY DISTRIBUTION
# ==========================================================

plt.figure(figsize=(8,5))

sns.histplot(
    y_prob,
    bins=20,
    kde=True
)

plt.title(
    "Prediction Probability Distribution",
    fontsize=12,
    fontweight="bold"
)

plt.xlabel(
    "Predicted Probability"
)

plt.savefig(
    "Output_Visualizations/probability_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

coef_df = pd.DataFrame(
    {
        "Feature": X.columns,
        "Coefficient": model.coef_[0]
    }
)

coef_df = coef_df.sort_values(
    by="Coefficient",
    ascending=False
)

coef_df.to_csv(
    "model/feature_coefficients.csv",
    index=False
)

plt.figure(figsize=(10,8))

sns.barplot(
    data=coef_df.head(15),
    x="Coefficient",
    y="Feature"
)

plt.title(
    "Top Logistic Regression Features",
    fontsize=12,
    fontweight="bold"
)

plt.savefig(
    "Output_Visualizations/feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================================
# SAVE REPORTS
# ==========================================================

with open(
    "model/evaluation_metrics.txt",
    "w"
) as f:

    f.write("LOGISTIC REGRESSION EVALUATION\n\n")
    f.write(f"Accuracy  : {accuracy:.4f}\n")
    f.write(f"Precision : {precision:.4f}\n")
    f.write(f"Recall    : {recall:.4f}\n")
    f.write(f"F1 Score  : {f1:.4f}\n")
    f.write(f"ROC AUC   : {roc_auc:.4f}\n")

with open(
    "model/classification_report.txt",
    "w"
) as f:

    f.write(
        classification_report(
            y_test,
            y_pred
        )
    )

with open(
    "model/confusion_matrix_values.txt",
    "w"
) as f:

    f.write("CONFUSION MATRIX\n\n")

    f.write(
        f"True Negative : {cm[0][0]}\n"
    )

    f.write(
        f"False Positive: {cm[0][1]}\n"
    )

    f.write(
        f"False Negative: {cm[1][0]}\n"
    )

    f.write(
        f"True Positive : {cm[1][1]}\n"
    )

threshold_df = pd.DataFrame(
    {
        "Threshold": thresholds,
        "Accuracy": acc_scores
    }
)

threshold_df.to_csv(
    "model/threshold_performance.csv",
    index=False
)

with open(
    "model/model_summary.txt",
    "w"
) as f:

    f.write(
        "LOGISTIC REGRESSION MODEL SUMMARY\n\n"
    )

    f.write(
        f"Training Samples : {len(X_train)}\n"
    )

    f.write(
        f"Testing Samples  : {len(X_test)}\n"
    )

    f.write(
        f"Number of Features : {X.shape[1]}\n"
    )

    f.write(
        f"Intercept : {model.intercept_[0]:.6f}\n\n"
    )

    f.write(
        f"Accuracy  : {accuracy:.4f}\n"
    )

    f.write(
        f"Precision : {precision:.4f}\n"
    )

    f.write(
        f"Recall    : {recall:.4f}\n"
    )

    f.write(
        f"F1 Score  : {f1:.4f}\n"
    )

    f.write(
        f"ROC AUC   : {roc_auc:.4f}\n"
    )

# ==========================================================
# SAVE PREDICTIONS
# ==========================================================

pred_df = pd.DataFrame(
    {
        "Actual": y_test,
        "Predicted": y_pred,
        "Probability": y_prob
    }
)

pred_df.to_csv(
    "Predictions/predicted_results.csv",
    index=False
)

prob_df = pd.DataFrame(
    {
        "Probability": y_prob
    }
)

prob_df.to_csv(
    "Predictions/prediction_probabilities.csv",
    index=False
)

print("\n✅ Task 4 Completed Successfully!")