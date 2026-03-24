"""
Project: Heart Failure Prediction System

Objective:
Predict the likelihood of heart failure using clinical features
to assist early medical diagnosis.

Approach:
- Data preprocessing (scaling, splitting)
- Multiple ML models comparison
- Evaluation using Accuracy, ROC-AUC, Confusion Matrix
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

sns.set(style="whitegrid")
df = pd.read_csv(
    "heart_failure_clinical_records_dataset.csv"
)

print(df.head())
print(df.info())
print(df.describe())
X = df.drop("DEATH_EVENT", axis=1)
y = df["DEATH_EVENT"]

# Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation")
plt.show()

# Class distribution
sns.countplot(x="DEATH_EVENT", data=df)
plt.title("Class Distribution")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
models = {

    "Logistic Regression": LogisticRegression(
        penalty="l2",
        C=0.5,                  # regularization strength
        solver="liblinear",
        class_weight="balanced",
        max_iter=2000,
        random_state=42
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=11,         # smoother decision boundary
        weights="distance",     # closer points matter more
        metric="minkowski",
        p=2                     # Euclidean distance
    ),

    "Naive Bayes": GaussianNB(
        var_smoothing=1e-8      # improves numerical stability
    ),

    "Decision Tree": DecisionTreeClassifier(
        criterion="entropy",
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42
    ),

    "SVM": SVC(
        kernel="rbf",           # nonlinear decision boundary
        C=1.5,
        gamma="scale",
        probability=True,
        class_weight="balanced",
        random_state=42
    )
}
results = {}
reports = {}
conf_matrices = {}
roc_data = {}

for name, model in models.items():
    
    # Choose scaled or unscaled data
    if name in ["Naive Bayes", "Decision Tree"]:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # Store results
    results[name] = accuracy_score(y_test, y_pred)
    reports[name] = classification_report(y_test, y_pred, output_dict=True)
    conf_matrices[name] = confusion_matrix(y_test, y_pred)
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_data[name] = (fpr, tpr, auc(fpr, tpr))
    
    print(f"\n--- {name} ---")
    print("Accuracy:", results[name])
    print(classification_report(y_test, y_pred))

plt.figure(figsize=(14, 10))

for i, (name, cm) in enumerate(conf_matrices.items(), 1):
    plt.subplot(2, 3, i)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(name)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

plt.tight_layout()
plt.show()
plt.figure(figsize=(8, 6))

for name, (fpr, tpr, roc_auc) in roc_data.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()
results_df = pd.DataFrame.from_dict(
    results, orient="index", columns=["Accuracy"]
).sort_values(by="Accuracy", ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x=results_df.index, y="Accuracy", data=results_df)
plt.xticks(rotation=45)
plt.title("Model Accuracy Comparison")
plt.show()

print(results_df)
