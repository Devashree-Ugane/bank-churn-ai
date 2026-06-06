import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import pickle
import json
from datetime import datetime

# ================================
# LOAD DATASET
# ================================
data = pd.read_csv("data/churn_data.csv")

if 'customer_id' in data.columns:
    data = data.drop('customer_id', axis=1)

X = data.drop("churn", axis=1)
y = data["churn"]

# ================================
# COLUMN TYPES
# ================================
categorical_cols = ["country", "gender"]
yes_no_cols = ["credit_card", "active_member"]
numeric_cols = [col for col in X.columns if col not in categorical_cols + yes_no_cols]

# ================================
# PREPROCESSOR
# ================================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(drop="first"), categorical_cols)
    ],
    remainder="passthrough"
)

# ================================
# TRAIN-TEST SPLIT
# ================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ================================
# MODEL COMPARISON
# ================================
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
}

comparison_results = {}
best_model_name = None
best_roc = -1

print("Training and comparing models...")

for name, clf in models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    roc = roc_auc_score(y_test, y_prob)
    comparison_results[name] = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc, 4)
    }

    if roc > best_roc:
        best_roc = roc
        best_model_name = name

print(f"Best model: {best_model_name} (ROC-AUC: {best_roc:.4f})")

# ================================
# TRAIN BEST MODEL AS FINAL PIPELINE
# ================================
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", models[best_model_name])
])
pipeline.fit(X_train, y_train)

y_pred_final = pipeline.predict(X_test)
y_prob_final = pipeline.predict_proba(X_test)[:, 1]

final_metrics = {
    "accuracy": round(accuracy_score(y_test, y_pred_final), 4),
    "precision": round(precision_score(y_test, y_pred_final), 4),
    "recall": round(recall_score(y_test, y_pred_final), 4),
    "f1": round(f1_score(y_test, y_pred_final), 4),
    "roc_auc": round(roc_auc_score(y_test, y_prob_final), 4),
    "best_model": best_model_name,
    "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "train_size": len(X_train),
    "test_size": len(X_test)
}

cm = confusion_matrix(y_test, y_pred_final)
final_metrics["confusion_matrix"] = cm.tolist()

# ================================
# SAVE PIPELINE
# ================================
with open("models/pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)

# ================================
# SAVE ALL JSON FILES
# ================================

# Data stats for drift detection
data_stats = {}
for col in numeric_cols:
    data_stats[col] = {
        "mean": float(X_train[col].mean()),
        "std": float(X_train[col].std()),
        "min": float(X_train[col].min()),
        "max": float(X_train[col].max()),
        "p5": float(np.percentile(X_train[col], 5)),
        "p95": float(np.percentile(X_train[col], 95))
    }

with open("models/data_stats.json", "w") as f:
    json.dump(data_stats, f, indent=2)

# Model metrics
with open("models/metrics.json", "w") as f:
    json.dump(final_metrics, f, indent=2)

# Model comparison
with open("models/model_comparison.json", "w") as f:
    json.dump(comparison_results, f, indent=2)

# Retraining history (append each run)
history_path = "models/retrain_history.json"
try:
    with open(history_path, "r") as f:
        history = json.load(f)
except:
    history = []

history.append({
    "timestamp": final_metrics["trained_at"],
    "best_model": best_model_name,
    "roc_auc": final_metrics["roc_auc"],
    "f1": final_metrics["f1"],
    "accuracy": final_metrics["accuracy"]
})

with open(history_path, "w") as f:
    json.dump(history, f, indent=2)

print("✅ Training completed!")
print(f"   Accuracy : {final_metrics['accuracy']}")
print(f"   Precision: {final_metrics['precision']}")
print(f"   Recall   : {final_metrics['recall']}")
print(f"   F1 Score : {final_metrics['f1']}")
print(f"   ROC-AUC  : {final_metrics['roc_auc']}")
print("✅ All model files saved!")
