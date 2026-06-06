import pandas as pd
import numpy as np
import os
import json
import pickle
from datetime import datetime

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    RandomizedSearchCV
)

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

from xgboost import XGBClassifier

# ================================
# PATH SETUP
# ================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "churn_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ================================
# LOAD DATA
# ================================
data = pd.read_csv(DATA_PATH)

if "customer_id" in data.columns:
    data = data.drop("customer_id", axis=1)

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
# TRAIN TEST SPLIT
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ================================
# MODELS
# ================================
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        random_state=42
    ),

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced"
    ),

    "XGBoost": XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
}

# ================================
# RANDOM SEARCH PARAMS (XGBoost ONLY)
# ================================
param_dist = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.8, 1.0],
}

# ================================
# MODEL TRAINING + COMPARISON
# ================================
comparison_results = {}

best_model_name = None
best_score = -1

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("Training models...")

for name, clf in models.items():

    # ----------------------------
    # XGBOOST TUNING
    # ----------------------------
    if name == "XGBoost":
        search = RandomizedSearchCV(
            clf,
            param_distributions=param_dist,
            n_iter=5,
            scoring="roc_auc",
            cv=3,
            random_state=42,
            n_jobs=-1
        )

        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", search)
        ])
    else:
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])

    # ----------------------------
    # CROSS VALIDATION
    # ----------------------------
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")

    # Train final model
    pipe.fit(X_train, y_train)

    # If XGBoost, replace with best estimator
    if name == "XGBoost":
        best_estimator = pipe.named_steps["classifier"].best_estimator_
        pipe.named_steps["classifier"] = best_estimator

    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    roc = roc_auc_score(y_test, y_prob)
    recall = recall_score(y_test, y_pred)

    # ----------------------------
    # RECALL-AWARE SCORING
    # ----------------------------
    score = (0.7 * roc) + (0.3 * recall)

    comparison_results[name] = {
        "cv_roc_auc_mean": round(cv_scores.mean(), 4),
        "cv_roc_auc_std": round(cv_scores.std(), 4),
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc, 4)
    }

    if score > best_score:
        best_score = score
        best_model_name = name

print(f"Best Model Selected: {best_model_name}")

# ================================
# FINAL PIPELINE TRAINING
# ================================
final_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", models[best_model_name])
])

final_pipeline.fit(X_train, y_train)

y_pred_final = final_pipeline.predict(X_test)
y_prob_final = final_pipeline.predict_proba(X_test)[:, 1]

final_metrics = {
    "accuracy": round(accuracy_score(y_test, y_pred_final), 4),
    "precision": round(precision_score(y_test, y_pred_final), 4),
    "recall": round(recall_score(y_test, y_pred_final), 4),
    "f1": round(f1_score(y_test, y_pred_final), 4),
    "roc_auc": round(roc_auc_score(y_test, y_prob_final), 4),
    "best_model": best_model_name,
    "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "confusion_matrix": confusion_matrix(y_test, y_pred_final).tolist()
}

# ================================
# SAVE MODEL
# ================================
with open(os.path.join(MODEL_DIR, "pipeline.pkl"), "wb") as f:
    pickle.dump(final_pipeline, f)

# ================================
# SAVE METRICS
# ================================
with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
    json.dump(final_metrics, f, indent=2)

with open(os.path.join(MODEL_DIR, "model_comparison.json"), "w") as f:
    json.dump(comparison_results, f, indent=2)

print("Training complete!")
