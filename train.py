import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import pickle
import json

# Load dataset
data = pd.read_csv("data/churn_data.csv")

# Drop ID column
if 'customer_id' in data.columns:
    data = data.drop('customer_id', axis=1)

# Split features & target
X = data.drop("churn", axis=1)
y = data["churn"]

# Column types
categorical_cols = ["country", "gender"]
yes_no_cols = ["credit_card", "active_member"]
numeric_cols = [col for col in X.columns if col not in categorical_cols + yes_no_cols]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(drop="first"), categorical_cols)
    ],
    remainder="passthrough"
)

# Pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
pipeline.fit(X_train, y_train)

# Save pipeline
with open("models/pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)

# --- SAVE DATA STATS FOR DRIFT ---
data_stats = {}

for col in numeric_cols:
    data_stats[col] = {
        "mean": float(X_train[col].mean()),
        "std": float(X_train[col].std())
    }

with open("models/data_stats.json", "w") as f:
    json.dump(data_stats, f)

print("✅ Training completed + data stats saved!")