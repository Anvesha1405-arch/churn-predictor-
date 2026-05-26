"""
Run this ONCE before launching the app:
    python setup_model.py
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
from pathlib import Path

BASE_DIR  = Path(__file__).parent
CSV_PATH  = BASE_DIR / "customer_churn_prediction_dataset.csv"
MODEL_OUT = BASE_DIR / "model.pkl"

print("="*55)
print("  ChurnScope — Model Setup")
print("="*55)

print("\n[1/4] Loading dataset...")
df = pd.read_csv(CSV_PATH)
print(f"      {len(df)} rows x {len(df.columns)} columns")

print("[2/4] Preprocessing...")
df = df.drop(columns=["customerID"], errors="ignore")
df["Churn"] = (df["Churn"] == "Yes").astype(int)

cat_cols = [c for c in df.columns if c != "Churn"
            and df[c].dtype not in [np.float64, np.int64, np.int32, np.float32]]

df_enc = pd.get_dummies(df, columns=cat_cols, dtype=int)

X = df_enc.drop(columns=["Churn"]).astype(float)
y = df_enc["Churn"]
feature_names = X.columns.to_numpy()
print(f"      Features: {len(feature_names)}, Classes: {y.value_counts().to_dict()}")

print("[3/4] Training (scaled Logistic Regression)...")
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s  = scaler.transform(X_te)

clf = LogisticRegression(max_iter=5000, C=1.0, random_state=42)
clf.fit(X_tr_s, y_tr)

acc = accuracy_score(y_te, clf.predict(X_te_s))
print(f"      Accuracy: {acc:.1%}")
print(classification_report(y_te, clf.predict(X_te_s), target_names=["Stay","Churn"]))

# Save as dict so app can access everything
bundle = {
    "scaler": scaler,
    "clf": clf,
    "feature_names": feature_names,
    "classes": clf.classes_,
}
print("[4/4] Saving model.pkl ...")
joblib.dump(bundle, MODEL_OUT)
print(f"      -> {MODEL_OUT}")
print("\n  Done! Run:  streamlit run app.py")
print("="*55)
