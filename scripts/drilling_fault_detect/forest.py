import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# === Step 1: Load the data ===
tool = "TN23"  # Change if needed
path = f"split_cnc_data/{tool}"

X_train = np.load(f"{path}/X_train.npy")
y_train = np.load(f"{path}/y_train.npy")
X_test = np.load(f"{path}/X_test.npy")
y_test = np.load(f"{path}/y_test.npy")

# === Step 2: Flatten time-series for Random Forest ===
N, F, T = X_train.shape
X_train_flat = X_train.reshape(N, F * T)
X_test_flat = X_test.reshape(X_test.shape[0], F * T)

# === Step 3: Load feature names ===
with open(f"{path}/feature_names.json") as f:
    raw_feature_names = json.load(f)

# Build feature names like: SIG1_t0, SIG1_t1, ..., SIG2_t0, ...
feature_names = []
for feat in raw_feature_names:
    for t in range(T):
        feature_names.append(f"{feat}_t{t}")

# === Step 4: Train Random Forest ===
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train_flat, y_train)

# === Step 5: Evaluate ===
y_pred = clf.predict(X_test_flat)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["NIO", "IO"]))

# === Step 6: Plot top 15 features ===
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

top_n = 15
top_indices = indices[:top_n]
top_features = [feature_names[i] for i in top_indices]
top_importances = [importances[i] for i in top_indices]

plt.figure(figsize=(10, 6))
plt.barh(range(top_n), top_importances[::-1])
plt.yticks(range(top_n), top_features[::-1])
plt.xlabel("Feature Importance")
plt.title(f"Top {top_n} Features by Random Forest")
plt.tight_layout()
plt.show()
