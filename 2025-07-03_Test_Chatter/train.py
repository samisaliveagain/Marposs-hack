import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib

def main():
    # Load dataset
    data = np.load("chatter_dataset.npz")
    X, y = data["X"], data["y"]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42, stratify=y
    )

    # Initialize and train classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    print("✅ Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["non-chatter", "chatter"]))
    print("🔍 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save the model and scaler
    joblib.dump(clf, "chatter_rf_model.joblib")
    joblib.dump(scaler, "scaler.joblib")
    print("✅ Model and scaler saved.")

if __name__ == "__main__":
    main()
