import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import os

def main():
    # Load dataset from processed folder
    data_path = os.path.join("processed", "chatter_dataset.npz")
    data = np.load(data_path)
    X, y = data["X"], data["y"]

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42, stratify=y
    )

    # Train a Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate the classifier
    y_pred = clf.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["non-chatter", "chatter"]))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model and scaler to the processed folder
    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, os.path.join("models", "chatter_rf_model.joblib"))
    joblib.dump(scaler, os.path.join("models", "scaler.joblib"))
    print("Model and scaler saved to 'models/' directory.")

if __name__ == "__main__":
    main()
