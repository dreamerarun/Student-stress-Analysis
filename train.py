# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv("stress_dataset.csv")

# Encode categorical stress labels
df["stress_level"] = df["stress_level"].map({
    "Low": 0, "Moderate": 1, "High": 2, "Very High": 3
})

X = df[["sleep_hours", "bp_sys", "bp_dia", "heart_rate", "study_hours", "mood_score"]]
y = df["stress_level"]

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, pred):.2f}")

# Save model
joblib.dump(model, "stress_predictor.pkl")
print("✅ Model saved as stress_predictor.pkl")

