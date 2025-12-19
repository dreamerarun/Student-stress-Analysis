# train_academic_model.py
"""
Generates a synthetic academic stress dataset and trains a RandomForest model.
Output:
 - academic_stress_dataset.csv
 - stress_predictor.pkl
Usage:
    python train_academic_model.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import random

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

OUT_CSV = "academic_stress_dataset.csv"
MODEL_FILE = "stress_predictor.pkl"

def synth_row():
    # Features:
    # sleep_hours (2-10), bp_sys (90-150), bp_dia(60-95), heart_rate(55-110)
    # study_hours (0-14), mood_score (1-10), gpa (0.0-4.0), procrastination (0-10), support_level (0-10)
    sleep = np.clip(np.random.normal(7, 1.5), 2, 10)
    bp_sys = int(np.clip(np.random.normal(120, 12), 90, 160))
    bp_dia = int(np.clip(np.random.normal(78, 8), 60, 100))
    hr = int(np.clip(np.random.normal(75, 10), 50, 120))
    study = max(0, min(14, np.random.normal(4, 2.5)))
    mood = int(np.clip(np.random.normal(6, 2), 1, 10))
    gpa = round(np.clip(np.random.normal(3.0, 0.5), 0.0, 4.0), 2)
    procrast = int(np.clip(np.random.normal(4, 3), 0, 10))
    support = int(np.clip(np.random.normal(6, 3), 0, 10))
    # academic anxiety (0-10)
    anxiety = int(np.clip(np.random.normal(5, 3), 0, 10))
    # generate base stress percentage by combining features
    # Higher study hours may increase stress if sleep low and procrast high
    score = 0
    score += (10 - sleep) * 6            # less sleep -> more stress
    score += max(0, study - 4) * 3       # more study may increase stress (but depends)
    score += (procrast) * 4
    score += (10 - mood) * 3
    score += (10 - support) * 2
    score += (anxiety) * 4
    # normalize roughly to 0-100
    pct = int(np.clip(score/ (6*10 + 3*10 + 4*10 + 3*10 + 2*10 + 4*10) * 100 * 1.1, 0, 100))
    # map pct to label
    if pct <= 30:
        label = "Low"
    elif pct <= 60:
        label = "Moderate"
    elif pct <= 80:
        label = "High"
    else:
        label = "Very High"
    return {
        "sleep_hours": round(float(sleep),2),
        "bp_sys": bp_sys,
        "bp_dia": bp_dia,
        "heart_rate": hr,
        "study_hours": round(float(study),2),
        "mood_score": mood,
        "gpa": gpa,
        "procrastination": procrast,
        "support_level": support,
        "academic_anxiety": anxiety,
        "stress_pct": pct,
        "stress_level": label
    }

def generate_dataset(n=800):
    rows = [synth_row() for _ in range(n)]
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved synthetic dataset to {OUT_CSV} (rows={len(df)})")
    return df

def train_model(df):
    features = ["sleep_hours","bp_sys","bp_dia","heart_rate","study_hours","mood_score","gpa","procrastination","support_level","academic_anxiety"]
    X = df[features]
    mapping = {"Low":0,"Moderate":1,"High":2,"Very High":3}
    y = df["stress_level"].map(mapping)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=SEED,stratify=y)
    clf = RandomForestClassifier(n_estimators=200, random_state=SEED)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test,preds))
    print(classification_report(y_test,preds))
    joblib.dump(clf, MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")
    return clf

if __name__ == "__main__":
    # if dataset exists, load; else create
    if os.path.exists(OUT_CSV):
        df = pd.read_csv(OUT_CSV)
        print(f"Loaded existing dataset {OUT_CSV} ({len(df)} rows)")
    else:
        df = generate_dataset(1000)
    clf = train_model(df)

