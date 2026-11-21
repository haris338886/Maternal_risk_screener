"""Train, calibrate, evaluate, and save the maternal risk classifier."""

import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, log_loss

from src.data_loader import load_dataset
from src.config import FEATURE_ORDER, MODEL_PATH


def train_and_save():
    df = load_dataset()

    # Use to_numpy() to bypass pyarrow extension dtypes (Python 3.12+ / pandas 2.2)
    X = df[FEATURE_ORDER].to_numpy(dtype="float64")
    y = df["RiskLevel"].astype(str).to_numpy()

    # 60 / 20 / 20 stratified split: train / calibration / test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=42
    )
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    base_rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        random_state=42,
        class_weight="balanced",
    )
    base_rf.fit(X_train, y_train)

    # Sigmoid calibration on held-out calibration set (small dataset → sigmoid > isotonic)
    calibrated = CalibratedClassifierCV(
        estimator=base_rf, method="sigmoid", cv="prefit"
    )
    calibrated.fit(X_cal, y_cal)

    # --- Evaluation ---
    y_pred = calibrated.predict(X_test)
    y_proba = calibrated.predict_proba(X_test)
    print("=== Test set performance ===")
    print(classification_report(y_test, y_pred, target_names=calibrated.classes_))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"Log loss: {log_loss(y_test, y_proba, labels=calibrated.classes_):.4f}")

    print("\n=== Error analysis by risk level ===")
    for cls in calibrated.classes_:
        mask = y_test == cls
        acc = (y_pred[mask] == cls).mean() if mask.sum() else float("nan")
        print(f"{cls:>10s}: accuracy={acc:.3f}  n={int(mask.sum())}")

    # --- Persist ---
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(calibrated, MODEL_PATH)
    print(f"\nModel saved → {MODEL_PATH}")


if __name__ == "__main__":
    train_and_save()