"""Inference wrapper: returns prediction + probabilities + traffic light."""

import joblib
import numpy as np
from src.config import (
    MODEL_PATH,
    FEATURE_ORDER,
    RISK_LABELS,
    HIGH_RISK_PROBA_THRESHOLD,
    YELLOW_THRESHOLD,
)

_model = None


def _load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_risk(features: dict) -> dict:
    """
    features: dict with keys FEATURE_ORDER
    Returns:
        {
          "risk_level": "low risk" | "mid risk" | "high risk",
          "probabilities": {label: float},
          "traffic_light": "green" | "yellow" | "red",
        }
    """
    model = _load_model()
    X = np.array([[features[k] for k in FEATURE_ORDER]], dtype="float64")
    proba = model.predict_proba(X)[0]
    classes = list(model.classes_)

    prob_map = {cls: float(p) for cls, p in zip(classes, proba)}
    for lbl in RISK_LABELS:
        prob_map.setdefault(lbl, 0.0)

    pred_label = max(prob_map, key=prob_map.get)
    high_p = prob_map["high risk"]

    if high_p >= HIGH_RISK_PROBA_THRESHOLD:
        light = "red"
    elif high_p >= YELLOW_THRESHOLD:
        light = "yellow"
    else:
        light = "green"

    return {
        "risk_level": pred_label,
        "probabilities": prob_map,
        "traffic_light": light,
    }