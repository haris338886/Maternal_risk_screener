"""Project-wide constants: feature ranges, paths, thresholds."""

FEATURE_RANGES = {
    "Age":         (10, 70),      # years
    "SystolicBP":  (70, 180),     # mmHg
    "DiastolicBP": (40, 120),     # mmHg
    "BS":          (3.0, 20.0),   # mmol/L
    "BodyTemp":    (97.0, 103.0), # Fahrenheit
    "HeartRate":   (50, 110),     # bpm
}

FEATURE_ORDER = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]

RISK_LABELS = ["low risk", "mid risk", "high risk"]

# Traffic-light thresholds based on P(high risk)
HIGH_RISK_PROBA_THRESHOLD = 0.35
YELLOW_THRESHOLD = HIGH_RISK_PROBA_THRESHOLD * 0.5

MODEL_PATH = "models/calibrated_rf.joblib"
DATA_PATH = "data/maternal_health_risk.csv"