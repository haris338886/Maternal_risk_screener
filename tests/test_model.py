"""Sanity checks for the persisted model."""

import os
import joblib
import numpy as np
import pytest

from src.config import MODEL_PATH


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="model not trained yet")
def test_model_loaded_and_predicts():
    model = joblib.load(MODEL_PATH)
    assert hasattr(model, "predict_proba")

    X = np.array([[25, 120, 80, 7.0, 98.0, 75]], dtype="float64")
    proba = model.predict_proba(X)
    assert proba.shape == (1, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)