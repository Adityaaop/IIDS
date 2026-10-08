from pathlib import Path
import joblib
import numpy as np
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"


def test_scaler_artifact_exists_and_loads():
    """Ensure StandardScaler artifact loads and has correct feature dimension."""
    scaler_path = MODEL_DIR / "scaler.pkl"
    assert scaler_path.exists(), "scaler.pkl artifact missing"
    scaler = joblib.load(scaler_path)
    # NSL-KDD continuous and encoded feature set length
    dummy_features = np.zeros((1, scaler.n_features_in_))
    scaled = scaler.transform(dummy_features)
    assert scaled.shape == (1, scaler.n_features_in_)


def test_rf_model_artifact_exists_and_loads():
    """Ensure Random Forest classifier artifact loads and predicts probabilities."""
    rf_path = MODEL_DIR / "rf_model.pkl"
    assert rf_path.exists(), "rf_model.pkl artifact missing"
    rf = joblib.load(rf_path)
    assert hasattr(rf, "predict_proba")
    dummy_input = np.zeros((1, rf.n_features_in_))
    probs = rf.predict_proba(dummy_input)
    assert probs.shape[1] == 5  # 5 attack classes: Normal, DoS, Probe, R2L, U2R
