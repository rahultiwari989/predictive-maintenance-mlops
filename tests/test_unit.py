from pathlib import Path

from src.config import (
    FEATURE_COLS,
    WINDOW_SIZE,
    RUL_CAP,
    MODEL_FILE,
    SCALER_FILE,
    METADATA_FILE,
)


def test_window_configuration():
    assert WINDOW_SIZE == 30


def test_feature_configuration():
    assert len(FEATURE_COLS) == 16


def test_rul_cap_is_valid():
    assert RUL_CAP > 0


def test_model_artifacts_exist():
    assert Path(MODEL_FILE).exists()
    assert Path(SCALER_FILE).exists()
    assert Path(METADATA_FILE).exists()