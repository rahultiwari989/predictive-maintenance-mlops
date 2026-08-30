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


def test_model_artifact_paths_are_configured():
    assert str(MODEL_FILE).endswith(".keras")
    assert str(SCALER_FILE).endswith(".pkl")
    assert str(METADATA_FILE).endswith(".json")