import json
import joblib
import numpy as np
import tensorflow as tf

from src.config import MODEL_FILE, SCALER_FILE, METADATA_FILE, WINDOW_SIZE, RUL_CAP


class RULPredictor:
    def __init__(self):
        self.model = tf.keras.models.load_model(MODEL_FILE)
        self.scaler = joblib.load(SCALER_FILE)
        self.metadata = json.loads(METADATA_FILE.read_text())
        self.feature_cols = self.metadata["feature_cols"]

    def predict(self, sensor_window):
        data = np.asarray(sensor_window, dtype=np.float32)
        expected = len(self.feature_cols)
        if data.shape != (WINDOW_SIZE, expected):
            raise ValueError(
                f"Expected input shape ({WINDOW_SIZE}, {expected}), got {data.shape}"
            )
        data = self.scaler.transform(data)
        data = data.reshape(1, WINDOW_SIZE, expected)
        pred = float(self.model.predict(data, verbose=0)[0, 0])
        return float(np.clip(pred, 0, RUL_CAP))
