from pathlib import Path

import pandas as pd
import requests

from src.config import (
    COLS,
    FEATURE_COLS,
    TEST_FILE,
    WINDOW_SIZE,
)


BASE_URL = "http://localhost:8000"

ENGINE_ID = 1


def load_test_data():

    df = pd.read_csv(
        TEST_FILE,
        sep=r"\s+",
        engine="python",
        header=None,
        names=COLS,
    )

    return df


def create_inference_window(df):

    engine_df = df[
        df["engine_id"] == ENGINE_ID
    ].sort_values("cycle")

    if engine_df.empty:
        raise ValueError(
            f"Engine {ENGINE_ID} not found."
        )

    if len(engine_df) < WINDOW_SIZE:
        raise ValueError(
            f"Engine {ENGINE_ID} has only "
            f"{len(engine_df)} cycles."
        )

    window = engine_df[
        FEATURE_COLS
    ].tail(WINDOW_SIZE)

    return window


def call_api(window):

    payload = {
        "sensor_window": window.values.tolist()
    }

    response = requests.post(
        f"{BASE_URL}/predict",
        json=payload,
        timeout=30,
    )

    print(
        f"HTTP status: {response.status_code}"
    )

    response.raise_for_status()

    return response.json()


def main():

    print("=" * 60)
    print("Predictive Maintenance - E2E API Test")
    print("=" * 60)

    print(
        f"Engine: {ENGINE_ID}"
    )

    print(
        f"Expected window: "
        f"{WINDOW_SIZE} x {len(FEATURE_COLS)}"
    )

    # ---------------------------------------------------------
    # 1. Load raw test data
    # ---------------------------------------------------------

    df = load_test_data()

    print(
        f"Raw test data shape: {df.shape}"
    )

    # ---------------------------------------------------------
    # 2. Extract final 30 cycles
    # ---------------------------------------------------------

    window = create_inference_window(df)

    print(
        f"Raw inference window: {window.shape}"
    )

    # ---------------------------------------------------------
    # 3. Send RAW data to API
    # ---------------------------------------------------------

    result = call_api(window)

    print("\nAPI response:")

    print(result)

    print("\n" + "=" * 60)

    print(
        "SUCCESS: "
        "Raw sensor data -> FastAPI -> scaler -> model -> RUL"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()