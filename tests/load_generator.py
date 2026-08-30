import time
import requests
import pandas as pd

from src.config import (
    COLS,
    FEATURE_COLS,
    TEST_FILE,
    WINDOW_SIZE,
)


API_URL = "http://localhost:8001"

ENGINE_ID = 1

REQUESTS_PER_SECOND = 10

DURATION_SECONDS = 120


def load_window():

    df = pd.read_csv(
        TEST_FILE,
        sep=r"\s+",
        engine="python",
        header=None,
        names=COLS,
    )

    engine_df = df[
        df["engine_id"] == ENGINE_ID
    ].sort_values("cycle")

    window = engine_df[
        FEATURE_COLS
    ].tail(WINDOW_SIZE)

    return window.values.tolist()


def main():

    sensor_window = load_window()

    payload = {
        "sensor_window": sensor_window
    }

    end_time = time.time() + DURATION_SECONDS

    requests_sent = 0

    while time.time() < end_time:

        start = time.time()

        for _ in range(REQUESTS_PER_SECOND):

            try:

                response = requests.post(
                    f"{API_URL}/predict",
                    json=payload,
                    timeout=30,
                )

                requests_sent += 1

            except requests.RequestException:
                pass

        elapsed = time.time() - start

        sleep_time = max(
            0,
            1 - elapsed
        )

        time.sleep(sleep_time)

    print(
        f"Requests sent: {requests_sent}"
    )


if __name__ == "__main__":
    main()