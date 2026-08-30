import os

import pandas as pd
import requests
import streamlit as st

from src.config import COLS, FEATURE_COLS, TEST_FILE, WINDOW_SIZE


API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
)

st.title("⚙️ Predictive Maintenance Dashboard")
st.caption("RUL inference through the FastAPI model-serving service")

# -------------------------------------------------------------------
# Load C-MAPSS test data
# -------------------------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv(
        TEST_FILE,
        sep=r"\s+",
        engine="python",
        header=None,
        names=COLS,
    )
    return df


try:
    df = load_data()
except Exception as exc:
    st.error(f"Unable to load test data: {exc}")
    st.stop()


# -------------------------------------------------------------------
# Sidebar controls
# -------------------------------------------------------------------

engine_ids = sorted(df["engine_id"].unique())

st.sidebar.header("Equipment")

engine_id = st.sidebar.selectbox(
    "Select Engine",
    engine_ids,
)

engine_df = df[
    df["engine_id"] == engine_id
].sort_values("cycle")

if len(engine_df) < WINDOW_SIZE:
    st.error(
        f"Engine {engine_id} has fewer than "
        f"{WINDOW_SIZE} cycles."
    )
    st.stop()


window = engine_df[
    FEATURE_COLS
].tail(WINDOW_SIZE)

last_cycle = int(
    engine_df["cycle"].iloc[-1]
)

st.sidebar.metric(
    "Latest cycle",
    last_cycle,
)

st.sidebar.metric(
    "Inference window",
    f"{WINDOW_SIZE} cycles",
)


# -------------------------------------------------------------------
# Prediction
# -------------------------------------------------------------------

if st.button(
    "Predict Remaining Useful Life",
    type="primary",
):

    payload = {
        "sensor_window": window.values.tolist()
    }

    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=60,
        )

        response.raise_for_status()

        result = response.json()

        rul = float(
            result["predicted_rul"]
        )

        status = result["status"]

        st.session_state["prediction"] = {
            "rul": rul,
            "status": status,
        }

    except requests.RequestException as exc:

        st.error(
            f"Prediction service unavailable: {exc}"
        )


# -------------------------------------------------------------------
# Display prediction
# -------------------------------------------------------------------

prediction = st.session_state.get(
    "prediction"
)

if prediction:

    rul = prediction["rul"]
    status = prediction["status"]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Predicted RUL",
        f"{rul:.2f} cycles",
    )

    col2.metric(
        "Latest Cycle",
        last_cycle,
    )

    col3.metric(
        "Health Status",
        status.upper(),
    )

    if status == "critical":
        st.error(
            "CRITICAL: Remaining useful life is 10 cycles or less."
        )
    elif status == "warning":
        st.warning(
            "WARNING: Remaining useful life is between 10 and 30 cycles."
        )
    else:
        st.success(
            "HEALTHY: Remaining useful life is above 30 cycles."
        )


# -------------------------------------------------------------------
# Sensor data
# -------------------------------------------------------------------

st.subheader(
    f"Engine {engine_id} — Last {WINDOW_SIZE} Cycles"
)

display_df = window.copy()

display_df.insert(
    0,
    "cycle",
    engine_df["cycle"].tail(WINDOW_SIZE).values,
)

st.dataframe(
    display_df,
    use_container_width=True,
)

st.subheader("Sensor Trends")

selected_features = st.multiselect(
    "Select features",
    FEATURE_COLS,
    default=FEATURE_COLS[:3],
)

if selected_features:

    chart_df = display_df[
        ["cycle"] + selected_features
    ].set_index("cycle")

    st.line_chart(chart_df)


# -------------------------------------------------------------------
# Service status
# -------------------------------------------------------------------

with st.expander("API Service Status"):

    try:

        health = requests.get(
            f"{API_URL}/health",
            timeout=5,
        )

        ready = requests.get(
            f"{API_URL}/ready",
            timeout=5,
        )

        st.json(
            {
                "api_url": API_URL,
                "health": health.json(),
                "ready": ready.json(),
            }
        )

    except requests.RequestException as exc:

        st.error(
            f"API service is unavailable: {exc}"
        )
