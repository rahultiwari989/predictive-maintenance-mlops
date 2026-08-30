import json
import joblib
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error

from src.config import ARTIFACT_DIR, MODEL_FILE, SCALER_FILE, METADATA_FILE, TRAIN_FILE, WINDOW_SIZE, RUL_CAP
from src.model import build_model
from src.preprocessing import load_cmapss, add_rul, get_feature_cols, split_by_engine, fit_scaler, apply_scaler, create_sequences


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    df = add_rul(load_cmapss(TRAIN_FILE))
    train_df, val_df = split_by_engine(df)
    feature_cols = get_feature_cols(df)

    # Production-safe: fit preprocessing only on training engines.
    scaler = fit_scaler(train_df, feature_cols)
    train_df = apply_scaler(train_df, scaler, feature_cols)
    val_df = apply_scaler(val_df, scaler, feature_cols)

    X_tr, y_tr = create_sequences(train_df, feature_cols, WINDOW_SIZE)
    X_val, y_val = create_sequences(val_df, feature_cols, WINDOW_SIZE)

    print(f"Training shape:   {X_tr.shape}")
    print(f"Validation shape: {X_val.shape}")
    print(f"Features: {len(feature_cols)}")

    model = build_model(WINDOW_SIZE, len(feature_cols))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        patience=3,
        restore_best_weights=True,
    )

    model.fit(
        X_tr,
        y_tr,
        validation_data=(X_val, y_val),
        epochs=40,
        batch_size=64,
        callbacks=[early_stop],
        verbose=1,
    )

    pred = model.predict(X_val, verbose=0).reshape(-1)
    rmse = float(np.sqrt(mean_squared_error(y_val, pred)))
    print(f"Validation RMSE: {rmse:.4f}")

    model.save(MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)

    metadata = {
        "model_type": "CNN-LSTM baseline (notebook-compatible)",
        "window_size": WINDOW_SIZE,
        "rul_cap": RUL_CAP,
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "validation_rmse": rmse,
    }
    METADATA_FILE.write_text(json.dumps(metadata, indent=2))
    print(f"Saved model:  {MODEL_FILE}")
    print(f"Saved scaler: {SCALER_FILE}")
    print(f"Saved metadata: {METADATA_FILE}")


if __name__ == "__main__":
    main()
