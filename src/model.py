import tensorflow as tf


def build_model(window_size, n_features):
    inputs = tf.keras.Input(shape=(window_size, n_features))

    # Kept faithful to the current notebook baseline.
    # NOTE: the Conv1D/MaxPooling branch is currently not connected to the LSTM.
    x = tf.keras.layers.Conv1D(32, kernel_size=3, activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling1D(2)(x)

    x = tf.keras.layers.LSTM(
        64,
        return_sequences=True,
        dropout=0.3,
        recurrent_dropout=0.2,
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    )(inputs)
    x = tf.keras.layers.LSTM(
        32,
        dropout=0.3,
        recurrent_dropout=0.2,
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name="embedding",
    )(x)
    output = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(inputs, output)
