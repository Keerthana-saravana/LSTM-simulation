import numpy as np
import tensorflow as tf


# Load trained LSTM model
MODEL_PATH = "lstm_power_final.keras"

model = tf.keras.models.load_model(MODEL_PATH)


def predict_power(sequence):
    """
    Predict power usage from a 10-timestep sequence.

    Expected input shape:
    (10, 6)

    Returns:
    Predicted scaled power usage
    """

    sequence = np.array(sequence, dtype=np.float32)

    # Check input shape
    if sequence.shape != (10, 6):
        raise ValueError(
            f"Expected input shape (10, 6), "
            f"but received {sequence.shape}"
        )

    # Add batch dimension
    sequence = np.expand_dims(sequence, axis=0)

    # Prediction
    prediction = model.predict(sequence, verbose=0)

    return float(prediction[0][0])


def predict_from_npy(file_path):
    """
    Load a saved .npy sequence and predict power usage.
    """

    sequence = np.load(file_path)

    return predict_power(sequence)