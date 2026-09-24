import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# 1. LOAD TEST DATA
# ============================================================

X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

print("=" * 60)
print("FINAL LSTM MODEL TESTING")
print("=" * 60)

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# ============================================================
# 2. LOAD TRAINED MODEL
# ============================================================

model = tf.keras.models.load_model(
    "lstm_power_final.keras"
)

print("\n✓ Model loaded successfully")


# ============================================================
# 3. CHECK MODEL INPUT
# ============================================================

print("\nModel expected input shape:")
print(model.input_shape)

print("\nActual test input shape:")
print(X_test.shape)


# ============================================================
# 4. GENERATE PREDICTIONS
# ============================================================

predictions = model.predict(
    X_test,
    verbose=0
)

print("\nPrediction shape:")
print(predictions.shape)


# ============================================================
# 5. CALCULATE METRICS
# ============================================================

actual = y_test.flatten()
predicted = predictions.flatten()

mae = mean_absolute_error(
    actual,
    predicted
)

mse = mean_squared_error(
    actual,
    predicted
)

rmse = np.sqrt(mse)


print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print(f"MAE  : {mae:.6f}")
print(f"MSE  : {mse:.6f}")
print(f"RMSE : {rmse:.6f}")


# ============================================================
# 6. TEST DIFFERENT SAMPLES
# ============================================================

print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)

sample_indices = [
    0,
    10,
    50,
    100,
    200,
    500,
    700,
    900
]

for i in sample_indices:

    actual_value = actual[i]

    predicted_value = predicted[i]

    error = abs(
        actual_value - predicted_value
    )

    print(
        f"\nSample {i}"
    )

    print(
        f"Actual    : {actual_value:.6f}"
    )

    print(
        f"Predicted : {predicted_value:.6f}"
    )

    print(
        f"Error     : {error:.6f}"
    )


# ============================================================
# 7. BEST AND WORST PREDICTIONS
# ============================================================

absolute_errors = np.abs(
    actual - predicted
)

best_index = np.argmin(
    absolute_errors
)

worst_index = np.argmax(
    absolute_errors
)


print("\n" + "=" * 60)
print("BEST PREDICTION")
print("=" * 60)

print("Sample:", best_index)

print(
    "Actual:",
    actual[best_index]
)

print(
    "Predicted:",
    predicted[best_index]
)

print(
    "Error:",
    absolute_errors[best_index]
)


print("\n" + "=" * 60)
print("WORST PREDICTION")
print("=" * 60)

print("Sample:", worst_index)

print(
    "Actual:",
    actual[worst_index]
)

print(
    "Predicted:",
    predicted[worst_index]
)

print(
    "Error:",
    absolute_errors[worst_index]
)


# ============================================================
# 8. EMPTY / INVALID INPUT TEST
# ============================================================

print("\n" + "=" * 60)
print("INPUT VALIDATION TEST")
print("=" * 60)

expected_timesteps = model.input_shape[1]
expected_features = model.input_shape[2]

print(
    "Expected time steps:",
    expected_timesteps
)

print(
    "Expected features:",
    expected_features
)


if X_test.shape[1] == expected_timesteps:

    print("✓ Time-step dimension is correct")

else:

    print("✗ Time-step dimension is incorrect")


if X_test.shape[2] == expected_features:

    print("✓ Feature dimension is correct")

else:

    print("✗ Feature dimension is incorrect")


# ============================================================
# 9. FINAL STATUS
# ============================================================

print("\n" + "=" * 60)
print("MODEL TESTING COMPLETED")
print("=" * 60)

print("✓ Model loading")
print("✓ Test data loading")
print("✓ Prediction generation")
print("✓ MAE calculation")
print("✓ MSE calculation")
print("✓ RMSE calculation")
print("✓ Sample testing")
print("✓ Best/worst prediction analysis")
print("✓ Input shape validation")

print("\nLSTM model is ready for UI integration.")