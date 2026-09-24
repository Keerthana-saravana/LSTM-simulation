# ============================================================
# LSTM POWER USAGE PREDICTION
# Member 2 - Day 2
# ============================================================

import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

SEED = 42

np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

print("=" * 60)
print("LSTM POWER USAGE PREDICTION")
print("=" * 60)


# ============================================================
# 2. CONFIGURATION
# ============================================================

DATA_FILE = "lstm_power_dataset.csv"

X_TRAIN_FILE = "X_train.npy"
X_TEST_FILE = "X_test.npy"
Y_TRAIN_FILE = "y_train.npy"
Y_TEST_FILE = "y_test.npy"

MODEL_FILE = "lstm_power_final.keras"

RESULTS_FILE = "prediction_results.csv"

TRAINING_GRAPH = "training_validation_loss.png"
PREDICTION_GRAPH = "actual_vs_predicted.png"
ERROR_GRAPH = "error_distribution.png"


# ============================================================
# 3. CHECK REQUIRED FILES
# ============================================================

required_files = [
    DATA_FILE,
    X_TRAIN_FILE,
    X_TEST_FILE,
    Y_TRAIN_FILE,
    Y_TEST_FILE
]

print("\nChecking required files...")

for file in required_files:

    if not os.path.exists(file):

        raise FileNotFoundError(
            f"\nFile not found: {file}\n"
            f"Make sure all required files are in the project folder."
        )

    print("✓", file)


# ============================================================
# 4. LOAD ORIGINAL CSV
# ============================================================

print("\n" + "=" * 60)
print("LOADING DATASET")
print("=" * 60)

data = pd.read_csv(DATA_FILE)

print("Dataset shape:", data.shape)

print("\nDataset columns:")
for column in data.columns:
    print(" -", column)


# ============================================================
# 5. LOAD PREPARED LSTM ARRAYS
# ============================================================

print("\n" + "=" * 60)
print("LOADING PREPARED LSTM DATA")
print("=" * 60)

X_train = np.load(X_TRAIN_FILE)
X_test = np.load(X_TEST_FILE)

y_train = np.load(Y_TRAIN_FILE)
y_test = np.load(Y_TEST_FILE)


print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)

print("y_train shape:", y_train.shape)
print("y_test shape :", y_test.shape)


# ============================================================
# 6. BASIC VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("CHECKING DATA")
print("=" * 60)

if X_train.ndim != 3:
    raise ValueError(
        f"X_train must have 3 dimensions "
        f"(samples, timesteps, features). "
        f"Found: {X_train.ndim}"
    )

if X_test.ndim != 3:
    raise ValueError(
        f"X_test must have 3 dimensions. "
        f"Found: {X_test.ndim}"
    )

if y_train.ndim != 2:
    raise ValueError(
        f"y_train must have 2 dimensions. "
        f"Found: {y_train.ndim}"
    )

if y_test.ndim != 2:
    raise ValueError(
        f"y_test must have 2 dimensions. "
        f"Found: {y_test.ndim}"
    )


if X_train.shape[0] != y_train.shape[0]:

    raise ValueError(
        "Number of X_train samples does not match y_train."
    )


if X_test.shape[0] != y_test.shape[0]:

    raise ValueError(
        "Number of X_test samples does not match y_test."
    )


print("✓ X_train dimensions are correct")
print("✓ X_test dimensions are correct")
print("✓ y_train dimensions are correct")
print("✓ y_test dimensions are correct")


# ============================================================
# 7. CHECK FOR NaN / INFINITY
# ============================================================

print("\nChecking for invalid values...")

print(
    "X_train NaN:",
    np.isnan(X_train).sum()
)

print(
    "X_test NaN:",
    np.isnan(X_test).sum()
)

print(
    "y_train NaN:",
    np.isnan(y_train).sum()
)

print(
    "y_test NaN:",
    np.isnan(y_test).sum()
)

if not np.isfinite(X_train).all():
    raise ValueError("X_train contains NaN or infinite values.")

if not np.isfinite(X_test).all():
    raise ValueError("X_test contains NaN or infinite values.")

if not np.isfinite(y_train).all():
    raise ValueError("y_train contains NaN or infinite values.")

if not np.isfinite(y_test).all():
    raise ValueError("y_test contains NaN or infinite values.")

print("✓ No NaN or infinite values found")


# ============================================================
# 8. CREATE TRAINING + VALIDATION SPLIT
# ============================================================

print("\n" + "=" * 60)
print("CREATING TRAIN / VALIDATION SPLIT")
print("=" * 60)

# IMPORTANT:
# Do not use the final test set for validation.
# Since this is time-series data, do not shuffle.

validation_ratio = 0.20

split_index = int(
    len(X_train) * (1 - validation_ratio)
)

X_train_actual = X_train[:split_index]
y_train_actual = y_train[:split_index]

X_val = X_train[split_index:]
y_val = y_train[split_index:]


print("Actual training:")
print("X:", X_train_actual.shape)
print("y:", y_train_actual.shape)

print("\nValidation:")
print("X:", X_val.shape)
print("y:", y_val.shape)

print("\nFinal untouched test:")
print("X:", X_test.shape)
print("y:", y_test.shape)


# ============================================================
# 9. MODEL CONFIGURATION
# ============================================================

TIME_STEPS = X_train.shape[1]
NUM_FEATURES = X_train.shape[2]

print("\n" + "=" * 60)
print("MODEL INPUT")
print("=" * 60)

print("Time steps :", TIME_STEPS)
print("Features   :", NUM_FEATURES)

print(
    "\nInput shape expected by LSTM:",
    f"(batch_size, {TIME_STEPS}, {NUM_FEATURES})"
)


# ============================================================
# 10. BUILD LSTM MODEL
# ============================================================

print("\n" + "=" * 60)
print("BUILDING LSTM MODEL")
print("=" * 60)

model = Sequential([

    Input(
        shape=(TIME_STEPS, NUM_FEATURES)
    ),

    LSTM(
        64,
        return_sequences=True
    ),

    Dropout(0.20),

    LSTM(
        32,
        return_sequences=False
    ),

    Dropout(0.20),

    Dense(
        32,
        activation="relu"
    ),

    Dense(
        1
    )
])


# ============================================================
# 11. COMPILE MODEL
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="mse",

    metrics=[
        tf.keras.metrics.MeanAbsoluteError(
            name="mae"
        )
    ]
)


print("\nModel summary:")
model.summary()


# ============================================================
# 12. CALLBACKS
# ============================================================

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=7,

    min_delta=0.00001,

    restore_best_weights=True,

    verbose=1
)


reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=3,

    min_lr=0.00001,

    verbose=1
)


checkpoint = ModelCheckpoint(

    MODEL_FILE,

    monitor="val_loss",

    save_best_only=True,

    mode="min",

    verbose=1
)


# ============================================================
# 13. TRAIN MODEL
# ============================================================

print("\n" + "=" * 60)
print("STARTING LSTM TRAINING")
print("=" * 60)

history = model.fit(

    X_train_actual,

    y_train_actual,

    epochs=50,

    batch_size=32,

    validation_data=(
        X_val,
        y_val
    ),

    callbacks=[
        early_stopping,
        reduce_lr,
        checkpoint
    ],

    shuffle=False,

    verbose=1
)


# ============================================================
# 14. TRAINING SUMMARY
# ============================================================

epochs_completed = len(
    history.history["loss"]
)

best_epoch = (
    np.argmin(
        history.history["val_loss"]
    ) + 1
)

best_val_loss = min(
    history.history["val_loss"]
)

print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

print(
    "Epochs completed:",
    epochs_completed
)

print(
    "Best epoch:",
    best_epoch
)

print(
    "Best validation loss:",
    best_val_loss
)


# ============================================================
# 15. LOAD BEST MODEL
# ============================================================

print("\nLoading best saved model...")

model = tf.keras.models.load_model(
    MODEL_FILE
)

print("✓ Best model loaded")


# ============================================================
# 16. FINAL TEST EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("FINAL TEST EVALUATION")
print("=" * 60)

test_loss, test_mae = model.evaluate(

    X_test,

    y_test,

    verbose=0
)


print("Test MSE :", test_loss)
print("Test MAE :", test_mae)


# ============================================================
# 17. GENERATE PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

predictions = model.predict(

    X_test,

    verbose=0
)


print(
    "Prediction shape:",
    predictions.shape
)


# ============================================================
# 18. CALCULATE FINAL METRICS
# ============================================================

mae = mean_absolute_error(

    y_test.flatten(),

    predictions.flatten()
)


mse = mean_squared_error(

    y_test.flatten(),

    predictions.flatten()
)


rmse = np.sqrt(mse)


print("\n" + "=" * 60)
print("FINAL TEST METRICS")
print("=" * 60)

print(
    f"MAE  : {mae:.8f}"
)

print(
    f"MSE  : {mse:.8f}"
)

print(
    f"RMSE : {rmse:.8f}"
)


# ============================================================
# 19. CREATE RESULTS TABLE
# ============================================================

results = pd.DataFrame({

    "Actual_Scaled":
        y_test.flatten(),

    "Predicted_Scaled":
        predictions.flatten()
})


results["Error"] = (

    results["Actual_Scaled"]

    -

    results["Predicted_Scaled"]
)


results["Absolute_Error"] = (

    results["Error"].abs()

)


results["Squared_Error"] = (

    results["Error"] ** 2

)


print("\n" + "=" * 60)
print("FIRST 20 PREDICTIONS")
print("=" * 60)

print(
    results.head(20).to_string(
        index=False
    )
)


# ============================================================
# 20. SAVE RESULTS
# ============================================================

results.to_csv(

    RESULTS_FILE,

    index=False
)


print(
    f"\n✓ Results saved to: {RESULTS_FILE}"
)


# ============================================================
# 21. TRAINING / VALIDATION LOSS GRAPH
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(

    history.history["loss"],

    label="Training Loss",

    linewidth=2
)


plt.plot(

    history.history["val_loss"],

    label="Validation Loss",

    linewidth=2
)


plt.axvline(

    best_epoch - 1,

    linestyle="--",

    label=f"Best Epoch ({best_epoch})"
)


plt.xlabel("Epoch")

plt.ylabel("MSE Loss")

plt.title(
    "LSTM Training vs Validation Loss"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


plt.savefig(

    TRAINING_GRAPH,

    dpi=300
)


plt.show()


# ============================================================
# 22. ACTUAL VS PREDICTED
# ============================================================

plt.figure(
    figsize=(14, 6)
)

plt.plot(

    y_test.flatten(),

    label="Actual",

    linewidth=1.5
)


plt.plot(

    predictions.flatten(),

    label="Predicted",

    linewidth=1.5
)


plt.xlabel(
    "Test Sample"
)

plt.ylabel(
    "Scaled Power Usage"
)

plt.title(
    "Actual vs Predicted Power Usage"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


plt.savefig(

    PREDICTION_GRAPH,

    dpi=300
)


plt.show()


# ============================================================
# 23. FIRST 100 TEST SAMPLES
# ============================================================

plt.figure(
    figsize=(14, 6)
)

plt.plot(

    y_test[:100].flatten(),

    label="Actual",

    linewidth=2
)


plt.plot(

    predictions[:100].flatten(),

    label="Predicted",

    linewidth=2
)


plt.xlabel(
    "Test Sample"
)

plt.ylabel(
    "Scaled Power Usage"
)

plt.title(
    "Actual vs Predicted - First 100 Test Samples"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.show()


# ============================================================
# 24. ERROR DISTRIBUTION
# ============================================================

errors = (

    y_test.flatten()

    -

    predictions.flatten()

)


plt.figure(
    figsize=(10, 6)
)

plt.hist(

    errors,

    bins=30
)

plt.xlabel(
    "Prediction Error"
)

plt.ylabel(
    "Frequency"
)

plt.title(
    "Prediction Error Distribution"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()


plt.savefig(

    ERROR_GRAPH,

    dpi=300
)


plt.show()


# ============================================================
# 25. ERROR STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("ERROR ANALYSIS")
print("=" * 60)

print(
    "Mean Error:",
    np.mean(errors)
)

print(
    "Mean Absolute Error:",
    np.mean(np.abs(errors))
)

print(
    "Maximum Absolute Error:",
    np.max(np.abs(errors))
)

print(
    "Minimum Absolute Error:",
    np.min(np.abs(errors))
)


# ============================================================
# 26. SAMPLE PREDICTIONS
# ============================================================

print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)

sample_indices = [
    0,
    10,
    50,
    100,
    200
]


for index in sample_indices:

    actual = y_test[index][0]

    predicted = predictions[index][0]

    error = abs(
        actual - predicted
    )

    print(
        f"\nSample {index}"
    )

    print(
        f"Actual    : {actual:.6f}"
    )

    print(
        f"Predicted : {predicted:.6f}"
    )

    print(
        f"Abs Error : {error:.6f}"
    )


# ============================================================
# 27. SINGLE SEQUENCE TEST
# ============================================================

print("\n" + "=" * 60)
print("SINGLE SEQUENCE TEST")
print("=" * 60)

single_sample = X_test[0:1]

print(
    "Input shape:",
    single_sample.shape
)


single_prediction = model.predict(

    single_sample,

    verbose=0
)


print(
    "Prediction:",
    single_prediction[0][0]
)

print(
    "Actual:",
    y_test[0][0]
)


# ============================================================
# 28. SAVE MODEL AGAIN
# ============================================================

model.save(
    MODEL_FILE
)

print(
    f"\n✓ Final model saved: {MODEL_FILE}"
)


# ============================================================
# 29. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("DAY 2 LSTM TRAINING COMPLETE")
print("=" * 60)

print(
    f"Input shape       : ({TIME_STEPS}, {NUM_FEATURES})"
)

print(
    f"Training samples  : {len(X_train_actual)}"
)

print(
    f"Validation samples: {len(X_val)}"
)

print(
    f"Test samples      : {len(X_test)}"
)

print(
    f"Best epoch        : {best_epoch}"
)

print(
    f"MAE               : {mae:.8f}"
)

print(
    f"MSE               : {mse:.8f}"
)

print(
    f"RMSE              : {rmse:.8f}"
)

print("\nGenerated files:")

print(
    "✓", MODEL_FILE
)

print(
    "✓", RESULTS_FILE
)

print(
    "✓", TRAINING_GRAPH
)

print(
    "✓", PREDICTION_GRAPH
)

print(
    "✓", ERROR_GRAPH
)

print("=" * 60)