import numpy as np
from model_predict import predict_power


# Take first test sequence
X_test = np.load("X_test.npy")

sample = X_test[0]

print("Input shape:", sample.shape)

prediction = predict_power(sample)

print("Predicted scaled power:", prediction)