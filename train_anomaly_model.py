import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Очікується файл normal_data.csv з нормальними прикладами
data = np.loadtxt("normal_data.csv", delimiter=",")
data = data.astype("float32")

data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

input_dim = data.shape[1]

model = models.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(32, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(input_dim, activation="linear")
])

model.compile(optimizer="adam", loss="mse")

model.fit(data, data, epochs=20, batch_size=16)

model.save("my_anomaly_model.keras")

print("Модель аномалій збережено як my_anomaly_model.keras")
