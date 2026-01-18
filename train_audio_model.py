import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models

DATA_DIR = "audio_data"
CLASSES = sorted(os.listdir(DATA_DIR))
N_CLASSES = len(CLASSES)

X, y = [], []

for label, class_name in enumerate(CLASSES):
    class_dir = os.path.join(DATA_DIR, class_name)
    for file in os.listdir(class_dir):
        if file.endswith(".wav"):
            path = os.path.join(class_dir, file)
            y_audio, sr = librosa.load(path, sr=22050, duration=5.0)
            
            mel = librosa.feature.melspectrogram(
                y=y_audio, sr=sr, n_mels=128, fmax=8000
            )
            mel_db = librosa.power_to_db(mel, ref=np.max)

            # Гарантуємо однакову форму (128, 128)
            if mel_db.shape[1] < 128:
                pad_width = 128 - mel_db.shape[1]
                mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode="constant")
            else:
                mel_db = mel_db[:, :128]

            X.append(mel_db)
            y.append(label)

X = np.array(X)[..., np.newaxis]  # (N, 128, 128, 1)
X = X / 255.0
y = np.array(y)

print("Форма X:", X.shape)
print("Форма y:", y.shape)

model = models.Sequential([
    layers.Input(shape=(128, 128, 1)),
    layers.Conv2D(16, (3, 3), activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(N_CLASSES, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(X, y, epochs=10, batch_size=8)
model.save("my_audio_model.keras")

print("my_audio_model.keras створено успішно")
