import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

data_dir = "spectrum_data"
classes = ["normal", "abnormal"]

X, y = [], []

for idx, cls in enumerate(classes):
    folder = os.path.join(data_dir, cls)
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        signal = np.loadtxt(path, delimiter=",", encoding="utf-8-sig")
        signal = signal.flatten()
        
        if len(signal) < 128:
            signal = np.tile(signal, int(np.ceil(128 / len(signal))))
            signal = signal[:128]
        
        plt.specgram(signal, NFFT=64, Fs=1, noverlap=32)
        plt.axis("off")
        plt.savefig("temp_spec.png", bbox_inches='tight', pad_inches=0)
        plt.close()

        from tensorflow.keras.preprocessing import image
        img = image.load_img("temp_spec.png", target_size=(64, 64))
        x = image.img_to_array(img)
        x = x / 255.0
        X.append(x)
        y.append(idx)

X = np.array(X)
y = np.array(y)

model = Sequential([
    Conv2D(16, (3,3), activation="relu", input_shape=(64,64,3)),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(32, activation="relu"),
    Dense(len(classes), activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X, y, epochs=10, batch_size=2)
model.save("tf_tasks/models/my_spectrum_model.keras")
print("Модель спектру створена!")
