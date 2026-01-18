import numpy as np
import librosa
import os
from tensorflow.keras.models import load_model

MODEL_PATH = os.path.join(os.path.dirname(__file__), "my_audio_model.keras")
audio_model = load_model(MODEL_PATH)

def preprocess_audio(audio_path, n_mels=128, fixed_len=128):
    y, sr = librosa.load(audio_path, sr=None, duration=5.0)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)

    # Змінюємо ширину спектрограми до fixed_len
    if S_db.shape[1] < fixed_len:
        pad_width = fixed_len - S_db.shape[1]
        S_db = np.pad(S_db, ((0,0),(0,pad_width)), mode='constant')
    elif S_db.shape[1] > fixed_len:
        S_db = S_db[:, :fixed_len]

    S_db = np.expand_dims(S_db, axis=(0,-1))  # (1, 128, 128, 1)
    return S_db

labels = {0: "Кіт", 1: "Собака"}

def predict_audio(audio_path):
    S_db = preprocess_audio(audio_path)
    preds = audio_model.predict(S_db)
    label = np.argmax(preds, axis=1)[0]
    return f"Розпізнаний клас: {labels[label]}"
