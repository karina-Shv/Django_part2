from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import numpy as np
from PIL import Image
import cv2
import librosa
import tempfile

# Імпорт моделей з окремих файлів
from tf_tasks.models.task2_video_model import predict_video
from tf_tasks.models.task3_audio_model import predict_audio
from tf_tasks.models.task4_spectrum_model import predict_spectrum
from tf_tasks.models.task5_anomaly_model import predict_anomaly

from tensorflow.keras.models import load_model

# Task 1 - Розпізнавання цифр
MODEL_PATH = "tf_tasks/models/mnist_model.h5"
mnist_model = load_model(MODEL_PATH)

def index(request):
    return render(request, "tf_tasks/index.html")

def task1(request):
    result = ""
    if request.method == "POST" and request.FILES.get("image"):
        img = Image.open(request.FILES["image"]).convert("L")  # Грейскейл
        img = img.resize((28, 28))
        img = np.array(img) / 255.0
        img = img.reshape(1, 28, 28)

        pred = mnist_model.predict(img)
        result = f"Розпізнана цифра: {np.argmax(pred)}"

    return render(request, "tf_tasks/task1.html", {"result": result})

# Task 2 - Відео
def task2(request):
    result = ""
    if request.method == "POST" and request.FILES.get("video"):
        video_file = request.FILES["video"]
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            for chunk in video_file.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name
        result = predict_video(tmp_path)
    return render(request, "tf_tasks/task2.html", {"result": result})

# Task 3 - Аудіо
def task3(request):
    result = ""
    if request.method == "POST" and request.FILES.get("audio"):
        audio_file = request.FILES["audio"]
        with open("temp_audio.wav", "wb") as f:
            for chunk in audio_file.chunks():
                f.write(chunk)
        result = predict_audio("temp_audio.wav")
    return render(request, "tf_tasks/task3.html", {"result": result})

# Task 4 - Спектр сигналу
def task4(request):
    result = ""
    if request.method == "POST" and request.FILES.get("audio"):
        audio_file = request.FILES["audio"]
        result = predict_spectrum(audio_file)
    return render(request, "tf_tasks/task4.html", {"result": result})


# Task 5 - Аномалії
def task5(request):
    result = ""
    if request.method == "POST" and request.FILES.get("file"):
        data_file = request.FILES["file"]
        result = predict_anomaly(data_file)
    return render(request, "tf_tasks/task5.html", {"result": result})
