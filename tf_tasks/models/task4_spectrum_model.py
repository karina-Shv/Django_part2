# tf_tasks/models/task4_spectrum_model.py
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import io

spectrum_model = load_model("tf_tasks/models/my_spectrum_model.keras")

def predict_spectrum(uploaded_file):
    """
    uploaded_file - InMemoryUploadedFile з Django
    Повертає клас сигналу: normal або abnormal
    """
    from scipy.io import wavfile

    uploaded_file.seek(0)
    sr, signal_array = wavfile.read(uploaded_file)
    signal_array = signal_array.astype("float32") / 32767.0

    if signal_array.ndim > 1:  # стерео
        signal_array = signal_array[:,0]

    # Генерація спектру у буфер
    fig, ax = plt.subplots()
    ax.specgram(signal_array, NFFT=1024, Fs=sr, noverlap=512)
    ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    img = image.load_img(buf, target_size=(64,64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0

    labels = {0:"normal", 1:"abnormal"}
    preds = spectrum_model.predict(x)
    label = np.argmax(preds, axis=1)[0]

    return f"Розпізнаний клас: {labels[label]}"
