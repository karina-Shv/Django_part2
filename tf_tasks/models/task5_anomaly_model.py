import numpy as np
from tensorflow.keras.models import load_model

anomaly_model = load_model("tf_tasks/models/my_anomaly_model.keras")

THRESHOLD = 0.05  

def predict_anomaly(data_file):
    """
    Приймає CSV-файл з числовими даними (рядки = зразки, колонки = ознаки),
    повертає результат детекції аномалій.
    """
    try:
        data = np.loadtxt(data_file, delimiter=",")
    except Exception:
        raise ValueError("Файл має бути CSV з числовими даними")

    data = np.array(data, dtype="float32")

    # Переконаємося, що масив має 2D форму
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)  # 1 зразок, shape (1,10)

    reconstruction = anomaly_model.predict(data)

    error = np.mean((data - reconstruction) ** 2)

    if error > THRESHOLD:
        return "Аномалія знайдена"
    else:
        return "Аномалії немає"
