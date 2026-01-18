#Django part2

Проєкт містить веб-додаток на Django з 5 задачами машинного навчання:
розпізнавання цифр, відео, аудіо, спектрів сигналів та аномалій.

Структура задач
Task 1	Розпізнавання рукописних цифр (MNIST)
Task 2	Класифікація відео (MobileNetV2)
Task 3	Класифікація аудіо
Task 4	Аналіз спектру сигналу
Task 5	Детекція аномалій

Вимоги
Python 3.10+
Django
TensorFlow / Keras
NumPy
OpenCV
Librosa
Matplotlib
Pillow
SciPy

Встановлення:
pip install django tensorflow numpy opencv-python librosa matplotlib pillow scipy

Запуск проєкту
python manage.py runserver

Моделі
Усі моделі збережені у папці:
tf_tasks/models/

Модель	Призначення
mnist_model.h5	Task 1
my_video_model.keras	Task 2
my_audio_model.keras	Task 3
my_spectrum_model.keras	Task 4
my_anomaly_model.keras	Task 5

Примітки
Для Task 4 спектр повинен будуватися з тими ж параметрами, що і під час тренування.
Якщо модель не виявляє аномалії — необхідно перетренувати її з реальними аномальними прикладами.
Для Task 5 вхідні дані мають мати ту ж форму, що й під час навчання (наприклад, 10 значень у рядку).
Усі HTML-файли знаходяться в tf_tasks/templates/tf_tasks/
