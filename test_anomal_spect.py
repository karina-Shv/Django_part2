import numpy as np
from scipy.io.wavfile import write

sr = 22050
T = 2
t = np.linspace(0, T, int(T*sr), endpoint=False)

# Нормальна суміш двох частот
normal_signal = 0.5*np.sin(2*np.pi*440*t) + 0.3*np.sin(2*np.pi*880*t)

# Додаємо дуже яскраву аномалію: 5000 Гц високої амплітуди
anomaly_signal = 0.8*np.sin(2*np.pi*5000*t)

# Додаємо аномалію лише на 0.5 секунд у середині
signal_with_anomaly = normal_signal.copy()
start_idx = int(0.75*sr)
end_idx = start_idx + int(0.5*sr)
signal_with_anomaly[start_idx:end_idx] += anomaly_signal[start_idx:end_idx]

# Нормалізація
signal_with_anomaly = signal_with_anomaly / np.max(np.abs(signal_with_anomaly))

write("anomalous_test_strong.wav", sr, (signal_with_anomaly*32767).astype(np.int16))
print("Файл anomalous_test_strong.wav готовий для Task4")
