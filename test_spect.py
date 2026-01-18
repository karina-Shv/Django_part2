import numpy as np
from scipy.io.wavfile import write

sr = 22050
T = 2
t = np.linspace(0, T, int(T*sr), endpoint=False)

signal = 0.5*np.sin(2*np.pi*440*t) + 0.3*np.sin(2*np.pi*880*t)

write("test_spectrum.wav", sr, (signal*32767).astype(np.int16))

print("Файл test_spectrum.wav готовий для Task4")