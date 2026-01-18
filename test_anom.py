import numpy as np

test_data = np.array([10.1, 9.8, 10.5, 10.2, 9.9, 10.3, 10.0, 10.4, 10.2, 9.7])

np.savetxt("test_anomaly_-.csv", test_data, delimiter=",")

print("Файл test_anomaly_-.csv готовий для Task5")