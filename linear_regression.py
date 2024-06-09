import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Data
NL = np.array([1, 2, 2, 2, 5, 6, 6, 6, 2, 0])
NT = np.array([91, 65, 45, 36, 66, 61, 63, 42, 61, 69])

# Fungsi untuk menghitung galat RMS
def rms_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Model Linear
X = NL.reshape(-1, 1)
linear_model = LinearRegression()
linear_model.fit(X, NT)
y_pred_linear = linear_model.predict(X)

# Plot hasil regresi linear
plt.scatter(NL, NT, color='blue', label='Data Asli')
plt.plot(NL, y_pred_linear, color='red', label='Regresi Linear')
plt.xlabel('Jumlah Latihan Soal (NL)')
plt.ylabel('Nilai Ujian (NT)')
plt.legend()
plt.title('Model Linear')
plt.show()

# Hitung RMS Error untuk model linear
rms_linear = rms_error(NT, y_pred_linear)
print('RMS Error (Linear):', rms_linear)
