import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

# Data
NL = np.array([1, 2, 2, 2, 5, 6, 6, 6, 2, 0])
NT = np.array([91, 65, 45, 36, 66, 61, 63, 42, 61, 69])

# Fungsi untuk menghitung galat RMS
def rms_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Model Eksponensial
def exponential_model(x, a, b):
    return a * np.exp(b * x)

params_exp, _ = curve_fit(exponential_model, NL, NT)
y_pred_exp = exponential_model(NL, *params_exp)

# Plot hasil regresi eksponensial
plt.scatter(NL, NT, color='blue', label='Data Asli')
plt.plot(NL, y_pred_exp, color='red', label='Regresi Eksponensial')
plt.xlabel('Jumlah Latihan Soal (NL)')
plt.ylabel('Nilai Ujian (NT)')
plt.legend()
plt.title('Model Eksponensial')
plt.show()

# Hitung RMS Error untuk model eksponensial
rms_exp = rms_error(NT, y_pred_exp)
print('RMS Error (Eksponensial):', rms_exp)
