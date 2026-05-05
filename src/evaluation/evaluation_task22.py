import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf

from src.preprocessing.dataloader_task22 import load_and_preprocess

N = 3
N_FEATURES = 5
OPEN_COL = 0

X_train, y_train, X_val, y_val, X_test, y_test, scaler = load_and_preprocess(time_step=60, n=N)

model = tf.keras.models.load_model(f"stock_model_task22_HPG_N{N}.h5")

predictions = model.predict(X_test).flatten()

def inverse_open(scaled_vals):
    dummy = np.zeros((len(scaled_vals), N_FEATURES))
    dummy[:, OPEN_COL] = scaled_vals
    return scaler.inverse_transform(dummy)[:, OPEN_COL]

y_test_real = inverse_open(y_test)
y_pred_real = inverse_open(predictions)

mae  = mean_absolute_error(y_test_real, y_pred_real)
rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
mape = np.mean(np.abs((y_test_real - y_pred_real) / y_test_real)) * 100

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAPE : {mape:.2f}%")

plt.figure(figsize=(14, 5))
plt.plot(y_test_real, label='Actual', color='blue')
plt.plot(y_pred_real, label='Predicted', color='red', linestyle='--')
plt.title(f'Task 2.2 — Vietnam HPG Day+{N} Forecast')
plt.xlabel('Days')
plt.ylabel('Open Price (VND)')
plt.legend()
plt.tight_layout()
plt.show()