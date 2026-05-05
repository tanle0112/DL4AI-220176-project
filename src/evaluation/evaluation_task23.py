import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf

from src.preprocessing.dataloader_task23 import load_and_preprocess

K = 7
N_FEATURES = 5
OPEN_COL = 0

X_train, y_train, X_val, y_val, X_test, y_test, scaler = load_and_preprocess(
    time_step=60, forecast_day=K
)

model = tf.keras.models.load_model(f"stock_model_task23_HPG_day{K}.h5")

predictions = model.predict(X_test)

def inverse_open(scaled_vals_2d):
    result = np.zeros_like(scaled_vals_2d)
    dummy = np.zeros((scaled_vals_2d.shape[0], N_FEATURES))
    for d in range(K):
        dummy[:, OPEN_COL] = scaled_vals_2d[:, d]
        result[:, d] = scaler.inverse_transform(dummy)[:, OPEN_COL]
    return result

y_test_real = inverse_open(y_test)
y_pred_real = inverse_open(predictions)

mae  = mean_absolute_error(y_test_real.flatten(), y_pred_real.flatten())
rmse = np.sqrt(mean_squared_error(y_test_real.flatten(), y_pred_real.flatten()))
mape = np.mean(np.abs((y_test_real - y_pred_real) / y_test_real)) * 100

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAPE : {mape:.2f}%")

plt.figure(figsize=(14, 5))
plt.plot(y_test_real[:, 0], label='Actual Day+1', color='blue')
plt.plot(y_pred_real[:, 0], label='Predicted Day+1', color='red', linestyle='--')
plt.title(f'Task 2.3 — Vietnam HPG {K}-Day Forecast')
plt.xlabel('Days')
plt.ylabel('Open Price (VND)')
plt.legend()
plt.tight_layout()
plt.show()